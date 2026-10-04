#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr Py_ssize_t kFaceCount = 6;

struct Buffer {
    Py_buffer view{};
    bool acquired = false;

    ~Buffer() {
        if (acquired) {
            PyBuffer_Release(&view);
        }
    }
};

class OwnedPyObject {
public:
    explicit OwnedPyObject(PyObject* object = nullptr) noexcept : object_(object) {}

    OwnedPyObject(const OwnedPyObject&) = delete;
    OwnedPyObject& operator=(const OwnedPyObject&) = delete;

    ~OwnedPyObject() {
        Py_XDECREF(object_);
    }

    PyObject* get() const noexcept {
        return object_;
    }

    PyObject* release() noexcept {
        PyObject* object = object_;
        object_ = nullptr;
        return object;
    }

private:
    PyObject* object_;
};

class AllowThreads {
public:
    AllowThreads() : thread_state_(PyEval_SaveThread()) {}

    AllowThreads(const AllowThreads&) = delete;
    AllowThreads& operator=(const AllowThreads&) = delete;

    ~AllowThreads() {
        PyEval_RestoreThread(thread_state_);
    }

private:
    PyThreadState* thread_state_;
};

class ThreadGroup {
public:
    explicit ThreadGroup(unsigned int capacity) {
        threads_.reserve(capacity);
    }

    ThreadGroup(const ThreadGroup&) = delete;
    ThreadGroup& operator=(const ThreadGroup&) = delete;

    ~ThreadGroup() {
        join();
    }

    template <typename Function>
    void start(Function&& function) {
        threads_.emplace_back(std::forward<Function>(function));
    }

    void join() noexcept {
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

private:
    std::vector<std::thread> threads_;
};

template <typename Function>
PyObject* translate_cpp_exceptions(Function&& function) noexcept {
    try {
        return function();
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
    } catch (const std::length_error& error) {
        PyErr_Format(PyExc_OverflowError, "native geometry allocation failed: %s", error.what());
    } catch (const std::system_error& error) {
        PyErr_Format(PyExc_RuntimeError, "native geometry worker failed: %s", error.what());
    } catch (const std::exception& error) {
        PyErr_Format(PyExc_RuntimeError, "native geometry failure: %s", error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown native geometry failure");
    }
    return nullptr;
}

bool acquire_contiguous_buffer(
    PyObject* object,
    Buffer& buffer,
    int dimensions,
    const char* name) {
    if (PyObject_GetBuffer(
            object,
            &buffer.view,
            PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return false;
    }
    buffer.acquired = true;
    if (buffer.view.ndim != dimensions) {
        PyErr_Format(PyExc_ValueError, "%s must have %d dimensions", name, dimensions);
        return false;
    }
    if (!PyBuffer_IsContiguous(&buffer.view, 'C')) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return false;
    }
    return true;
}

bool is_native_format(const Py_buffer& view, char code) {
    if (view.format == nullptr) {
        return false;
    }
    const char* format = view.format;
    while (*format == '@' || *format == '=') {
        ++format;
    }
    return format[0] == code && format[1] == '\0';
}

bool is_int64_format(const Py_buffer& view) {
    return view.itemsize == static_cast<Py_ssize_t>(sizeof(std::int64_t))
        && (is_native_format(view, 'q') || is_native_format(view, 'l'));
}

bool is_float_format(const Py_buffer& view, char& format) {
    format = is_native_format(view, 'f') ? 'f' : (is_native_format(view, 'd') ? 'd' : '\0');
    return (format == 'f' && view.itemsize == static_cast<Py_ssize_t>(sizeof(float)))
        || (format == 'd' && view.itemsize == static_cast<Py_ssize_t>(sizeof(double)));
}

Py_ssize_t positive_mod(Py_ssize_t value, Py_ssize_t modulus) {
    const Py_ssize_t result = value % modulus;
    return result < 0 ? result + modulus : result;
}

unsigned int worker_count(Py_ssize_t work_items) {
    if (work_items < 32768) {
        return 1;
    }
    const unsigned int available = std::max(1U, std::thread::hardware_concurrency());
    return std::min<unsigned int>(available, 8U);
}

template <typename T>
void sample_pixel(
    const T* source,
    Py_ssize_t height,
    Py_ssize_t width,
    Py_ssize_t channels,
    double x,
    double y,
    bool wrap_x,
    bool bilinear,
    T* destination) {
    if (!bilinear) {
        Py_ssize_t sample_x = static_cast<Py_ssize_t>(std::floor(x + 0.5));
        sample_x = wrap_x ? positive_mod(sample_x, width)
                          : std::clamp<Py_ssize_t>(sample_x, 0, width - 1);
        const Py_ssize_t sample_y = std::clamp<Py_ssize_t>(
            static_cast<Py_ssize_t>(std::floor(y + 0.5)), 0, height - 1);
        const Py_ssize_t source_offset = (sample_y * width + sample_x) * channels;
        for (Py_ssize_t channel = 0; channel < channels; ++channel) {
            destination[channel] = source[source_offset + channel];
        }
        return;
    }

    const auto x0_raw = static_cast<Py_ssize_t>(std::floor(x));
    const auto y0_raw = static_cast<Py_ssize_t>(std::floor(y));
    const Py_ssize_t x0 = wrap_x ? positive_mod(x0_raw, width)
                                 : std::clamp<Py_ssize_t>(x0_raw, 0, width - 1);
    const Py_ssize_t x1 = wrap_x ? positive_mod(x0_raw + 1, width)
                                 : std::clamp<Py_ssize_t>(x0_raw + 1, 0, width - 1);
    const Py_ssize_t y0 = std::clamp<Py_ssize_t>(y0_raw, 0, height - 1);
    const Py_ssize_t y1 = std::clamp<Py_ssize_t>(y0_raw + 1, 0, height - 1);
    const double wx = x - static_cast<double>(x0_raw);
    const double wy = y - static_cast<double>(y0_raw);
    const Py_ssize_t source_00 = (y0 * width + x0) * channels;
    const Py_ssize_t source_01 = (y0 * width + x1) * channels;
    const Py_ssize_t source_10 = (y1 * width + x0) * channels;
    const Py_ssize_t source_11 = (y1 * width + x1) * channels;
    for (Py_ssize_t channel = 0; channel < channels; ++channel) {
        const double top = static_cast<double>(source[source_00 + channel]) * (1.0 - wx)
            + static_cast<double>(source[source_01 + channel]) * wx;
        const double bottom = static_cast<double>(source[source_10 + channel]) * (1.0 - wx)
            + static_cast<double>(source[source_11 + channel]) * wx;
        destination[channel] = static_cast<T>(top * (1.0 - wy) + bottom * wy);
    }
}

template <typename T>
void sample_gnomonic_forward_face(
    const T* source,
    Py_ssize_t source_height,
    Py_ssize_t source_width,
    Py_ssize_t channels,
    const double* map_xy,
    Py_ssize_t pixels,
    bool bilinear,
    T* output) {
    for (Py_ssize_t pixel = 0; pixel < pixels; ++pixel) {
        sample_pixel(
            source,
            source_height,
            source_width,
            channels,
            map_xy[pixel * 2],
            map_xy[pixel * 2 + 1],
            true,
            bilinear,
            output + pixel * channels);
    }
}

PyObject* equirectangular_to_gnomonic_batch_impl(PyObject* args) {
    PyObject* image_object = nullptr;
    PyObject* plans_object = nullptr;
    int interpolation = 0;
    if (!PyArg_ParseTuple(
            args,
            "OOi:equirectangular_to_gnomonic_batch",
            &image_object,
            &plans_object,
            &interpolation)) {
        return nullptr;
    }
    if (interpolation != 0 && interpolation != 1) {
        PyErr_SetString(PyExc_ValueError, "interpolation must be 0 or 1");
        return nullptr;
    }
    if (!PyTuple_Check(plans_object) || PyTuple_GET_SIZE(plans_object) <= 0) {
        PyErr_SetString(PyExc_ValueError, "plans must be a non-empty tuple");
        return nullptr;
    }

    Buffer image;
    if (PyObject_GetBuffer(
            image_object,
            &image.view,
            PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return nullptr;
    }
    image.acquired = true;
    char value_format = '\0';
    if ((image.view.ndim != 2 && image.view.ndim != 3)
        || !PyBuffer_IsContiguous(&image.view, 'C')) {
        PyErr_SetString(PyExc_ValueError, "image must be a C-contiguous HW or HWC array");
        return nullptr;
    }
    if (!is_float_format(image.view, value_format)) {
        PyErr_SetString(PyExc_TypeError, "image must use native float32 or float64");
        return nullptr;
    }
    const Py_ssize_t source_height = image.view.shape[0];
    const Py_ssize_t source_width = image.view.shape[1];
    const Py_ssize_t channels = image.view.ndim == 2 ? 1 : image.view.shape[2];
    if (source_height <= 0 || source_width <= 0 || channels <= 0) {
        PyErr_SetString(PyExc_ValueError, "image dimensions must be positive");
        return nullptr;
    }

    const Py_ssize_t face_count = PyTuple_GET_SIZE(plans_object);
    std::vector<Buffer> maps(static_cast<std::size_t>(face_count));
    std::vector<Py_ssize_t> face_pixels(static_cast<std::size_t>(face_count));
    std::vector<void*> output_buffers(static_cast<std::size_t>(face_count));
    OwnedPyObject result(PyTuple_New(face_count));
    if (result.get() == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t face_index = 0; face_index < face_count; ++face_index) {
        PyObject* plan = PyTuple_GET_ITEM(plans_object, face_index);
        if (!acquire_contiguous_buffer(plan, maps[face_index], 3, "pixel_map")) {
            return nullptr;
        }
        const Py_buffer& map_view = maps[face_index].view;
        if (!is_native_format(map_view, 'd')
            || map_view.itemsize != sizeof(double)) {
            PyErr_SetString(PyExc_TypeError, "pixel maps must use native float64");
            return nullptr;
        }
        if (map_view.shape[0] <= 0 || map_view.shape[1] <= 0
            || map_view.shape[2] != 2
            || map_view.shape[0] > PY_SSIZE_T_MAX / map_view.shape[1]) {
            PyErr_SetString(PyExc_ValueError, "pixel-map shape must be (H, W, 2)");
            return nullptr;
        }
        const Py_ssize_t pixels = map_view.shape[0] * map_view.shape[1];
        if (channels > PY_SSIZE_T_MAX / pixels
            || image.view.itemsize > PY_SSIZE_T_MAX / (pixels * channels)) {
            PyErr_SetString(PyExc_OverflowError, "face output is too large");
            return nullptr;
        }
        const auto* map_values = static_cast<const double*>(map_view.buf);
        for (Py_ssize_t pixel = 0; pixel < pixels; ++pixel) {
            if (!std::isfinite(map_values[pixel * 2])
                || !std::isfinite(map_values[pixel * 2 + 1])) {
                PyErr_SetString(PyExc_ValueError, "plan contains a non-finite coordinate");
                return nullptr;
            }
        }
        PyObject* output = PyByteArray_FromStringAndSize(
            nullptr, pixels * channels * image.view.itemsize);
        if (output == nullptr) {
            return nullptr;
        }
        PyTuple_SET_ITEM(result.get(), face_index, output);
        output_buffers[face_index] = PyByteArray_AS_STRING(output);
        face_pixels[face_index] = pixels;
    }

    const auto run_face = [&](Py_ssize_t face_index) {
        if (value_format == 'f') {
            sample_gnomonic_forward_face<float>(
                static_cast<const float*>(image.view.buf),
                source_height,
                source_width,
                channels,
                static_cast<const double*>(maps[face_index].view.buf),
                face_pixels[face_index],
                interpolation == 1,
                static_cast<float*>(output_buffers[face_index]));
        } else {
            sample_gnomonic_forward_face<double>(
                static_cast<const double*>(image.view.buf),
                source_height,
                source_width,
                channels,
                static_cast<const double*>(maps[face_index].view.buf),
                face_pixels[face_index],
                interpolation == 1,
                static_cast<double*>(output_buffers[face_index]));
        }
    };

    const unsigned int workers = worker_count(face_count * face_pixels.front());
    if (workers == 1 || face_count == 1) {
        AllowThreads allow_threads;
        for (Py_ssize_t face_index = 0; face_index < face_count; ++face_index) {
            run_face(face_index);
        }
    } else {
        std::atomic<Py_ssize_t> next_face{0};
        const unsigned int actual_workers = std::min<unsigned int>(
            workers, static_cast<unsigned int>(face_count));
        ThreadGroup threads(actual_workers);
        {
            AllowThreads allow_threads;
            for (unsigned int worker = 0; worker < actual_workers; ++worker) {
                threads.start([&]() noexcept {
                    while (true) {
                        const Py_ssize_t face_index = next_face.fetch_add(1);
                        if (face_index >= face_count) {
                            break;
                        }
                        run_face(face_index);
                    }
                });
            }
            threads.join();
        }
    }
    return result.release();
}

PyObject* equirectangular_to_gnomonic_batch(PyObject*, PyObject* args) {
    return translate_cpp_exceptions(
        [&]() { return equirectangular_to_gnomonic_batch_impl(args); });
}

template <typename T>
T sample_mask_bilinear(
    const bool* source,
    Py_ssize_t height,
    Py_ssize_t width,
    double x,
    double y) {
    const auto x0_raw = static_cast<Py_ssize_t>(std::floor(x));
    const auto y0_raw = static_cast<Py_ssize_t>(std::floor(y));
    const Py_ssize_t x0 = std::clamp<Py_ssize_t>(x0_raw, 0, width - 1);
    const Py_ssize_t x1 = std::clamp<Py_ssize_t>(x0_raw + 1, 0, width - 1);
    const Py_ssize_t y0 = std::clamp<Py_ssize_t>(y0_raw, 0, height - 1);
    const Py_ssize_t y1 = std::clamp<Py_ssize_t>(y0_raw + 1, 0, height - 1);
    const double wx = x - static_cast<double>(x0_raw);
    const double wy = y - static_cast<double>(y0_raw);
    const double top = static_cast<double>(source[y0 * width + x0]) * (1.0 - wx)
        + static_cast<double>(source[y0 * width + x1]) * wx;
    const double bottom = static_cast<double>(source[y1 * width + x0]) * (1.0 - wx)
        + static_cast<double>(source[y1 * width + x1]) * wx;
    return static_cast<T>(top * (1.0 - wy) + bottom * wy);
}

template <typename T>
void gaussian_reconstruct_range(
    const std::vector<Buffer>& faces,
    const std::vector<Buffer>& masks,
    const std::vector<Buffer>& indices,
    const std::vector<Buffer>& maps_x,
    const std::vector<Buffer>& maps_y,
    const std::vector<Buffer>& scores,
    Py_ssize_t channels,
    Py_ssize_t begin_pixel,
    Py_ssize_t end_pixel,
    T* output,
    T* weight_sum,
    unsigned char* valid_output,
    T* sample) noexcept {
    constexpr double tolerance = 32.0 * static_cast<double>(std::numeric_limits<float>::epsilon());
    for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
        const auto* flat_indices =
            static_cast<const std::int64_t*>(indices[face_index].view.buf);
        const Py_ssize_t count = indices[face_index].view.shape[0];
        const auto* first = std::lower_bound(flat_indices, flat_indices + count, begin_pixel);
        const auto* last = std::lower_bound(first, flat_indices + count, end_pixel);
        const auto* map_x = static_cast<const double*>(maps_x[face_index].view.buf);
        const auto* map_y = static_cast<const double*>(maps_y[face_index].view.buf);
        const auto* center_score = static_cast<const double*>(scores[face_index].view.buf);
        const auto* source = static_cast<const T*>(faces[face_index].view.buf);
        const auto* source_mask = static_cast<const bool*>(masks[face_index].view.buf);
        const Py_ssize_t face_height = faces[face_index].view.shape[0];
        const Py_ssize_t face_width = faces[face_index].view.shape[1];
        for (const auto* current = first; current != last; ++current) {
            const Py_ssize_t sample_index = current - flat_indices;
            const Py_ssize_t destination_pixel = *current;
            const double x = map_x[sample_index];
            const double y = map_y[sample_index];
            const T sampled_mask = sample_mask_bilinear<T>(
                source_mask, face_height, face_width, x, y);
            if (static_cast<double>(sampled_mask) < 1.0 - tolerance) {
                continue;
            }
            sample_pixel(
                source,
                face_height,
                face_width,
                channels,
                x,
                y,
                false,
                true,
                sample);
            bool finite = true;
            for (Py_ssize_t channel = 0; channel < channels; ++channel) {
                finite = finite && std::isfinite(static_cast<double>(sample[channel]));
            }
            if (!finite) {
                continue;
            }
            const T weight = static_cast<T>(
                std::exp(6.0 * (center_score[sample_index] - 1.0)));
            const Py_ssize_t destination = destination_pixel * channels;
            for (Py_ssize_t channel = 0; channel < channels; ++channel) {
                output[destination + channel] = static_cast<T>(
                    output[destination + channel]
                    + static_cast<T>(sample[channel] * weight));
            }
            weight_sum[destination_pixel] = static_cast<T>(
                weight_sum[destination_pixel] + weight);
        }
    }
    const T nan = std::numeric_limits<T>::quiet_NaN();
    for (Py_ssize_t pixel = begin_pixel; pixel < end_pixel; ++pixel) {
        const T weight = weight_sum[pixel];
        const Py_ssize_t destination = pixel * channels;
        if (weight > static_cast<T>(0)) {
            for (Py_ssize_t channel = 0; channel < channels; ++channel) {
                output[destination + channel] = static_cast<T>(
                    output[destination + channel] / weight);
            }
            valid_output[pixel] = 1;
        } else {
            for (Py_ssize_t channel = 0; channel < channels; ++channel) {
                output[destination + channel] = nan;
            }
        }
    }
}

template <typename T>
void run_gaussian_reconstruction(
    const std::vector<Buffer>& faces,
    const std::vector<Buffer>& masks,
    const std::vector<Buffer>& indices,
    const std::vector<Buffer>& maps_x,
    const std::vector<Buffer>& maps_y,
    const std::vector<Buffer>& scores,
    Py_ssize_t channels,
    Py_ssize_t output_pixels,
    unsigned int workers,
    T* output,
    unsigned char* valid_output) {
    if (channels > PY_SSIZE_T_MAX / static_cast<Py_ssize_t>(workers)) {
        throw std::length_error("worker scratch is too large");
    }
    std::vector<T> weight_sum(static_cast<std::size_t>(output_pixels), static_cast<T>(0));
    std::vector<T> scratch(
        static_cast<std::size_t>(channels) * static_cast<std::size_t>(workers));
    const auto run = [&](unsigned int worker, Py_ssize_t begin, Py_ssize_t end) noexcept {
        gaussian_reconstruct_range<T>(
            faces,
            masks,
            indices,
            maps_x,
            maps_y,
            scores,
            channels,
            begin,
            end,
            output,
            weight_sum.data(),
            valid_output,
            scratch.data() + static_cast<std::size_t>(worker) * channels);
    };
    if (workers == 1) {
        AllowThreads allow_threads;
        run(0, 0, output_pixels);
        return;
    }
    ThreadGroup threads(workers);
    {
        AllowThreads allow_threads;
        for (unsigned int worker = 0; worker < workers; ++worker) {
            const Py_ssize_t begin = output_pixels * worker / workers;
            const Py_ssize_t end = output_pixels * (worker + 1) / workers;
            threads.start(
                [&, worker, begin, end]() noexcept { run(worker, begin, end); });
        }
        threads.join();
    }
}

PyObject* gnomonic_gaussian_to_equirectangular_impl(PyObject* args) {
    PyObject* faces_object = nullptr;
    PyObject* masks_object = nullptr;
    PyObject* plans_object = nullptr;
    Py_ssize_t output_height = 0;
    Py_ssize_t output_width = 0;
    if (!PyArg_ParseTuple(
            args,
            "OOOnn:gnomonic_gaussian_to_equirectangular",
            &faces_object,
            &masks_object,
            &plans_object,
            &output_height,
            &output_width)) {
        return nullptr;
    }
    if (!PyTuple_Check(faces_object) || !PyTuple_Check(masks_object)
        || !PyTuple_Check(plans_object)) {
        PyErr_SetString(PyExc_TypeError, "faces, masks, and plans must be tuples");
        return nullptr;
    }
    const Py_ssize_t face_count = PyTuple_GET_SIZE(faces_object);
    if (face_count <= 0 || PyTuple_GET_SIZE(masks_object) != face_count
        || PyTuple_GET_SIZE(plans_object) != face_count) {
        PyErr_SetString(PyExc_ValueError, "faces, masks, and plans must have equal non-zero length");
        return nullptr;
    }
    if (output_height <= 0 || output_width <= 0
        || output_height > PY_SSIZE_T_MAX / output_width) {
        PyErr_SetString(PyExc_ValueError, "output dimensions are invalid");
        return nullptr;
    }
    const Py_ssize_t output_pixels = output_height * output_width;
    std::vector<Buffer> faces(static_cast<std::size_t>(face_count));
    std::vector<Buffer> masks(static_cast<std::size_t>(face_count));
    std::vector<Buffer> indices(static_cast<std::size_t>(face_count));
    std::vector<Buffer> maps_x(static_cast<std::size_t>(face_count));
    std::vector<Buffer> maps_y(static_cast<std::size_t>(face_count));
    std::vector<Buffer> scores(static_cast<std::size_t>(face_count));
    Py_ssize_t channels = 1;
    Py_ssize_t itemsize = 0;
    int dimensions = 0;
    char value_format = '\0';
    for (Py_ssize_t face_index = 0; face_index < face_count; ++face_index) {
        PyObject* face_object = PyTuple_GET_ITEM(faces_object, face_index);
        if (PyObject_GetBuffer(
                face_object,
                &faces[face_index].view,
                PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
            return nullptr;
        }
        faces[face_index].acquired = true;
        Py_buffer& face = faces[face_index].view;
        char format = '\0';
        if ((face.ndim != 2 && face.ndim != 3) || !PyBuffer_IsContiguous(&face, 'C')) {
            PyErr_SetString(PyExc_ValueError, "faces must be C-contiguous HW or HWC arrays");
            return nullptr;
        }
        if (!is_float_format(face, format)) {
            PyErr_SetString(PyExc_TypeError, "faces must use native float32 or float64");
            return nullptr;
        }
        const Py_ssize_t face_channels = face.ndim == 2 ? 1 : face.shape[2];
        if (face.shape[0] <= 0 || face.shape[1] <= 0 || face_channels <= 0) {
            PyErr_SetString(PyExc_ValueError, "face dimensions must be positive");
            return nullptr;
        }
        if (face_index == 0) {
            channels = face_channels;
            itemsize = face.itemsize;
            dimensions = face.ndim;
            value_format = format;
        } else if (
            face_channels != channels || face.itemsize != itemsize
            || face.ndim != dimensions || format != value_format) {
            PyErr_SetString(PyExc_ValueError, "all faces must share layout, channels, and dtype");
            return nullptr;
        }

        if (!acquire_contiguous_buffer(
                PyTuple_GET_ITEM(masks_object, face_index), masks[face_index], 2, "mask")) {
            return nullptr;
        }
        const Py_buffer& mask = masks[face_index].view;
        if (!is_native_format(mask, '?') || mask.itemsize != sizeof(bool)
            || mask.shape[0] != face.shape[0] || mask.shape[1] != face.shape[1]) {
            PyErr_SetString(PyExc_TypeError, "masks must be boolean arrays matching their faces");
            return nullptr;
        }

        PyObject* plan = PyTuple_GET_ITEM(plans_object, face_index);
        if (!PyTuple_Check(plan) || PyTuple_GET_SIZE(plan) != 4
            || !acquire_contiguous_buffer(
                PyTuple_GET_ITEM(plan, 0), indices[face_index], 1, "flat_indices")
            || !acquire_contiguous_buffer(
                PyTuple_GET_ITEM(plan, 1), maps_x[face_index], 1, "map_x")
            || !acquire_contiguous_buffer(
                PyTuple_GET_ITEM(plan, 2), maps_y[face_index], 1, "map_y")
            || !acquire_contiguous_buffer(
                PyTuple_GET_ITEM(plan, 3), scores[face_index], 1, "center_score")) {
            return nullptr;
        }
        const Py_buffer& index_view = indices[face_index].view;
        const Py_buffer& x_view = maps_x[face_index].view;
        const Py_buffer& y_view = maps_y[face_index].view;
        const Py_buffer& score_view = scores[face_index].view;
        if (!is_int64_format(index_view)) {
            PyErr_SetString(PyExc_TypeError, "flat_indices must use native int64");
            return nullptr;
        }
        if (!is_native_format(x_view, 'd') || x_view.itemsize != sizeof(double)
            || !is_native_format(y_view, 'd') || y_view.itemsize != sizeof(double)
            || !is_native_format(score_view, 'd') || score_view.itemsize != sizeof(double)) {
            PyErr_SetString(PyExc_TypeError, "map and score arrays must use native float64");
            return nullptr;
        }
        const Py_ssize_t count = index_view.shape[0];
        if (x_view.shape[0] != count || y_view.shape[0] != count
            || score_view.shape[0] != count) {
            PyErr_SetString(PyExc_ValueError, "plan array lengths are inconsistent");
            return nullptr;
        }
        const auto* flat = static_cast<const std::int64_t*>(index_view.buf);
        const auto* x_values = static_cast<const double*>(x_view.buf);
        const auto* y_values = static_cast<const double*>(y_view.buf);
        const auto* score_values = static_cast<const double*>(score_view.buf);
        std::int64_t previous = -1;
        for (Py_ssize_t sample_index = 0; sample_index < count; ++sample_index) {
            if (flat[sample_index] < 0 || flat[sample_index] >= output_pixels
                || flat[sample_index] <= previous || !std::isfinite(x_values[sample_index])
                || !std::isfinite(y_values[sample_index])
                || !std::isfinite(score_values[sample_index])) {
                PyErr_SetString(PyExc_ValueError, "plan contains an invalid or unsorted sample");
                return nullptr;
            }
            previous = flat[sample_index];
        }
    }
    if (channels > PY_SSIZE_T_MAX / output_pixels
        || itemsize > PY_SSIZE_T_MAX / (output_pixels * channels)) {
        PyErr_SetString(PyExc_OverflowError, "output is too large");
        return nullptr;
    }
    OwnedPyObject output_object(PyByteArray_FromStringAndSize(
        nullptr, output_pixels * channels * itemsize));
    OwnedPyObject valid_object(
        PyByteArray_FromStringAndSize(nullptr, output_pixels));
    if (output_object.get() == nullptr || valid_object.get() == nullptr) {
        return nullptr;
    }
    void* output = PyByteArray_AS_STRING(output_object.get());
    auto* valid_output = reinterpret_cast<unsigned char*>(
        PyByteArray_AS_STRING(valid_object.get()));
    std::memset(output, 0, static_cast<std::size_t>(output_pixels * channels * itemsize));
    std::memset(valid_output, 0, static_cast<std::size_t>(output_pixels));

    const unsigned int workers = worker_count(output_pixels);
    if (value_format == 'f') {
        run_gaussian_reconstruction<float>(
            faces,
            masks,
            indices,
            maps_x,
            maps_y,
            scores,
            channels,
            output_pixels,
            workers,
            static_cast<float*>(output),
            valid_output);
    } else {
        run_gaussian_reconstruction<double>(
            faces,
            masks,
            indices,
            maps_x,
            maps_y,
            scores,
            channels,
            output_pixels,
            workers,
            static_cast<double*>(output),
            valid_output);
    }
    OwnedPyObject result(PyTuple_New(2));
    if (result.get() == nullptr) {
        return nullptr;
    }
    PyTuple_SET_ITEM(result.get(), 0, output_object.release());
    PyTuple_SET_ITEM(result.get(), 1, valid_object.release());
    return result.release();
}

PyObject* gnomonic_gaussian_to_equirectangular(PyObject*, PyObject* args) {
    return translate_cpp_exceptions(
        [&]() { return gnomonic_gaussian_to_equirectangular_impl(args); });
}

template <typename T>
void sample_faces(
    const std::array<Buffer, kFaceCount>& faces,
    const std::array<Buffer, kFaceCount>& indices,
    const std::array<Buffer, kFaceCount>& maps_x,
    const std::array<Buffer, kFaceCount>& maps_y,
    Py_ssize_t face_height,
    Py_ssize_t face_width,
    Py_ssize_t channels,
    T* output) {
    for (Py_ssize_t face_index = 0; face_index < kFaceCount; ++face_index) {
        const T* source = static_cast<const T*>(faces[face_index].view.buf);
        const auto* flat_indices =
            static_cast<const std::int64_t*>(indices[face_index].view.buf);
        const auto* map_x = static_cast<const double*>(maps_x[face_index].view.buf);
        const auto* map_y = static_cast<const double*>(maps_y[face_index].view.buf);
        const Py_ssize_t count = indices[face_index].view.shape[0];
        for (Py_ssize_t sample_index = 0; sample_index < count; ++sample_index) {
            const double x = map_x[sample_index];
            const double y = map_y[sample_index];
            const auto x0_raw = static_cast<Py_ssize_t>(std::floor(x));
            const auto y0_raw = static_cast<Py_ssize_t>(std::floor(y));
            const Py_ssize_t x0 = std::clamp<Py_ssize_t>(x0_raw, 0, face_width - 1);
            const Py_ssize_t x1 =
                std::clamp<Py_ssize_t>(x0_raw + 1, 0, face_width - 1);
            const Py_ssize_t y0 =
                std::clamp<Py_ssize_t>(y0_raw, 0, face_height - 1);
            const Py_ssize_t y1 =
                std::clamp<Py_ssize_t>(y0_raw + 1, 0, face_height - 1);
            const double wx = x - static_cast<double>(x0_raw);
            const double wy = y - static_cast<double>(y0_raw);
            const Py_ssize_t source_00 = (y0 * face_width + x0) * channels;
            const Py_ssize_t source_01 = (y0 * face_width + x1) * channels;
            const Py_ssize_t source_10 = (y1 * face_width + x0) * channels;
            const Py_ssize_t source_11 = (y1 * face_width + x1) * channels;
            const Py_ssize_t destination = flat_indices[sample_index] * channels;
            for (Py_ssize_t channel = 0; channel < channels; ++channel) {
                const double top =
                    static_cast<double>(source[source_00 + channel]) * (1.0 - wx)
                    + static_cast<double>(source[source_01 + channel]) * wx;
                const double bottom =
                    static_cast<double>(source[source_10 + channel]) * (1.0 - wx)
                    + static_cast<double>(source[source_11 + channel]) * wx;
                output[destination + channel] = static_cast<T>(
                    top * (1.0 - wy) + bottom * wy);
            }
        }
    }
}

PyObject* cubemap_to_equirectangular_impl(PyObject* args) {
    PyObject* faces_object = nullptr;
    PyObject* plans_object = nullptr;
    Py_ssize_t output_height = 0;
    Py_ssize_t output_width = 0;
    if (!PyArg_ParseTuple(
            args,
            "OOnn:cubemap_to_equirectangular",
            &faces_object,
            &plans_object,
            &output_height,
            &output_width)) {
        return nullptr;
    }
    if (!PyTuple_Check(faces_object)
        || PyTuple_GET_SIZE(faces_object) != kFaceCount) {
        PyErr_SetString(PyExc_ValueError, "faces must be a tuple of six arrays");
        return nullptr;
    }
    if (!PyTuple_Check(plans_object)
        || PyTuple_GET_SIZE(plans_object) != kFaceCount) {
        PyErr_SetString(PyExc_ValueError, "plans must be a tuple of six triples");
        return nullptr;
    }
    if (output_height <= 0 || output_width <= 0
        || output_height > PY_SSIZE_T_MAX / output_width) {
        PyErr_SetString(PyExc_ValueError, "output dimensions are invalid");
        return nullptr;
    }

    std::array<Buffer, kFaceCount> faces;
    std::array<Buffer, kFaceCount> indices;
    std::array<Buffer, kFaceCount> maps_x;
    std::array<Buffer, kFaceCount> maps_y;
    Py_ssize_t face_height = 0;
    Py_ssize_t face_width = 0;
    Py_ssize_t channels = 1;
    Py_ssize_t itemsize = 0;
    char value_format = '\0';
    Py_ssize_t total_samples = 0;
    const Py_ssize_t output_pixels = output_height * output_width;

    for (Py_ssize_t face_index = 0; face_index < kFaceCount; ++face_index) {
        PyObject* face_object = PyTuple_GET_ITEM(faces_object, face_index);
        if (PyObject_GetBuffer(
                face_object,
                &faces[face_index].view,
                PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
            return nullptr;
        }
        faces[face_index].acquired = true;
        const Py_buffer& face = faces[face_index].view;
        if ((face.ndim != 2 && face.ndim != 3)
            || !PyBuffer_IsContiguous(&face, 'C')) {
            PyErr_SetString(
                PyExc_ValueError,
                "each face must be a C-contiguous HW or HWC array");
            return nullptr;
        }
        const char format =
            is_native_format(face, 'f') ? 'f' : (is_native_format(face, 'd') ? 'd' : '\0');
        if (format == '\0'
            || (format == 'f' && face.itemsize != static_cast<Py_ssize_t>(sizeof(float)))
            || (format == 'd' && face.itemsize != static_cast<Py_ssize_t>(sizeof(double)))) {
            PyErr_SetString(PyExc_TypeError, "faces must use native float32 or float64");
            return nullptr;
        }
        const Py_ssize_t face_channels = face.ndim == 2 ? 1 : face.shape[2];
        if (face.shape[0] <= 0 || face.shape[1] <= 0 || face_channels <= 0) {
            PyErr_SetString(PyExc_ValueError, "face dimensions must be positive");
            return nullptr;
        }
        if (face_index == 0) {
            face_height = face.shape[0];
            face_width = face.shape[1];
            channels = face_channels;
            itemsize = face.itemsize;
            value_format = format;
        } else if (
            face.shape[0] != face_height || face.shape[1] != face_width
            || face_channels != channels || face.itemsize != itemsize
            || format != value_format || face.ndim != faces[0].view.ndim) {
            PyErr_SetString(
                PyExc_ValueError,
                "all faces must have identical shape, layout, and dtype");
            return nullptr;
        }

        PyObject* plan = PyTuple_GET_ITEM(plans_object, face_index);
        if (!PyTuple_Check(plan) || PyTuple_GET_SIZE(plan) != 3) {
            PyErr_SetString(PyExc_ValueError, "each plan must be an indices/x/y triple");
            return nullptr;
        }
        if (!acquire_contiguous_buffer(
                PyTuple_GET_ITEM(plan, 0), indices[face_index], 1, "flat_indices")
            || !acquire_contiguous_buffer(
                PyTuple_GET_ITEM(plan, 1), maps_x[face_index], 1, "map_x")
            || !acquire_contiguous_buffer(
                PyTuple_GET_ITEM(plan, 2), maps_y[face_index], 1, "map_y")) {
            return nullptr;
        }
        const Py_buffer& index_view = indices[face_index].view;
        const Py_buffer& x_view = maps_x[face_index].view;
        const Py_buffer& y_view = maps_y[face_index].view;
        if (!is_int64_format(index_view)) {
            PyErr_SetString(PyExc_TypeError, "flat_indices must use native int64");
            return nullptr;
        }
        if (!is_native_format(x_view, 'd') || x_view.itemsize != sizeof(double)
            || !is_native_format(y_view, 'd') || y_view.itemsize != sizeof(double)) {
            PyErr_SetString(PyExc_TypeError, "map arrays must use native float64");
            return nullptr;
        }
        const Py_ssize_t count = index_view.shape[0];
        if (x_view.shape[0] != count || y_view.shape[0] != count
            || count > output_pixels - total_samples) {
            PyErr_SetString(PyExc_ValueError, "plan array lengths are inconsistent");
            return nullptr;
        }
        const auto* flat_indices = static_cast<const std::int64_t*>(index_view.buf);
        const auto* map_x = static_cast<const double*>(x_view.buf);
        const auto* map_y = static_cast<const double*>(y_view.buf);
        for (Py_ssize_t sample_index = 0; sample_index < count; ++sample_index) {
            if (flat_indices[sample_index] < 0
                || flat_indices[sample_index] >= output_pixels
                || !std::isfinite(map_x[sample_index])
                || !std::isfinite(map_y[sample_index])) {
                PyErr_SetString(PyExc_ValueError, "plan contains an invalid sample");
                return nullptr;
            }
        }
        total_samples += count;
    }
    if (total_samples != output_pixels
        || channels > PY_SSIZE_T_MAX / output_pixels
        || itemsize > PY_SSIZE_T_MAX / (output_pixels * channels)) {
        PyErr_SetString(PyExc_ValueError, "plans do not cover the output safely");
        return nullptr;
    }

    const Py_ssize_t output_bytes = output_pixels * channels * itemsize;
    OwnedPyObject result(PyByteArray_FromStringAndSize(nullptr, output_bytes));
    if (result.get() == nullptr) {
        return nullptr;
    }
    void* output = PyByteArray_AS_STRING(result.get());
    {
        AllowThreads allow_threads;
        if (value_format == 'f') {
            sample_faces<float>(
                faces,
                indices,
                maps_x,
                maps_y,
                face_height,
                face_width,
                channels,
                static_cast<float*>(output));
        } else {
            sample_faces<double>(
                faces,
                indices,
                maps_x,
                maps_y,
                face_height,
                face_width,
                channels,
                static_cast<double*>(output));
        }
    }
    return result.release();
}

PyObject* cubemap_to_equirectangular(PyObject*, PyObject* args) {
    return translate_cpp_exceptions(
        [&]() { return cubemap_to_equirectangular_impl(args); });
}

PyMethodDef methods[] = {
    {"equirectangular_to_gnomonic_batch",
     equirectangular_to_gnomonic_batch,
     METH_VARARGS,
     "Fused bilinear/nearest ERP-to-arbitrary-N gnomonic sampling."},
    {"gnomonic_gaussian_to_equirectangular",
     gnomonic_gaussian_to_equirectangular,
     METH_VARARGS,
     "Fused selective Gaussian arbitrary-N gnomonic-to-ERP reconstruction."},
    {"cubemap_to_equirectangular",
     cubemap_to_equirectangular,
     METH_VARARGS,
     "Fused selective bilinear cubemap-to-ERP sampling."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_geometry",
    "First-party PanorAi geometry kernels.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__geometry() {
    return PyModule_Create(&module);
}
