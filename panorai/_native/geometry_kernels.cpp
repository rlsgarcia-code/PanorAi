#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

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

PyObject* cubemap_to_equirectangular(PyObject*, PyObject* args) {
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
    PyObject* result = PyByteArray_FromStringAndSize(nullptr, output_bytes);
    if (result == nullptr) {
        return nullptr;
    }
    void* output = PyByteArray_AS_STRING(result);
    Py_BEGIN_ALLOW_THREADS
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
    Py_END_ALLOW_THREADS
    return result;
}

PyMethodDef methods[] = {
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
