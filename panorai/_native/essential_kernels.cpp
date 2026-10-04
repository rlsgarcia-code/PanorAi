#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

using Polynomial = std::array<double, 20>;

constexpr std::array<std::array<int, 3>, 20> kMonomials{{
    {{3, 0, 0}}, {{2, 1, 0}}, {{1, 2, 0}}, {{0, 3, 0}},
    {{2, 0, 1}}, {{1, 1, 1}}, {{0, 2, 1}}, {{1, 0, 2}},
    {{0, 1, 2}}, {{0, 0, 3}}, {{2, 0, 0}}, {{1, 1, 0}},
    {{0, 2, 0}}, {{1, 0, 1}}, {{0, 1, 1}}, {{0, 0, 2}},
    {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}, {{0, 0, 0}},
}};

int monomial_index(const std::array<int, 3>& exponent) {
    for (int index = 0; index < static_cast<int>(kMonomials.size()); ++index) {
        if (kMonomials[index] == exponent) {
            return index;
        }
    }
    return -1;
}

Polynomial zero_polynomial() {
    Polynomial result{};
    result.fill(0.0);
    return result;
}

Polynomial add(const Polynomial& first, const Polynomial& second) {
    Polynomial result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = first[index] + second[index];
    }
    return result;
}

Polynomial subtract(const Polynomial& first, const Polynomial& second) {
    Polynomial result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = first[index] - second[index];
    }
    return result;
}

Polynomial scale(const Polynomial& polynomial, double value) {
    Polynomial result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = value * polynomial[index];
    }
    return result;
}

Polynomial multiply(const Polynomial& first, const Polynomial& second) {
    Polynomial result = zero_polynomial();
    for (int first_index = 0; first_index < 20; ++first_index) {
        if (first[first_index] == 0.0) {
            continue;
        }
        for (int second_index = 0; second_index < 20; ++second_index) {
            if (second[second_index] == 0.0) {
                continue;
            }
            const auto& a = kMonomials[first_index];
            const auto& b = kMonomials[second_index];
            std::array<int, 3> exponent{{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
            if (exponent[0] + exponent[1] + exponent[2] > 3) {
                continue;
            }
            const int index = monomial_index(exponent);
            if (index >= 0) {
                result[index] += first[first_index] * second[second_index];
            }
        }
    }
    return result;
}

using PolyMatrix3 = std::array<std::array<Polynomial, 3>, 3>;

PolyMatrix3 matrix_multiply(const PolyMatrix3& first, const PolyMatrix3& second) {
    PolyMatrix3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            Polynomial value = zero_polynomial();
            for (int inner = 0; inner < 3; ++inner) {
                value = add(value, multiply(first[row][inner], second[inner][column]));
            }
            result[row][column] = value;
        }
    }
    return result;
}

PolyMatrix3 transpose(const PolyMatrix3& matrix) {
    PolyMatrix3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row][column] = matrix[column][row];
        }
    }
    return result;
}

Polynomial determinant(const PolyMatrix3& matrix) {
    const Polynomial positive = add(
        add(
            multiply(multiply(matrix[0][0], matrix[1][1]), matrix[2][2]),
            multiply(multiply(matrix[0][1], matrix[1][2]), matrix[2][0])),
        multiply(multiply(matrix[0][2], matrix[1][0]), matrix[2][1]));
    const Polynomial negative = add(
        add(
            multiply(multiply(matrix[0][2], matrix[1][1]), matrix[2][0]),
            multiply(multiply(matrix[0][1], matrix[1][0]), matrix[2][2])),
        multiply(multiply(matrix[0][0], matrix[1][2]), matrix[2][1]));
    return subtract(positive, negative);
}

struct Buffer {
    Py_buffer view{};
    bool acquired = false;

    ~Buffer() {
        if (acquired) {
            PyBuffer_Release(&view);
        }
    }
};

bool acquire_double_buffer(
    PyObject* object,
    Buffer& buffer,
    int dimensions,
    const Py_ssize_t* shape,
    const char* name) {
    if (PyObject_GetBuffer(object, &buffer.view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return false;
    }
    buffer.acquired = true;
    if (buffer.view.ndim != dimensions || buffer.view.itemsize != sizeof(double) ||
        buffer.view.format == nullptr || std::strcmp(buffer.view.format, "d") != 0) {
        PyErr_Format(PyExc_ValueError, "%s must be a float64 array with %d dimensions", name, dimensions);
        return false;
    }
    for (int axis = 0; axis < dimensions; ++axis) {
        if (buffer.view.shape[axis] != shape[axis]) {
            PyErr_Format(PyExc_ValueError, "%s has an invalid shape", name);
            return false;
        }
    }
    return true;
}

double read2(const Py_buffer& view, Py_ssize_t row, Py_ssize_t column) {
    const char* address = static_cast<const char*>(view.buf)
        + row * view.strides[0] + column * view.strides[1];
    double value = 0.0;
    std::memcpy(&value, address, sizeof(double));
    return value;
}

double read3(
    const Py_buffer& view,
    Py_ssize_t first,
    Py_ssize_t second,
    Py_ssize_t third) {
    const char* address = static_cast<const char*>(view.buf)
        + first * view.strides[0] + second * view.strides[1]
        + third * view.strides[2];
    double value = 0.0;
    std::memcpy(&value, address, sizeof(double));
    return value;
}

std::int64_t read_int64(const Py_buffer& view, Py_ssize_t index) {
    const char* address = static_cast<const char*>(view.buf) + index * view.strides[0];
    std::int64_t value = 0;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

bool acquire_double_matrix(
    PyObject* object,
    Buffer& buffer,
    Py_ssize_t columns,
    const char* name) {
    if (PyObject_GetBuffer(object, &buffer.view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return false;
    }
    buffer.acquired = true;
    if (buffer.view.ndim != 2 || buffer.view.itemsize != sizeof(double) ||
        buffer.view.format == nullptr || std::strcmp(buffer.view.format, "d") != 0 ||
        buffer.view.shape[1] != columns) {
        PyErr_Format(
            PyExc_ValueError,
            "%s must be a float64 array with shape (N, %zd)",
            name,
            columns);
        return false;
    }
    return true;
}

bool acquire_index_vector(
    PyObject* object,
    Buffer& buffer,
    Py_ssize_t count,
    const char* name) {
    if (PyObject_GetBuffer(object, &buffer.view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return false;
    }
    buffer.acquired = true;
    const bool format_ok = buffer.view.format != nullptr &&
        (std::strcmp(buffer.view.format, "l") == 0 ||
         std::strcmp(buffer.view.format, "q") == 0);
    if (buffer.view.ndim != 1 || buffer.view.itemsize != sizeof(std::int64_t) ||
        !format_ok || buffer.view.shape[0] != count) {
        PyErr_Format(
            PyExc_ValueError,
            "%s must be an int64 array with shape (N,)",
            name);
        return false;
    }
    return true;
}

void cross3(const double first[3], const double second[3], double output[3]) {
    output[0] = first[1] * second[2] - first[2] * second[1];
    output[1] = first[2] * second[0] - first[0] * second[2];
    output[2] = first[0] * second[1] - first[1] * second[0];
}

double dot3(const double first[3], const double second[3]) {
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

void left_so3_jacobian(const double vector[3], double output[3][3]) {
    const double theta_squared = dot3(vector, vector);
    const double theta = std::sqrt(theta_squared);
    double first = 0.0;
    double second = 0.0;
    if (theta < 1e-6) {
        const double theta_fourth = theta_squared * theta_squared;
        first = 0.5 - theta_squared / 24.0 + theta_fourth / 720.0;
        second = 1.0 / 6.0 - theta_squared / 120.0 + theta_fourth / 5040.0;
    } else {
        first = (1.0 - std::cos(theta)) / theta_squared;
        second = (theta - std::sin(theta)) / (theta_squared * theta);
    }
    const double skew[3][3] = {
        {0.0, -vector[2], vector[1]},
        {vector[2], 0.0, -vector[0]},
        {-vector[1], vector[0], 0.0},
    };
    double skew_squared[3][3]{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            for (int inner = 0; inner < 3; ++inner) {
                skew_squared[row][column] += skew[row][inner] * skew[inner][column];
            }
            output[row][column] = (row == column ? 1.0 : 0.0)
                + first * skew[row][column] + second * skew_squared[row][column];
        }
    }
}

PyObject* five_point_coefficients(PyObject*, PyObject* argument) {
    const Py_ssize_t shape[2] = {9, 4};
    Buffer nullspace;
    if (!acquire_double_buffer(argument, nullspace, 2, shape, "nullspace")) {
        return nullptr;
    }
    constexpr Py_ssize_t output_count = 4 * 10 * 20;
    PyObject* bytes = PyBytes_FromStringAndSize(nullptr, output_count * sizeof(double));
    if (bytes == nullptr) {
        return nullptr;
    }
    char* output = PyBytes_AS_STRING(bytes);

    for (int constant_index = 0; constant_index < 4; ++constant_index) {
        std::array<int, 3> variables{};
        int cursor = 0;
        for (int index = 0; index < 4; ++index) {
            if (index != constant_index) {
                variables[cursor++] = index;
            }
        }
        PolyMatrix3 entries{};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                const int flat = 3 * row + column;
                Polynomial polynomial = zero_polynomial();
                polynomial[19] = read2(nullspace.view, flat, constant_index);
                for (int axis = 0; axis < 3; ++axis) {
                    std::array<int, 3> exponent{{0, 0, 0}};
                    exponent[axis] = 1;
                    polynomial[monomial_index(exponent)] =
                        read2(nullspace.view, flat, variables[axis]);
                }
                entries[row][column] = polynomial;
            }
        }
        const PolyMatrix3 eet = matrix_multiply(entries, transpose(entries));
        const PolyMatrix3 cubic = matrix_multiply(eet, entries);
        const Polynomial trace = add(add(eet[0][0], eet[1][1]), eet[2][2]);
        std::array<Polynomial, 10> constraints{};
        int constraint = 0;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                constraints[constraint++] = subtract(
                    scale(cubic[row][column], 2.0),
                    multiply(trace, entries[row][column]));
            }
        }
        constraints[9] = determinant(entries);
        for (int row = 0; row < 10; ++row) {
            for (int column = 0; column < 20; ++column) {
                const double value = constraints[row][column];
                std::memcpy(output, &value, sizeof(value));
                output += sizeof(value);
            }
        }
    }
    return bytes;
}

PyObject* sampson_residuals(PyObject*, PyObject* args) {
    PyObject* first_object = nullptr;
    PyObject* second_object = nullptr;
    PyObject* essential_object = nullptr;
    int squared = 0;
    if (!PyArg_ParseTuple(args, "OOOp", &first_object, &second_object, &essential_object, &squared)) {
        return nullptr;
    }
    Buffer first;
    if (PyObject_GetBuffer(first_object, &first.view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return nullptr;
    }
    first.acquired = true;
    if (first.view.ndim != 2 || first.view.itemsize != sizeof(double) ||
        first.view.format == nullptr || std::strcmp(first.view.format, "d") != 0 ||
        first.view.shape[1] != 3) {
        PyErr_SetString(PyExc_ValueError, "bearings_a must be a float64 array with shape (N, 3)");
        return nullptr;
    }
    const Py_ssize_t second_shape[2] = {first.view.shape[0], 3};
    const Py_ssize_t essential_shape[2] = {3, 3};
    Buffer second;
    Buffer essential;
    if (!acquire_double_buffer(second_object, second, 2, second_shape, "bearings_b") ||
        !acquire_double_buffer(essential_object, essential, 2, essential_shape, "essential_matrix")) {
        return nullptr;
    }
    const Py_ssize_t count = first.view.shape[0];
    PyObject* bytes = PyBytes_FromStringAndSize(nullptr, count * sizeof(double));
    if (bytes == nullptr) {
        return nullptr;
    }
    char* output = PyBytes_AS_STRING(bytes);
    constexpr double threshold = 64.0 * std::numeric_limits<double>::epsilon();
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t index = 0; index < count; ++index) {
        double b1[3];
        double b2[3];
        double eb1[3]{};
        double etb2[3]{};
        for (int axis = 0; axis < 3; ++axis) {
            b1[axis] = read2(first.view, index, axis);
            b2[axis] = read2(second.view, index, axis);
        }
        for (int axis = 0; axis < 3; ++axis) {
            for (int inner = 0; inner < 3; ++inner) {
                eb1[axis] += read2(essential.view, axis, inner) * b1[inner];
                etb2[axis] += read2(essential.view, inner, axis) * b2[inner];
            }
        }
        double numerator = 0.0;
        double projection1 = 0.0;
        double projection2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            numerator += b2[axis] * eb1[axis];
            projection1 += b1[axis] * etb2[axis];
            projection2 += b2[axis] * eb1[axis];
        }
        double denominator_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double grad1 = etb2[axis] - b1[axis] * projection1;
            const double grad2 = eb1[axis] - b2[axis] * projection2;
            denominator_squared += grad1 * grad1 + grad2 * grad2;
        }
        const double denominator = std::sqrt(denominator_squared);
        double error = std::numeric_limits<double>::infinity();
        if (denominator > threshold) {
            error = std::abs(numerator) / denominator;
            if (squared) {
                error *= error;
            }
        }
        std::memcpy(output + index * sizeof(double), &error, sizeof(error));
    }
    Py_END_ALLOW_THREADS
    return bytes;
}

PyObject* spherical_ba_residual_jacobians(PyObject*, PyObject* args) {
    PyObject* measured_object = nullptr;
    PyObject* rotations_object = nullptr;
    PyObject* centers_object = nullptr;
    PyObject* points_object = nullptr;
    PyObject* camera_indices_object = nullptr;
    PyObject* point_indices_object = nullptr;
    PyObject* rotation_deltas_object = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOO",
            &measured_object,
            &rotations_object,
            &centers_object,
            &points_object,
            &camera_indices_object,
            &point_indices_object,
            &rotation_deltas_object)) {
        return nullptr;
    }

    Buffer measured;
    Buffer rotations;
    Buffer centers;
    Buffer points;
    Buffer camera_indices;
    Buffer point_indices;
    Buffer rotation_deltas;
    if (!acquire_double_matrix(measured_object, measured, 3, "measured_bearings")) {
        return nullptr;
    }
    const Py_ssize_t observation_count = measured.view.shape[0];
    if (PyObject_GetBuffer(
            rotations_object,
            &rotations.view,
            PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return nullptr;
    }
    rotations.acquired = true;
    if (rotations.view.ndim != 3 || rotations.view.itemsize != sizeof(double) ||
        rotations.view.format == nullptr || std::strcmp(rotations.view.format, "d") != 0 ||
        rotations.view.shape[1] != 3 || rotations.view.shape[2] != 3) {
        PyErr_SetString(
            PyExc_ValueError,
            "rotations must be a float64 array with shape (C, 3, 3)");
        return nullptr;
    }
    const Py_ssize_t camera_count = rotations.view.shape[0];
    if (!acquire_double_matrix(centers_object, centers, 3, "centers") ||
        centers.view.shape[0] != camera_count) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "centers must have shape (C, 3)");
        }
        return nullptr;
    }
    if (!acquire_double_matrix(points_object, points, 3, "points")) {
        return nullptr;
    }
    const Py_ssize_t point_count = points.view.shape[0];
    if (!acquire_index_vector(
            camera_indices_object,
            camera_indices,
            observation_count,
            "camera_indices") ||
        !acquire_index_vector(
            point_indices_object,
            point_indices,
            observation_count,
            "point_indices")) {
        return nullptr;
    }
    if (!acquire_double_matrix(
            rotation_deltas_object,
            rotation_deltas,
            3,
            "rotation_deltas") ||
        rotation_deltas.view.shape[0] != camera_count) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(
                PyExc_ValueError,
                "rotation_deltas must have shape (C, 3)");
        }
        return nullptr;
    }
    for (Py_ssize_t index = 0; index < observation_count; ++index) {
        const std::int64_t camera = read_int64(camera_indices.view, index);
        const std::int64_t point = read_int64(point_indices.view, index);
        if (camera < 0 || camera >= camera_count || point < 0 || point >= point_count) {
            PyErr_SetString(PyExc_ValueError, "observation indices are out of bounds");
            return nullptr;
        }
    }

    const Py_ssize_t residual_size = observation_count * 2 * sizeof(double);
    const Py_ssize_t jacobian_size = observation_count * 2 * 3 * sizeof(double);
    PyObject* residual_bytes = PyBytes_FromStringAndSize(nullptr, residual_size);
    PyObject* rotation_bytes = PyBytes_FromStringAndSize(nullptr, jacobian_size);
    PyObject* center_bytes = PyBytes_FromStringAndSize(nullptr, jacobian_size);
    PyObject* point_bytes = PyBytes_FromStringAndSize(nullptr, jacobian_size);
    if (residual_bytes == nullptr || rotation_bytes == nullptr ||
        center_bytes == nullptr || point_bytes == nullptr) {
        Py_XDECREF(residual_bytes);
        Py_XDECREF(rotation_bytes);
        Py_XDECREF(center_bytes);
        Py_XDECREF(point_bytes);
        return nullptr;
    }
    char* residual_output = PyBytes_AS_STRING(residual_bytes);
    char* rotation_output = PyBytes_AS_STRING(rotation_bytes);
    char* center_output = PyBytes_AS_STRING(center_bytes);
    char* point_output = PyBytes_AS_STRING(point_bytes);

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t index = 0; index < observation_count; ++index) {
        const Py_ssize_t camera = static_cast<Py_ssize_t>(
            read_int64(camera_indices.view, index));
        const Py_ssize_t point = static_cast<Py_ssize_t>(
            read_int64(point_indices.view, index));
        double measured_value[3];
        double center_value[3];
        double point_value[3];
        double rotation_value[3][3];
        double rotation_delta[3];
        for (int axis = 0; axis < 3; ++axis) {
            measured_value[axis] = read2(measured.view, index, axis);
            center_value[axis] = read2(centers.view, camera, axis);
            point_value[axis] = read2(points.view, point, axis);
            rotation_delta[axis] = read2(rotation_deltas.view, camera, axis);
            for (int column = 0; column < 3; ++column) {
                rotation_value[axis][column] = read3(
                    rotations.view, camera, axis, column);
            }
        }

        double ray[3]{};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                ray[row] += rotation_value[row][column]
                    * (point_value[column] - center_value[column]);
            }
        }
        const double ray_norm = std::sqrt(dot3(ray, ray));
        double residual_value[2]{kPi, kPi};
        double rotation_jacobian[2][3]{};
        double center_jacobian[2][3]{};
        double point_jacobian[2][3]{};

        if (ray_norm > 1e-12) {
            double predicted[3];
            for (int axis = 0; axis < 3; ++axis) {
                predicted[axis] = ray[axis] / ray_norm;
            }
            int basis_axis = 0;
            for (int axis = 1; axis < 3; ++axis) {
                if (std::abs(measured_value[axis]) <
                    std::abs(measured_value[basis_axis])) {
                    basis_axis = axis;
                }
            }
            double axis_value[3]{};
            axis_value[basis_axis] = 1.0;
            double first_basis[3];
            cross3(measured_value, axis_value, first_basis);
            const double first_norm = std::sqrt(dot3(first_basis, first_basis));
            for (double& value : first_basis) {
                value /= first_norm;
            }
            double second_basis[3];
            cross3(measured_value, first_basis, second_basis);

            double cosine = dot3(measured_value, predicted);
            cosine = std::max(-1.0, std::min(1.0, cosine));
            const double angle = std::acos(cosine);
            double tangent[3];
            for (int axis = 0; axis < 3; ++axis) {
                tangent[axis] = predicted[axis] - cosine * measured_value[axis];
            }
            const double tangent_norm = std::sqrt(dot3(tangent, tangent));
            const double first_coordinate = dot3(first_basis, tangent);
            const double second_coordinate = dot3(second_basis, tangent);
            double residual_wrt_prediction[2][3]{};
            if (tangent_norm >= 1e-12) {
                const double scale = angle / tangent_norm;
                residual_value[0] = scale * first_coordinate;
                residual_value[1] = scale * second_coordinate;
                if (tangent_norm >= 1e-8 && angle < kPi - 1e-8) {
                    double gradient_scale[3];
                    const double denominator =
                        tangent_norm * tangent_norm + cosine * cosine;
                    for (int axis = 0; axis < 3; ++axis) {
                        const double gradient_norm = tangent[axis] / tangent_norm;
                        const double gradient_angle =
                            (cosine * gradient_norm
                             - tangent_norm * measured_value[axis])
                            / denominator;
                        gradient_scale[axis] =
                            gradient_angle / tangent_norm
                            - angle * gradient_norm
                                / (tangent_norm * tangent_norm);
                    }
                    for (int axis = 0; axis < 3; ++axis) {
                        residual_wrt_prediction[0][axis] =
                            scale * first_basis[axis]
                            + first_coordinate * gradient_scale[axis];
                        residual_wrt_prediction[1][axis] =
                            scale * second_basis[axis]
                            + second_coordinate * gradient_scale[axis];
                    }
                } else if (angle < 1e-8) {
                    for (int axis = 0; axis < 3; ++axis) {
                        residual_wrt_prediction[0][axis] = first_basis[axis];
                        residual_wrt_prediction[1][axis] = second_basis[axis];
                    }
                }
            } else if (angle < 1e-8) {
                residual_value[0] = first_coordinate;
                residual_value[1] = second_coordinate;
                for (int axis = 0; axis < 3; ++axis) {
                    residual_wrt_prediction[0][axis] = first_basis[axis];
                    residual_wrt_prediction[1][axis] = second_basis[axis];
                }
            } else {
                residual_value[0] = angle;
                residual_value[1] = 0.0;
            }

            double residual_wrt_ray[2][3]{};
            for (int row = 0; row < 2; ++row) {
                for (int column = 0; column < 3; ++column) {
                    for (int inner = 0; inner < 3; ++inner) {
                        const double normalization =
                            ((inner == column ? 1.0 : 0.0)
                             - predicted[inner] * predicted[column])
                            / ray_norm;
                        residual_wrt_ray[row][column] +=
                            residual_wrt_prediction[row][inner] * normalization;
                    }
                }
            }
            for (int row = 0; row < 2; ++row) {
                for (int column = 0; column < 3; ++column) {
                    for (int inner = 0; inner < 3; ++inner) {
                        point_jacobian[row][column] +=
                            residual_wrt_ray[row][inner]
                            * rotation_value[inner][column];
                    }
                    center_jacobian[row][column] =
                        -point_jacobian[row][column];
                }
            }
            double left_jacobian[3][3];
            left_so3_jacobian(rotation_delta, left_jacobian);
            const double negative_skew_ray[3][3] = {
                {0.0, ray[2], -ray[1]},
                {-ray[2], 0.0, ray[0]},
                {ray[1], -ray[0], 0.0},
            };
            double ray_wrt_rotation[3][3]{};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    for (int inner = 0; inner < 3; ++inner) {
                        ray_wrt_rotation[row][column] +=
                            negative_skew_ray[row][inner]
                            * left_jacobian[inner][column];
                    }
                }
            }
            for (int row = 0; row < 2; ++row) {
                for (int column = 0; column < 3; ++column) {
                    for (int inner = 0; inner < 3; ++inner) {
                        rotation_jacobian[row][column] +=
                            residual_wrt_ray[row][inner]
                            * ray_wrt_rotation[inner][column];
                    }
                }
            }
        }

        for (int row = 0; row < 2; ++row) {
            std::memcpy(
                residual_output + (2 * index + row) * sizeof(double),
                &residual_value[row],
                sizeof(double));
            for (int column = 0; column < 3; ++column) {
                const Py_ssize_t offset =
                    (index * 6 + row * 3 + column) * sizeof(double);
                std::memcpy(
                    rotation_output + offset,
                    &rotation_jacobian[row][column],
                    sizeof(double));
                std::memcpy(
                    center_output + offset,
                    &center_jacobian[row][column],
                    sizeof(double));
                std::memcpy(
                    point_output + offset,
                    &point_jacobian[row][column],
                    sizeof(double));
            }
        }
    }
    Py_END_ALLOW_THREADS

    PyObject* result = PyTuple_New(4);
    if (result == nullptr) {
        Py_DECREF(residual_bytes);
        Py_DECREF(rotation_bytes);
        Py_DECREF(center_bytes);
        Py_DECREF(point_bytes);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, residual_bytes);
    PyTuple_SET_ITEM(result, 1, rotation_bytes);
    PyTuple_SET_ITEM(result, 2, center_bytes);
    PyTuple_SET_ITEM(result, 3, point_bytes);
    return result;
}

PyMethodDef methods[] = {
    {"five_point_coefficients", reinterpret_cast<PyCFunction>(five_point_coefficients), METH_O,
     "Build four five-point cubic coefficient matrices."},
    {"sampson_residuals", sampson_residuals, METH_VARARGS,
     "Compute spherical tangent-Sampson residuals."},
    {"spherical_ba_residual_jacobians", spherical_ba_residual_jacobians, METH_VARARGS,
     "Compute spherical bundle residuals and analytic local Jacobian blocks."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_essential",
    "First-party PanorAi essential-matrix kernels.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__essential() {
    return PyModule_Create(&module);
}
