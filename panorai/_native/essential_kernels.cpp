#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

namespace {

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

PyMethodDef methods[] = {
    {"five_point_coefficients", reinterpret_cast<PyCFunction>(five_point_coefficients), METH_O,
     "Build four five-point cubic coefficient matrices."},
    {"sampson_residuals", sampson_residuals, METH_VARARGS,
     "Compute spherical tangent-Sampson residuals."},
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
