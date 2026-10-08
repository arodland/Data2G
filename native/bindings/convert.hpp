// numpy <-> core containers, shared by the bind_*.cpp files.
#pragma once

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "codes/codes.hpp"
#include "util/mat.hpp"

namespace data2g::bind {

namespace py = pybind11;

template <typename T>
using In = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
std::vector<T> vec(const In<T>& a) {
    return {a.data(), a.data() + a.size()};
}

// A 1-D array is one row.
template <typename T>
Mat<T> mat(const In<T>& a) {
    if (a.ndim() > 2) throw std::invalid_argument("expected a 1-D or 2-D array");
    const std::size_t rows = a.ndim() == 2 ? a.shape(0) : 1, cols = a.ndim() == 2 ? a.shape(1) : a.size();
    Mat<T> m(rows, cols);
    std::copy(a.data(), a.data() + a.size(), m.data.begin());
    return m;
}

template <typename Out, typename T>
py::array_t<Out> np(std::span<const T> v) {
    py::array_t<Out> a(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

template <typename T>
py::array_t<T> np(const std::vector<T>& v) {
    return np<T>(std::span<const T>(v));
}

template <typename Out = void, typename T>
auto np(const Mat<T>& m) {
    using O = std::conditional_t<std::is_void_v<Out>, T, Out>;
    py::array_t<O> a({static_cast<py::ssize_t>(m.rows), static_cast<py::ssize_t>(m.cols)});
    std::copy(m.data.begin(), m.data.end(), a.mutable_data());
    return a;
}

// An input array as a span, without vec()'s copy: modem demodulates one
// window at a time out of a whole burst.
template <typename T>
std::span<const T> view(const In<T>& a) {
    return {a.data(), static_cast<std::size_t>(a.size())};
}

inline std::span<const std::uint8_t> bytes_view(const py::bytes& b) {
    const std::string_view s(b);
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

inline py::bytes to_bytes(std::span<const std::uint8_t> v) {
    return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}

// Submodes cross the boundary by SubmodeSpec.name.
inline const config::Submode& spec(const std::string& name) {
    const auto* s = codes::submode(name);
    if (!s) throw py::key_error("no submode " + name);
    return *s;
}

}  // namespace data2g::bind
