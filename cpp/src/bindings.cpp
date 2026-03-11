#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "vector_index.hpp"

namespace py = pybind11;

static std::vector<std::vector<float>> numpy_to_vectors(const py::array_t<float>& arr) {
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D float32 numpy array.");
    }
    const int rows = static_cast<int>(buf.shape[0]);
    const int cols = static_cast<int>(buf.shape[1]);
    const auto* ptr = static_cast<float*>(buf.ptr);
    std::vector<std::vector<float>> out(rows, std::vector<float>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[r][c] = ptr[r * cols + c];
        }
    }
    return out;
}

static std::vector<float> numpy_to_vector(const py::array_t<float>& arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D float32 numpy array.");
    }
    const int n = static_cast<int>(buf.shape[0]);
    const auto* ptr = static_cast<float*>(buf.ptr);
    return std::vector<float>(ptr, ptr + n);
}

PYBIND11_MODULE(pocket_memory_cpp, m) {
    m.doc() = "Pocket Memory C++ vector index bindings";

    py::class_<VectorIndex>(m, "VectorIndex")
        .def(py::init<>())
        .def(
            "configure",
            &VectorIndex::configure,
            py::arg("mode") = "flat",
            py::arg("dim") = 384,
            py::arg("hnsw_m") = 32,
            py::arg("ef_construction") = 128,
            py::arg("ef_search") = 64,
            py::arg("ivf_nlist") = 100,
            py::arg("ivf_nprobe") = 10)
        .def("add", [](VectorIndex& self, const std::vector<std::string>& ids, py::array_t<float> vectors) {
            self.add(ids, numpy_to_vectors(vectors));
        })
        .def("search", [](const VectorIndex& self, py::array_t<float> query, int top_k) {
            return self.search(numpy_to_vector(query), top_k);
        })
        .def("last_search_stats", &VectorIndex::last_search_stats)
        .def("mode", &VectorIndex::mode)
        .def("size", &VectorIndex::size);
}
