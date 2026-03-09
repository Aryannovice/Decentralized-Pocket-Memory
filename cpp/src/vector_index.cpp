#include "vector_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_FAISS
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#endif

namespace {
std::vector<float> normalized_copy(const std::vector<float>& v) {
    float norm = 0.0f;
    for (float x : v) {
        norm += x * x;
    }
    norm = std::sqrt(norm) + 1e-8f;

    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = v[i] / norm;
    }
    return out;
}
}

#ifdef USE_FAISS
namespace {

struct FaissState {
    std::unique_ptr<faiss::Index> index;
    std::unique_ptr<faiss::IndexFlatIP> quantizer;
};

FaissState& state() {
    static FaissState s;
    return s;
}

}
#endif

VectorIndex::VectorIndex() = default;

void VectorIndex::configure(
    const std::string& mode,
    int dim,
    int hnsw_m,
    int ef_construction,
    int ef_search,
    int ivf_nlist,
    int ivf_nprobe) {

    mode_ = mode;
    dim_ = dim;
    hnsw_m_ = hnsw_m;
    ef_construction_ = ef_construction;
    ef_search_ = ef_search;
    ivf_nlist_ = ivf_nlist;
    ivf_nprobe_ = ivf_nprobe;

#ifdef USE_FAISS
    if (dim_ <= 0) {
        throw std::runtime_error("configure: dim must be > 0");
    }

    auto& s = state();

    if (mode_ == "flat") {

        s.quantizer.reset();
        s.index = std::make_unique<faiss::IndexFlatIP>(dim_);

    } else if (mode_ == "hnsw") {

        s.quantizer.reset();
        auto hnsw = std::make_unique<faiss::IndexHNSWFlat>(
            dim_, hnsw_m_, faiss::METRIC_INNER_PRODUCT);

        hnsw->hnsw.efConstruction = ef_construction_;
        hnsw->hnsw.efSearch = ef_search_;

        s.index = std::move(hnsw);

    } else if (mode_ == "ivfpq") {

        s.quantizer = std::make_unique<faiss::IndexFlatIP>(dim_);

        auto ivfpq = std::make_unique<faiss::IndexIVFPQ>(
            s.quantizer.get(),
            dim_,
            ivf_nlist_,
            8,
            8,
            faiss::METRIC_INNER_PRODUCT);

        ivfpq->nprobe = ivf_nprobe_;

        s.index = std::move(ivfpq);

    } else {
        throw std::runtime_error("Unknown mode, expected: flat, hnsw, ivfpq");
    }

#else
    if (mode_ != "flat") {
        throw std::runtime_error("ANN mode requested but module built without FAISS.");
    }
#endif
}

void VectorIndex::add(
    const std::vector<std::string>& ids,
    const std::vector<std::vector<float>>& vectors) {

    if (ids.size() != vectors.size()) {
        throw std::runtime_error("ids and vectors must have matching lengths");
    }

    if (ids.empty()) {
        return;
    }

    const int add_dim = static_cast<int>(vectors[0].size());

    if (dim_ == 0) {
        dim_ = add_dim;
    }

    if (add_dim != dim_) {
        throw std::runtime_error("vector dim mismatch with configured dim");
    }

#ifdef USE_FAISS

    auto& s = state();

    if (!s.index) {
        configure("flat", dim_, hnsw_m_, ef_construction_,
                  ef_search_, ivf_nlist_, ivf_nprobe_);
    }

    using FaissIdx = decltype(s.index->ntotal);

    std::vector<float> matrix;
    matrix.reserve(vectors.size() * static_cast<size_t>(dim_));

    for (const auto& vec : vectors) {

        if (static_cast<int>(vec.size()) != dim_) {
            throw std::runtime_error("vector dim mismatch in add()");
        }

        auto normed = normalized_copy(vec);
        matrix.insert(matrix.end(), normed.begin(), normed.end());
    }

    if (mode_ == "ivfpq") {

        auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(s.index.get());

        if (ivfpq != nullptr && !ivfpq->is_trained) {
            ivfpq->train(
                static_cast<FaissIdx>(vectors.size()),
                matrix.data());
        }
    }

    s.index->add(
        static_cast<FaissIdx>(vectors.size()),
        matrix.data());

#endif

    for (size_t i = 0; i < ids.size(); ++i) {
        ids_.push_back(ids[i]);
        vectors_.push_back(normalized_copy(vectors[i]));
    }
}

std::vector<std::pair<std::string, float>>
VectorIndex::search(const std::vector<float>& query, int top_k) const {

    if (vectors_.empty() || query.empty() || top_k <= 0) {
        return {};
    }

    if (static_cast<int>(query.size()) != dim_) {
        throw std::runtime_error("query dim mismatch");
    }

#ifdef USE_FAISS

    auto& s = state();

    if (s.index) {

        using FaissIdx = decltype(s.index->ntotal);

        auto q = normalized_copy(query);

        std::vector<FaissIdx> labels(static_cast<size_t>(top_k), -1);
        std::vector<float> distances(static_cast<size_t>(top_k), 0.0f);

        s.index->search(
            1,
            q.data(),
            static_cast<FaissIdx>(top_k),
            distances.data(),
            labels.data());

        std::vector<std::pair<std::string, float>> out;
        out.reserve(static_cast<size_t>(top_k));

        for (int i = 0; i < top_k; ++i) {

            const auto label = labels[static_cast<size_t>(i)];

            if (label < 0) {
                continue;
            }

            if (static_cast<size_t>(label) >= ids_.size()) {
                continue;
            }

            out.push_back({
                ids_[static_cast<size_t>(label)],
                distances[static_cast<size_t>(i)]
            });
        }

        return out;
    }

#endif

    auto q = normalized_copy(query);

    std::vector<std::pair<std::string, float>> scored;
    scored.reserve(vectors_.size());

    for (size_t i = 0; i < vectors_.size(); ++i) {

        const auto& vec = vectors_[i];
        float dot = 0.0f;

        for (size_t j = 0; j < vec.size(); ++j) {
            dot += vec[j] * q[j];
        }

        scored.push_back({ids_[i], dot});
    }

    std::sort(
        scored.begin(),
        scored.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    if (static_cast<size_t>(top_k) < scored.size()) {
        scored.resize(static_cast<size_t>(top_k));
    }

    return scored;
}

std::string VectorIndex::mode() const {
    return mode_;
}

std::size_t VectorIndex::size() const {
    return ids_.size();
}