#include "vector_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

std::vector<uint8_t> pack_sign_bits(const std::vector<float>& v) {
    const size_t packed_size = (v.size() + 7U) / 8U;
    std::vector<uint8_t> packed(packed_size, 0U);
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] >= 0.0f) {
            packed[i / 8U] |= static_cast<uint8_t>(1U << (7U - (i % 8U)));
        }
    }
    return packed;
}

int hamming_similarity(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    const size_t size = std::min(a.size(), b.size());
    size_t i = 0U;
    int diff_bits = 0;

    while (i + sizeof(uint64_t) <= size) {
        uint64_t va = 0U;
        uint64_t vb = 0U;
        std::memcpy(&va, a.data() + i, sizeof(uint64_t));
        std::memcpy(&vb, b.data() + i, sizeof(uint64_t));
        diff_bits += __builtin_popcountll(va ^ vb);
        i += sizeof(uint64_t);
    }
    while (i < size) {
        diff_bits += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
        ++i;
    }

    return static_cast<int>(size * 8U) - diff_bits;
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

    } else if (mode_ == "hybrid_binary") {
        // Hybrid uses binary prefilter + local float rerank.
        s.quantizer.reset();
        s.index.reset();
    } else {
        throw std::runtime_error("Unknown mode, expected: flat, hnsw, ivfpq, hybrid_binary");
    }

#else
    if (mode_ != "flat" && mode_ != "hybrid_binary") {
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

    if (mode_ != "hybrid_binary" && !s.index) {
        configure("flat", dim_, hnsw_m_, ef_construction_,
                  ef_search_, ivf_nlist_, ivf_nprobe_);
    }

    if (mode_ != "hybrid_binary") {
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
    }

#endif

    for (size_t i = 0; i < ids.size(); ++i) {
        ids_.push_back(ids[i]);
        auto normed = normalized_copy(vectors[i]);
        vectors_.push_back(normed);
        binary_vectors_.push_back(pack_sign_bits(normed));
    }
}

std::vector<std::pair<std::string, float>>
VectorIndex::search(const std::vector<float>& query, int top_k) const {
    const auto total_start = std::chrono::steady_clock::now();
    last_prefilter_ms_ = 0.0;
    last_rerank_ms_ = 0.0;
    last_prefilter_candidates_ = 0U;
    last_total_ms_ = 0.0;

    if (vectors_.empty() || query.empty() || top_k <= 0) {
        return {};
    }

    if (static_cast<int>(query.size()) != dim_) {
        throw std::runtime_error("query dim mismatch");
    }

#ifdef USE_FAISS

    auto& s = state();

    if (s.index && mode_ != "hybrid_binary") {

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

        const auto total_end = std::chrono::steady_clock::now();
        last_total_ms_ = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        return out;
    }

#endif

    if (mode_ == "hybrid_binary") {
        const auto q = normalized_copy(query);
        const auto qbin = pack_sign_bits(q);

        const auto pre_start = std::chrono::steady_clock::now();
        std::vector<std::pair<size_t, int>> pre;
        pre.reserve(binary_vectors_.size());
        for (size_t i = 0; i < binary_vectors_.size(); ++i) {
            pre.push_back({i, hamming_similarity(qbin, binary_vectors_[i])});
        }

        const size_t prefilter_count = std::min(
            std::max(static_cast<size_t>(top_k) * 8U, static_cast<size_t>(top_k)),
            pre.size());
        last_prefilter_candidates_ = prefilter_count;
        std::partial_sort(
            pre.begin(),
            pre.begin() + static_cast<std::ptrdiff_t>(prefilter_count),
            pre.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        const auto pre_end = std::chrono::steady_clock::now();
        last_prefilter_ms_ = std::chrono::duration<double, std::milli>(pre_end - pre_start).count();

        const auto rerank_start = std::chrono::steady_clock::now();
        std::vector<std::pair<std::string, float>> reranked;
        reranked.reserve(prefilter_count);
        for (size_t i = 0; i < prefilter_count; ++i) {
            const size_t idx = pre[i].first;
            const auto& vec = vectors_[idx];
            float dot = 0.0f;
            for (size_t j = 0; j < vec.size(); ++j) {
                dot += vec[j] * q[j];
            }
            reranked.push_back({ids_[idx], dot});
        }
        std::sort(
            reranked.begin(),
            reranked.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        if (static_cast<size_t>(top_k) < reranked.size()) {
            reranked.resize(static_cast<size_t>(top_k));
        }
        const auto rerank_end = std::chrono::steady_clock::now();
        last_rerank_ms_ = std::chrono::duration<double, std::milli>(rerank_end - rerank_start).count();
        last_total_ms_ = std::chrono::duration<double, std::milli>(rerank_end - total_start).count();
        return reranked;
    }

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

    const auto total_end = std::chrono::steady_clock::now();
    last_total_ms_ = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    return scored;
}

std::string VectorIndex::mode() const {
    return mode_;
}

std::size_t VectorIndex::size() const {
    return ids_.size();
}

std::vector<double> VectorIndex::last_search_stats() const {
    return {
        static_cast<double>(last_prefilter_candidates_),
        last_prefilter_ms_,
        last_rerank_ms_,
        last_total_ms_,
    };
}