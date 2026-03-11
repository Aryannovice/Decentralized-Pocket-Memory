#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

class VectorIndex {
public:
    VectorIndex();
    void configure(
        const std::string& mode,
        int dim,
        int hnsw_m,
        int ef_construction,
        int ef_search,
        int ivf_nlist,
        int ivf_nprobe);
    void add(const std::vector<std::string>& ids, const std::vector<std::vector<float>>& vectors);
    std::vector<std::pair<std::string, float>> search(const std::vector<float>& query, int top_k) const;
    std::string mode() const;
    std::size_t size() const;
    std::vector<double> last_search_stats() const;

private:
    std::string mode_ = "flat";
    int dim_ = 0;
    int hnsw_m_ = 32;
    int ef_construction_ = 128;
    int ef_search_ = 64;
    int ivf_nlist_ = 100;
    int ivf_nprobe_ = 10;
    std::vector<std::string> ids_;
    std::vector<std::vector<float>> vectors_;
    std::vector<std::vector<uint8_t>> binary_vectors_;
    mutable double last_prefilter_ms_ = 0.0;
    mutable double last_rerank_ms_ = 0.0;
    mutable double last_total_ms_ = 0.0;
    mutable std::size_t last_prefilter_candidates_ = 0;
};
