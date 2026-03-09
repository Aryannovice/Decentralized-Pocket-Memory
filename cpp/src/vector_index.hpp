#pragma once

#include <cstddef>
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
};
