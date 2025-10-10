/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFHNSW.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>
#include <cstring>

namespace faiss {

IndexIVFHNSW::IndexIVFHNSW()
        : IndexIVF(nullptr, 0, 0, 0, METRIC_L2),
          M(32),
          efConstruction(200),
          efSearch(50) {
    // Default constructor - quantizer will be set later
}

IndexIVFHNSW::IndexIVFHNSW(
        int d,
        size_t nlist,
        int M,
        int efConstruction,
        int efSearch,
        MetricType metric)
        : IndexIVF(nullptr, d, nlist, d * sizeof(float), metric),
          M(M),
          efConstruction(efConstruction),
          efSearch(efSearch) {
    // Create HNSW quantizer
    quantizer = new IndexHNSW(d, M, metric);
    own_fields = true;
}

IndexIVFHNSW::~IndexIVFHNSW() {
    // The quantizer will be deleted by the parent class if own_fields is true
}

void IndexIVFHNSW::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(quantizer, "quantizer not initialized");
    
    // Train the HNSW quantizer
    quantizer->train(n, x);
    
    // Mark as trained
    is_trained = true;
}

void IndexIVFHNSW::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void IndexIVFHNSW::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index not trained");
    FAISS_THROW_IF_NOT_MSG(quantizer, "quantizer not initialized");
    
    // Add vectors to HNSW quantizer
    quantizer->add(n, x);
    
    // For now, we'll use a simple approach where we store the vectors
    // in the HNSW quantizer and use it for both quantization and storage
    // In a more sophisticated implementation, we would maintain separate
    // inverted lists and use the HNSW only for quantization
    
    // Update ntotal
    ntotal += n;
}

void IndexIVFHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index not trained");
    FAISS_THROW_IF_NOT_MSG(quantizer, "quantizer not initialized");
    
    // For now, delegate to the HNSW quantizer's search
    // In a more sophisticated implementation, we would:
    // 1. Use HNSW to find the nearest centroids
    // 2. Search the corresponding inverted lists
    quantizer->search(n, x, k, distances, labels, params);
}

void IndexIVFHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index not trained");
    FAISS_THROW_IF_NOT_MSG(quantizer, "quantizer not initialized");
    
    // Delegate to the HNSW quantizer's range search
    quantizer->range_search(n, x, radius, result, params);
}

void IndexIVFHNSW::reset() {
    if (quantizer) {
        quantizer->reset();
    }
    ntotal = 0;
}

void IndexIVFHNSW::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listno) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index not trained");
    
    // For this simplified implementation, we just copy the vectors as-is
    // In a more sophisticated implementation, we would encode them properly
    size_t code_size_per_vector = d * sizeof(float);
    if (include_listno) {
        // Calculate number of bytes needed for list number
        size_t listno_bits = 0;
        size_t temp = nlist - 1;
        while (temp > 0) {
            listno_bits++;
            temp >>= 1;
        }
        code_size_per_vector += (listno_bits + 7) / 8;
    }
    
    for (idx_t i = 0; i < n; ++i) {
        if (list_nos[i] >= 0) {
            // Copy the vector data
            memcpy(codes + i * code_size_per_vector, x + i * d, d * sizeof(float));
            
            // Add list number if requested
            if (include_listno && nlist > 1) {
                size_t listno_bits = 0;
                size_t temp = nlist - 1;
                while (temp > 0) {
                    listno_bits++;
                    temp >>= 1;
                }
                size_t listno_size = (listno_bits + 7) / 8;
                memcpy(codes + i * code_size_per_vector + d * sizeof(float), 
                       &list_nos[i], listno_size);
            }
        }
    }
}

const IndexHNSW* IndexIVFHNSW::get_hnsw_quantizer() const {
    return dynamic_cast<const IndexHNSW*>(quantizer);
}

void IndexIVFHNSW::set_hnsw_params(int M, int efConstruction, int efSearch) {
    this->M = M;
    this->efConstruction = efConstruction;
    this->efSearch = efSearch;
    
    const IndexHNSW* hnsw_quantizer = get_hnsw_quantizer();
    if (hnsw_quantizer) {
        // Update HNSW parameters if the quantizer is already created
        // Note: This is a simplified approach. In practice, you might need
        // to recreate the quantizer with new parameters
        // We need to cast away const to modify the parameters
        IndexHNSW* mutable_hnsw = const_cast<IndexHNSW*>(hnsw_quantizer);
        mutable_hnsw->hnsw.efConstruction = efConstruction;
        mutable_hnsw->hnsw.efSearch = efSearch;
    }
}

void IndexIVFHNSW::get_hnsw_params(int* M, int* efConstruction, int* efSearch) const {
    if (M) *M = this->M;
    if (efConstruction) *efConstruction = this->efConstruction;
    if (efSearch) *efSearch = this->efSearch;
}

} // namespace faiss