/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/IndexHNSW.h>

namespace faiss {

/** Index based on inverted file with HNSW quantizer
 *
 * This index uses an HNSW index as the quantizer for an IVF structure.
 * The HNSW quantizer provides fast approximate nearest neighbor search
 * for the coarse quantization step.
 */
struct IndexIVFHNSW : IndexIVF {
    /// HNSW quantizer parameters
    int M = 32;  ///< number of bi-directional links for each node
    int efConstruction = 200;  ///< size of the dynamic candidate list
    int efSearch = 50;  ///< size of the dynamic candidate list for search

    IndexIVFHNSW();

    IndexIVFHNSW(
            int d,
            size_t nlist,
            int M = 32,
            int efConstruction = 200,
            int efSearch = 50,
            MetricType metric = METRIC_L2);

    ~IndexIVFHNSW() override;

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listno = false) const override;

    /// Get the HNSW quantizer
    const IndexHNSW* get_hnsw_quantizer() const;

    /// Set HNSW parameters
    void set_hnsw_params(int M, int efConstruction, int efSearch);

    /// Get HNSW parameters
    void get_hnsw_params(int* M, int* efConstruction, int* efSearch) const;
};

} // namespace faiss