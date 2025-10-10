/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_IVF_HNSW_C_H
#define FAISS_INDEX_IVF_HNSW_C_H

#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Index based on inverted file with HNSW quantizer
 *
 * This index uses an HNSW index as the quantizer for an IVF structure.
 * The HNSW quantizer provides fast approximate nearest neighbor search
 * for the coarse quantization step.
 */
FAISS_DECLARE_CLASS_INHERITED(IndexIVFHNSW, Index)
FAISS_DECLARE_DESTRUCTOR(IndexIVFHNSW)
FAISS_DECLARE_INDEX_DOWNCAST(IndexIVFHNSW)

/// number of possible key values
FAISS_DECLARE_GETTER(IndexIVFHNSW, size_t, nlist)
/// number of probes at query time
FAISS_DECLARE_GETTER_SETTER(IndexIVFHNSW, size_t, nprobe)
/// quantizer that maps vectors to inverted lists
FAISS_DECLARE_GETTER(IndexIVFHNSW, FaissIndex*, quantizer)

/// HNSW parameter: number of bi-directional links for each node
FAISS_DECLARE_GETTER_SETTER(IndexIVFHNSW, int, M)
/// HNSW parameter: size of the dynamic candidate list for construction
FAISS_DECLARE_GETTER_SETTER(IndexIVFHNSW, int, efConstruction)
/// HNSW parameter: size of the dynamic candidate list for search
FAISS_DECLARE_GETTER_SETTER(IndexIVFHNSW, int, efSearch)

/// Create a new IndexIVFHNSW
int faiss_IndexIVFHNSW_new(FaissIndexIVFHNSW** p_index);

/// Create a new IndexIVFHNSW with parameters
int faiss_IndexIVFHNSW_new_with(
        FaissIndexIVFHNSW** p_index,
        int d,
        size_t nlist,
        int M,
        int efConstruction,
        int efSearch,
        FaissMetricType metric);

/// Create a new IndexIVFHNSW with default parameters
int faiss_IndexIVFHNSW_new_with_defaults(
        FaissIndexIVFHNSW** p_index,
        int d,
        size_t nlist);

/// Get the HNSW quantizer
int faiss_IndexIVFHNSW_get_hnsw_quantizer(
        const FaissIndexIVFHNSW* index,
        FaissIndex** quantizer);

/// Set HNSW parameters
int faiss_IndexIVFHNSW_set_hnsw_params(
        FaissIndexIVFHNSW* index,
        int M,
        int efConstruction,
        int efSearch);

/// Get HNSW parameters
int faiss_IndexIVFHNSW_get_hnsw_params(
        const FaissIndexIVFHNSW* index,
        int* M,
        int* efConstruction,
        int* efSearch);

#ifdef __cplusplus
}
#endif

#endif