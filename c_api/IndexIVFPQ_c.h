/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_IVFPQ_C_H
#define FAISS_INDEX_IVFPQ_C_H

#include "IndexIVF_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS_INHERITED(IndexIVFPQ, IndexIVF)
FAISS_DECLARE_DESTRUCTOR(IndexIVFPQ)
FAISS_DECLARE_INDEX_DOWNCAST(IndexIVFPQ)

/// Constructor for IndexIVFPQ
int faiss_IndexIVFPQ_new(
        FaissIndexIVFPQ** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        FaissMetricType metric);

/// Constructor for IndexIVFPQ with own_invlists parameter
int faiss_IndexIVFPQ_new_with(
        FaissIndexIVFPQ** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        FaissMetricType metric,
        int own_invlists);

/// Default constructor for IndexIVFPQ
int faiss_IndexIVFPQ_new_default(FaissIndexIVFPQ** p_index);

/// Get the ProductQuantizer M parameter
FAISS_DECLARE_GETTER(IndexIVFPQ, size_t, pq_M)

/// Get the ProductQuantizer nbits parameter
FAISS_DECLARE_GETTER(IndexIVFPQ, size_t, pq_nbits)

/// Get the ProductQuantizer code_size
FAISS_DECLARE_GETTER(IndexIVFPQ, size_t, pq_code_size)

/// Get the ProductQuantizer dsub parameter
FAISS_DECLARE_GETTER(IndexIVFPQ, size_t, pq_dsub)

/// Get the ProductQuantizer ksub parameter
FAISS_DECLARE_GETTER(IndexIVFPQ, size_t, pq_ksub)

/// Get whether polysemous training is enabled
FAISS_DECLARE_GETTER(IndexIVFPQ, int, do_polysemous_training)

/// Set whether polysemous training is enabled
FAISS_DECLARE_SETTER(IndexIVFPQ, int, do_polysemous_training)

/// Get the scan table threshold
FAISS_DECLARE_GETTER(IndexIVFPQ, size_t, scan_table_threshold)

/// Set the scan table threshold
FAISS_DECLARE_SETTER(IndexIVFPQ, size_t, scan_table_threshold)

/// Get the polysemous Hamming threshold
FAISS_DECLARE_GETTER(IndexIVFPQ, int, polysemous_ht)

/// Set the polysemous Hamming threshold
FAISS_DECLARE_SETTER(IndexIVFPQ, int, polysemous_ht)

/// Get whether precomputed table is used
FAISS_DECLARE_GETTER(IndexIVFPQ, int, use_precomputed_table)

/// Set whether precomputed table is used
FAISS_DECLARE_SETTER(IndexIVFPQ, int, use_precomputed_table)

/// Encode a single vector
int faiss_IndexIVFPQ_encode(
        const FaissIndexIVFPQ* index,
        idx_t key,
        const float* x,
        uint8_t* code);

/// Encode multiple vectors
int faiss_IndexIVFPQ_encode_multiple(
        const FaissIndexIVFPQ* index,
        size_t n,
        idx_t* keys,
        const float* x,
        uint8_t* codes,
        int compute_keys);

/// Decode multiple vectors
int faiss_IndexIVFPQ_decode_multiple(
        const FaissIndexIVFPQ* index,
        size_t n,
        const idx_t* keys,
        const uint8_t* codes,
        float* x);

/// Find exact duplicates in the dataset
int faiss_IndexIVFPQ_find_duplicates(
        const FaissIndexIVFPQ* index,
        idx_t* ids,
        size_t* lims,
        size_t* n_duplicates);

/// Build precomputed table
int faiss_IndexIVFPQ_precompute_table(FaissIndexIVFPQ* index);

/// Get the number of vectors needed for training the encoder
idx_t faiss_IndexIVFPQ_train_encoder_num_vectors(const FaissIndexIVFPQ* index);

/// Train the encoder
int faiss_IndexIVFPQ_train_encoder(
        FaissIndexIVFPQ* index,
        idx_t n,
        const float* x,
        const idx_t* assign);

/// Reconstruct a vector from offset
int faiss_IndexIVFPQ_reconstruct_from_offset(
        const FaissIndexIVFPQ* index,
        int64_t list_no,
        int64_t offset,
        float* recons);

/// Get the precomputed table max bytes setting
size_t faiss_get_precomputed_table_max_bytes(void);

/// Set the precomputed table max bytes setting
void faiss_set_precomputed_table_max_bytes(size_t value);

/// Get IndexIVFPQ statistics
typedef struct FaissIndexIVFPQStats {
    size_t nrefine;           ///< nb of refines (IVFPQR)
    size_t n_hamming_pass;    ///< nb of passed Hamming distance tests (for polysemous)
    size_t search_cycles;     ///< timings measured with the CPU RTC on all threads
    size_t refine_cycles;     ///< only for IVFPQR
} FaissIndexIVFPQStats;

/// Get the global IndexIVFPQ statistics
FaissIndexIVFPQStats* faiss_get_indexIVFPQ_stats(void);

/// Reset the IndexIVFPQ statistics
void faiss_IndexIVFPQStats_reset(FaissIndexIVFPQStats* stats);

#ifdef __cplusplus
}
#endif

#endif