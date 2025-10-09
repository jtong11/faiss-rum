/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexIVFPQ_c.h"
#include <faiss/IndexIVFPQ.h>
#include "IndexIVF_c.h"
#include "Index_c.h"
#include "macros_impl.h"

using faiss::IndexIVFPQ;
using faiss::IndexIVFPQStats;
using faiss::precomputed_table_max_bytes;

/// IndexIVFPQ definitions

DEFINE_DESTRUCTOR(IndexIVFPQ)
DEFINE_INDEX_DOWNCAST(IndexIVFPQ)

int faiss_IndexIVFPQ_new(
        FaissIndexIVFPQ** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        FaissMetricType metric) {
    try {
        IndexIVFPQ* index = new IndexIVFPQ(
                reinterpret_cast<faiss::Index*>(quantizer),
                d,
                nlist,
                M,
                nbits_per_idx,
                static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexIVFPQ*>(index);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFPQ_new_with(
        FaissIndexIVFPQ** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        FaissMetricType metric,
        int own_invlists) {
    try {
        IndexIVFPQ* index = new IndexIVFPQ(
                reinterpret_cast<faiss::Index*>(quantizer),
                d,
                nlist,
                M,
                nbits_per_idx,
                static_cast<faiss::MetricType>(metric),
                static_cast<bool>(own_invlists));
        *p_index = reinterpret_cast<FaissIndexIVFPQ*>(index);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFPQ_new_default(FaissIndexIVFPQ** p_index) {
    try {
        IndexIVFPQ* index = new IndexIVFPQ();
        *p_index = reinterpret_cast<FaissIndexIVFPQ*>(index);
    }
    CATCH_AND_HANDLE
}

/// ProductQuantizer getters
size_t faiss_IndexIVFPQ_pq_M(const FaissIndexIVFPQ* obj) {
    return reinterpret_cast<const faiss::IndexIVFPQ*>(obj)->pq.M;
}

size_t faiss_IndexIVFPQ_pq_nbits(const FaissIndexIVFPQ* obj) {
    return reinterpret_cast<const faiss::IndexIVFPQ*>(obj)->pq.nbits;
}

size_t faiss_IndexIVFPQ_pq_code_size(const FaissIndexIVFPQ* obj) {
    return reinterpret_cast<const faiss::IndexIVFPQ*>(obj)->pq.code_size;
}

size_t faiss_IndexIVFPQ_pq_dsub(const FaissIndexIVFPQ* obj) {
    return reinterpret_cast<const faiss::IndexIVFPQ*>(obj)->pq.dsub;
}

size_t faiss_IndexIVFPQ_pq_ksub(const FaissIndexIVFPQ* obj) {
    return reinterpret_cast<const faiss::IndexIVFPQ*>(obj)->pq.ksub;
}

/// Polysemous training getters/setters
DEFINE_GETTER(IndexIVFPQ, int, do_polysemous_training)
DEFINE_SETTER(IndexIVFPQ, int, do_polysemous_training)

/// Scan table threshold getters/setters
DEFINE_GETTER(IndexIVFPQ, size_t, scan_table_threshold)
DEFINE_SETTER(IndexIVFPQ, size_t, scan_table_threshold)

/// Polysemous Hamming threshold getters/setters
DEFINE_GETTER(IndexIVFPQ, int, polysemous_ht)
DEFINE_SETTER(IndexIVFPQ, int, polysemous_ht)

/// Precomputed table getters/setters
DEFINE_GETTER(IndexIVFPQ, int, use_precomputed_table)
DEFINE_SETTER(IndexIVFPQ, int, use_precomputed_table)

int faiss_IndexIVFPQ_encode(
        const FaissIndexIVFPQ* index,
        idx_t key,
        const float* x,
        uint8_t* code) {
    try {
        reinterpret_cast<const IndexIVFPQ*>(index)->encode(key, x, code);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFPQ_encode_multiple(
        const FaissIndexIVFPQ* index,
        size_t n,
        idx_t* keys,
        const float* x,
        uint8_t* codes,
        int compute_keys) {
    try {
        reinterpret_cast<const IndexIVFPQ*>(index)->encode_multiple(
                n, keys, x, codes, static_cast<bool>(compute_keys));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFPQ_decode_multiple(
        const FaissIndexIVFPQ* index,
        size_t n,
        const idx_t* keys,
        const uint8_t* codes,
        float* x) {
    try {
        reinterpret_cast<const IndexIVFPQ*>(index)->decode_multiple(
                n, keys, codes, x);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFPQ_find_duplicates(
        const FaissIndexIVFPQ* index,
        idx_t* ids,
        size_t* lims,
        size_t* n_duplicates) {
    try {
        *n_duplicates = reinterpret_cast<const IndexIVFPQ*>(index)->find_duplicates(ids, lims);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFPQ_precompute_table(FaissIndexIVFPQ* index) {
    try {
        reinterpret_cast<IndexIVFPQ*>(index)->precompute_table();
    }
    CATCH_AND_HANDLE
}

idx_t faiss_IndexIVFPQ_train_encoder_num_vectors(const FaissIndexIVFPQ* index) {
    return reinterpret_cast<const IndexIVFPQ*>(index)->train_encoder_num_vectors();
}

int faiss_IndexIVFPQ_train_encoder(
        FaissIndexIVFPQ* index,
        idx_t n,
        const float* x,
        const idx_t* assign) {
    try {
        reinterpret_cast<IndexIVFPQ*>(index)->train_encoder(n, x, assign);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFPQ_reconstruct_from_offset(
        const FaissIndexIVFPQ* index,
        int64_t list_no,
        int64_t offset,
        float* recons) {
    try {
        reinterpret_cast<const IndexIVFPQ*>(index)->reconstruct_from_offset(
                list_no, offset, recons);
    }
    CATCH_AND_HANDLE
}

size_t faiss_get_precomputed_table_max_bytes(void) {
    return precomputed_table_max_bytes;
}

void faiss_set_precomputed_table_max_bytes(size_t value) {
    precomputed_table_max_bytes = value;
}

FaissIndexIVFPQStats* faiss_get_indexIVFPQ_stats(void) {
    return reinterpret_cast<FaissIndexIVFPQStats*>(&faiss::indexIVFPQ_stats);
}

void faiss_IndexIVFPQStats_reset(FaissIndexIVFPQStats* stats) {
    reinterpret_cast<IndexIVFPQStats*>(stats)->reset();
}