/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexIVFHNSW_c.h"
#include <faiss/IndexIVFHNSW.h>
#include <faiss/IndexHNSW.h>
#include "Index_c.h"
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexIVFHNSW;
using faiss::IndexHNSW;
using faiss::MetricType;

// Helper function to convert FaissMetricType to MetricType
MetricType convert_metric_type(FaissMetricType metric) {
    switch (metric) {
        case METRIC_INNER_PRODUCT:
            return faiss::METRIC_INNER_PRODUCT;
        case METRIC_L2:
            return faiss::METRIC_L2;
        case METRIC_L1:
            return faiss::METRIC_L1;
        case METRIC_Linf:
            return faiss::METRIC_Linf;
        case METRIC_Lp:
            return faiss::METRIC_Lp;
        case METRIC_Canberra:
            return faiss::METRIC_Canberra;
        case METRIC_BrayCurtis:
            return faiss::METRIC_BrayCurtis;
        case METRIC_JensenShannon:
            return faiss::METRIC_JensenShannon;
        default:
            return faiss::METRIC_L2;
    }
}

DEFINE_DESTRUCTOR(IndexIVFHNSW)
DEFINE_INDEX_DOWNCAST(IndexIVFHNSW)

/// number of possible key values
DEFINE_GETTER(IndexIVFHNSW, size_t, nlist)
/// number of probes at query time
DEFINE_GETTER(IndexIVFHNSW, size_t, nprobe)
DEFINE_SETTER(IndexIVFHNSW, size_t, nprobe)

/// quantizer that maps vectors to inverted lists
DEFINE_GETTER_PERMISSIVE(IndexIVFHNSW, FaissIndex*, quantizer)

/// HNSW parameter: number of bi-directional links for each node
DEFINE_GETTER(IndexIVFHNSW, int, M)
DEFINE_SETTER(IndexIVFHNSW, int, M)

/// HNSW parameter: size of the dynamic candidate list for construction
DEFINE_GETTER(IndexIVFHNSW, int, efConstruction)
DEFINE_SETTER(IndexIVFHNSW, int, efConstruction)

/// HNSW parameter: size of the dynamic candidate list for search
DEFINE_GETTER(IndexIVFHNSW, int, efSearch)
DEFINE_SETTER(IndexIVFHNSW, int, efSearch)

int faiss_IndexIVFHNSW_new(FaissIndexIVFHNSW** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexIVFHNSW*>(new IndexIVFHNSW());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFHNSW_new_with(
        FaissIndexIVFHNSW** p_index,
        int d,
        size_t nlist,
        int M,
        int efConstruction,
        int efSearch,
        FaissMetricType metric) {
    try {
        *p_index = reinterpret_cast<FaissIndexIVFHNSW*>(
                new IndexIVFHNSW(d, nlist, M, efConstruction, efSearch, convert_metric_type(metric)));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFHNSW_new_with_defaults(
        FaissIndexIVFHNSW** p_index,
        int d,
        size_t nlist) {
    try {
        *p_index = reinterpret_cast<FaissIndexIVFHNSW*>(
                new IndexIVFHNSW(d, nlist));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFHNSW_get_hnsw_quantizer(
        const FaissIndexIVFHNSW* index,
        FaissIndex** quantizer) {
    try {
        const IndexIVFHNSW* idx = reinterpret_cast<const IndexIVFHNSW*>(index);
        IndexHNSW* hnsw_quantizer = const_cast<IndexHNSW*>(idx->get_hnsw_quantizer());
        *quantizer = reinterpret_cast<FaissIndex*>(hnsw_quantizer);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFHNSW_set_hnsw_params(
        FaissIndexIVFHNSW* index,
        int M,
        int efConstruction,
        int efSearch) {
    try {
        IndexIVFHNSW* idx = reinterpret_cast<IndexIVFHNSW*>(index);
        idx->set_hnsw_params(M, efConstruction, efSearch);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFHNSW_get_hnsw_params(
        const FaissIndexIVFHNSW* index,
        int* M,
        int* efConstruction,
        int* efSearch) {
    try {
        const IndexIVFHNSW* idx = reinterpret_cast<const IndexIVFHNSW*>(index);
        idx->get_hnsw_params(M, efConstruction, efSearch);
    }
    CATCH_AND_HANDLE
}