# IndexIVFPQ C API

This document describes the C API for IndexIVFPQ (Inverted File with Product Quantizer), which provides efficient approximate nearest neighbor search using product quantization.

## Overview

IndexIVFPQ combines two techniques:
1. **Inverted File (IVF)**: Divides the vector space into clusters and only searches the most relevant clusters
2. **Product Quantizer (PQ)**: Compresses vectors using product quantization for memory efficiency

## Header Files

```c
#include "IndexIVFPQ_c.h"
```

## Data Types

### FaissIndexIVFPQ
Opaque pointer type representing an IndexIVFPQ instance.

### FaissIndexIVFPQStats
Structure containing statistics about IndexIVFPQ operations:

```c
typedef struct FaissIndexIVFPQStats {
    size_t nrefine;           ///< number of refines (IVFPQR)
    size_t n_hamming_pass;    ///< number of passed Hamming distance tests (for polysemous)
    size_t search_cycles;     ///< timings measured with the CPU RTC on all threads
    size_t refine_cycles;     ///< only for IVFPQR
} FaissIndexIVFPQStats;
```

## Constructor Functions

### faiss_IndexIVFPQ_new
Creates a new IndexIVFPQ with specified parameters.

```c
int faiss_IndexIVFPQ_new(
    FaissIndexIVFPQ** p_index,
    FaissIndex* quantizer,
    size_t d,
    size_t nlist,
    size_t M,
    size_t nbits_per_idx,
    FaissMetricType metric
);
```

**Parameters:**
- `p_index`: Output pointer to the created index
- `quantizer`: Quantizer used for coarse quantization
- `d`: Dimension of the vectors
- `nlist`: Number of clusters (inverted lists)
- `M`: Number of subquantizers
- `nbits_per_idx`: Number of bits per subquantizer
- `metric`: Distance metric (METRIC_L2, METRIC_INNER_PRODUCT, etc.)

**Returns:** 0 on success, error code on failure

### faiss_IndexIVFPQ_new_with
Creates a new IndexIVFPQ with additional parameters.

```c
int faiss_IndexIVFPQ_new_with(
    FaissIndexIVFPQ** p_index,
    FaissIndex* quantizer,
    size_t d,
    size_t nlist,
    size_t M,
    size_t nbits_per_idx,
    FaissMetricType metric,
    int own_invlists
);
```

**Additional Parameters:**
- `own_invlists`: Whether the index owns the inverted lists

### faiss_IndexIVFPQ_new_default
Creates a new IndexIVFPQ with default parameters.

```c
int faiss_IndexIVFPQ_new_default(FaissIndexIVFPQ** p_index);
```

## Destructor

### faiss_IndexIVFPQ_free
Frees an IndexIVFPQ instance.

```c
void faiss_IndexIVFPQ_free(FaissIndexIVFPQ* index);
```

## Type Casting

### faiss_IndexIVFPQ_cast
Casts a generic FaissIndex to FaissIndexIVFPQ.

```c
FaissIndexIVFPQ* faiss_IndexIVFPQ_cast(FaissIndex* index);
```

**Returns:** Pointer to IndexIVFPQ if cast is valid, NULL otherwise

## Product Quantizer Properties

### faiss_IndexIVFPQ_pq_M
Gets the number of subquantizers.

```c
size_t faiss_IndexIVFPQ_pq_M(const FaissIndexIVFPQ* index);
```

### faiss_IndexIVFPQ_pq_nbits
Gets the number of bits per subquantizer.

```c
size_t faiss_IndexIVFPQ_pq_nbits(const FaissIndexIVFPQ* index);
```

### faiss_IndexIVFPQ_pq_code_size
Gets the size of the encoded vectors in bytes.

```c
size_t faiss_IndexIVFPQ_pq_code_size(const FaissIndexIVFPQ* index);
```

### faiss_IndexIVFPQ_pq_dsub
Gets the dimension of each subvector.

```c
size_t faiss_IndexIVFPQ_pq_dsub(const FaissIndexIVFPQ* index);
```

### faiss_IndexIVFPQ_pq_ksub
Gets the number of centroids per subquantizer.

```c
size_t faiss_IndexIVFPQ_pq_ksub(const FaissIndexIVFPQ* index);
```

## Configuration Parameters

### Polysemous Training

```c
// Get/set whether polysemous training is enabled
int faiss_IndexIVFPQ_do_polysemous_training(const FaissIndexIVFPQ* index);
void faiss_IndexIVFPQ_set_do_polysemous_training(FaissIndexIVFPQ* index, int value);
```

### Scan Table Threshold

```c
// Get/set the scan table threshold
size_t faiss_IndexIVFPQ_scan_table_threshold(const FaissIndexIVFPQ* index);
void faiss_IndexIVFPQ_set_scan_table_threshold(FaissIndexIVFPQ* index, size_t value);
```

### Polysemous Hamming Threshold

```c
// Get/set the polysemous Hamming threshold
int faiss_IndexIVFPQ_polysemous_ht(const FaissIndexIVFPQ* index);
void faiss_IndexIVFPQ_set_polysemous_ht(FaissIndexIVFPQ* index, int value);
```

### Precomputed Table

```c
// Get/set whether precomputed table is used
int faiss_IndexIVFPQ_use_precomputed_table(const FaissIndexIVFPQ* index);
void faiss_IndexIVFPQ_set_use_precomputed_table(FaissIndexIVFPQ* index, int value);
```

## Encoding and Decoding

### faiss_IndexIVFPQ_encode
Encodes a single vector.

```c
int faiss_IndexIVFPQ_encode(
    const FaissIndexIVFPQ* index,
    idx_t key,
    const float* x,
    uint8_t* code
);
```

**Parameters:**
- `index`: IndexIVFPQ instance
- `key`: Quantization key (list number)
- `x`: Input vector
- `code`: Output encoded vector

### faiss_IndexIVFPQ_encode_multiple
Encodes multiple vectors.

```c
int faiss_IndexIVFPQ_encode_multiple(
    const FaissIndexIVFPQ* index,
    size_t n,
    idx_t* keys,
    const float* x,
    uint8_t* codes,
    int compute_keys
);
```

**Parameters:**
- `n`: Number of vectors
- `keys`: Quantization keys (can be NULL if compute_keys=1)
- `x`: Input vectors (n * d)
- `codes`: Output encoded vectors (n * code_size)
- `compute_keys`: Whether to compute keys automatically

### faiss_IndexIVFPQ_decode_multiple
Decodes multiple vectors.

```c
int faiss_IndexIVFPQ_decode_multiple(
    const FaissIndexIVFPQ* index,
    size_t n,
    const idx_t* keys,
    const uint8_t* codes,
    float* x
);
```

## Utility Functions

### faiss_IndexIVFPQ_find_duplicates
Finds exact duplicates in the dataset.

```c
int faiss_IndexIVFPQ_find_duplicates(
    const FaissIndexIVFPQ* index,
    idx_t* ids,
    size_t* lims,
    size_t* n_duplicates
);
```

**Parameters:**
- `ids`: Output array for duplicate IDs
- `lims`: Output array for group limits
- `n_duplicates`: Output number of duplicate groups found

### faiss_IndexIVFPQ_precompute_table
Builds precomputed tables for faster search.

```c
int faiss_IndexIVFPQ_precompute_table(FaissIndexIVFPQ* index);
```

### faiss_IndexIVFPQ_train_encoder_num_vectors
Gets the number of vectors needed for training the encoder.

```c
idx_t faiss_IndexIVFPQ_train_encoder_num_vectors(const FaissIndexIVFPQ* index);
```

### faiss_IndexIVFPQ_train_encoder
Trains the encoder.

```c
int faiss_IndexIVFPQ_train_encoder(
    FaissIndexIVFPQ* index,
    idx_t n,
    const float* x,
    const idx_t* assign
);
```

### faiss_IndexIVFPQ_reconstruct_from_offset
Reconstructs a vector from its offset in an inverted list.

```c
int faiss_IndexIVFPQ_reconstruct_from_offset(
    const FaissIndexIVFPQ* index,
    int64_t list_no,
    int64_t offset,
    float* recons
);
```

## Global Settings

### Precomputed Table Max Bytes

```c
// Get/set the maximum bytes for precomputed tables
size_t faiss_get_precomputed_table_max_bytes(void);
void faiss_set_precomputed_table_max_bytes(size_t value);
```

## Statistics

### faiss_get_indexIVFPQ_stats
Gets the global IndexIVFPQ statistics.

```c
FaissIndexIVFPQStats* faiss_get_indexIVFPQ_stats(void);
```

### faiss_IndexIVFPQStats_reset
Resets the statistics.

```c
void faiss_IndexIVFPQStats_reset(FaissIndexIVFPQStats* stats);
```

## Example Usage

```c
#include "IndexIVFPQ_c.h"
#include "IndexFlat_c.h"
#include "Index_c.h"
#include "error_c.h"

int main() {
    // Create a quantizer
    FaissIndexFlat* quantizer = NULL;
    faiss_IndexFlat_new_with(&quantizer, 128, METRIC_L2);
    
    // Create an IndexIVFPQ
    FaissIndexIVFPQ* index = NULL;
    faiss_IndexIVFPQ_new(&index, (FaissIndex*)quantizer, 128, 100, 8, 8, METRIC_L2);
    
    // Configure parameters
    faiss_IndexIVFPQ_set_do_polysemous_training(index, 0);
    faiss_IndexIVFPQ_set_use_precomputed_table(index, 0);
    
    // Train the index
    float* training_data = /* your training data */;
    faiss_Index_train(index, n_training, training_data);
    
    // Add vectors
    float* vectors = /* your vectors */;
    faiss_Index_add(index, n_vectors, vectors);
    
    // Search
    float* queries = /* your queries */;
    float* distances = malloc(n_queries * k * sizeof(float));
    idx_t* labels = malloc(n_queries * k * sizeof(idx_t));
    faiss_Index_search(index, n_queries, queries, k, distances, labels);
    
    // Clean up
    faiss_IndexIVFPQ_free(index);
    faiss_IndexFlat_free(quantizer);
    free(distances);
    free(labels);
    
    return 0;
}
```

## Error Handling

All functions return 0 on success and a non-zero error code on failure. Use `faiss_get_last_error()` to get the error message:

```c
if (faiss_IndexIVFPQ_new(&index, quantizer, d, nlist, M, nbits, metric)) {
    fprintf(stderr, "Error: %s\n", faiss_get_last_error());
    exit(-1);
}
```

## Notes

- IndexIVFPQ inherits from IndexIVF, so all IndexIVF functions are also available
- The index must be trained before adding vectors
- Product quantization parameters (M, nbits) should be chosen based on your memory constraints and accuracy requirements
- Precomputed tables can speed up search but use more memory
- Polysemous filtering can improve search speed by filtering out distant vectors early