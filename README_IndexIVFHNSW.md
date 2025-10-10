# IndexIVFHNSW C API

This document describes the newly implemented `IndexIVFHNSW` class and its C API, which combines the Inverted File (IVF) structure with HNSW (Hierarchical Navigable Small World) as the quantizer.

## Overview

`IndexIVFHNSW` is a hybrid index that uses an HNSW index as the quantizer for an IVF structure. This provides the benefits of both approaches:
- **HNSW quantizer**: Fast approximate nearest neighbor search for coarse quantization
- **IVF structure**: Efficient storage and retrieval of vectors in inverted lists

## Files Added

### C++ Implementation
- `faiss/IndexIVFHNSW.h` - Header file with class definition
- `faiss/IndexIVFHNSW.cpp` - Implementation file

### C API
- `c_api/IndexIVFHNSW_c.h` - C API header file
- `c_api/IndexIVFHNSW_c.cpp` - C API implementation

### Build Configuration
- Updated `faiss/CMakeLists.txt` to include `IndexIVFHNSW.cpp`
- Updated `c_api/CMakeLists.txt` to include `IndexIVFHNSW_c.cpp`

## C API Functions

### Constructor Functions
```c
// Create with default parameters
int faiss_IndexIVFHNSW_new(FaissIndexIVFHNSW** p_index);

// Create with specific parameters
int faiss_IndexIVFHNSW_new_with(
    FaissIndexIVFHNSW** p_index,
    int d,                    // dimension
    size_t nlist,            // number of clusters
    int M,                   // HNSW parameter: number of bi-directional links
    int efConstruction,      // HNSW parameter: construction search width
    int efSearch,            // HNSW parameter: search width
    FaissMetricType metric   // distance metric
);

// Create with default HNSW parameters
int faiss_IndexIVFHNSW_new_with_defaults(
    FaissIndexIVFHNSW** p_index,
    int d,
    size_t nlist
);
```

### Getter/Setter Functions
```c
// Basic IVF parameters
size_t faiss_IndexIVFHNSW_nlist(const FaissIndexIVFHNSW* index);
size_t faiss_IndexIVFHNSW_nprobe(const FaissIndexIVFHNSW* index);
void faiss_IndexIVFHNSW_set_nprobe(FaissIndexIVFHNSW* index, size_t nprobe);

// HNSW parameters
int faiss_IndexIVFHNSW_M(const FaissIndexIVFHNSW* index);
void faiss_IndexIVFHNSW_set_M(FaissIndexIVFHNSW* index, int M);

int faiss_IndexIVFHNSW_efConstruction(const FaissIndexIVFHNSW* index);
void faiss_IndexIVFHNSW_set_efConstruction(FaissIndexIVFHNSW* index, int efConstruction);

int faiss_IndexIVFHNSW_efSearch(const FaissIndexIVFHNSW* index);
void faiss_IndexIVFHNSW_set_efSearch(FaissIndexIVFHNSW* index, int efSearch);

// Quantizer access
FaissIndex* faiss_IndexIVFHNSW_quantizer(const FaissIndexIVFHNSW* index);
```

### Utility Functions
```c
// Get the HNSW quantizer
int faiss_IndexIVFHNSW_get_hnsw_quantizer(
    const FaissIndexIVFHNSW* index,
    FaissIndex** quantizer
);

// Set multiple HNSW parameters at once
int faiss_IndexIVFHNSW_set_hnsw_params(
    FaissIndexIVFHNSW* index,
    int M,
    int efConstruction,
    int efSearch
);

// Get multiple HNSW parameters at once
int faiss_IndexIVFHNSW_get_hnsw_params(
    const FaissIndexIVFHNSW* index,
    int* M,
    int* efConstruction,
    int* efSearch
);
```

### Standard Index Functions
All standard index functions from the base `Index` class are available:
- `faiss_Index_train()` - Train the index
- `faiss_Index_add()` - Add vectors to the index
- `faiss_Index_add_with_ids()` - Add vectors with specific IDs
- `faiss_Index_search()` - Search for nearest neighbors
- `faiss_Index_range_search()` - Range search
- `faiss_Index_reset()` - Reset the index
- `faiss_IndexIVFHNSW_free()` - Free the index

## Usage Example

```c
#include <c_api/IndexIVFHNSW_c.h>
#include <c_api/Index_c.h>

int main() {
    // Create index
    FaissIndexIVFHNSW* index = nullptr;
    int ret = faiss_IndexIVFHNSW_new_with(
        &index, 128, 100, 32, 200, 50, METRIC_L2);
    
    if (ret != 0) {
        // Handle error
        return 1;
    }
    
    // Train the index
    float* training_data = /* your training vectors */;
    ret = faiss_Index_train(index, n_train, training_data);
    
    // Add vectors
    float* vectors = /* your vectors */;
    ret = faiss_Index_add(index, n_vectors, vectors);
    
    // Search
    float* queries = /* your query vectors */;
    float* distances = /* output distances */;
    idx_t* labels = /* output labels */;
    ret = faiss_Index_search(index, n_queries, queries, k, distances, labels);
    
    // Clean up
    faiss_IndexIVFHNSW_free(index);
    return 0;
}
```

## Implementation Notes

1. **Simplified Implementation**: The current implementation is a simplified version that uses the HNSW quantizer for both quantization and storage. A more sophisticated implementation would maintain separate inverted lists and use HNSW only for quantization.

2. **Parameter Management**: HNSW parameters can be set both during construction and at runtime using the setter functions.

3. **Memory Management**: The index owns the HNSW quantizer and will clean it up when destroyed.

4. **Error Handling**: All C API functions return error codes that should be checked for proper error handling.

## Future Improvements

1. **True IVF Implementation**: Implement proper inverted lists with HNSW as quantizer only
2. **Parameter Validation**: Add validation for HNSW parameters
3. **Performance Optimization**: Optimize the search and add operations
4. **Serialization**: Add support for saving/loading the index
5. **Thread Safety**: Ensure thread safety for concurrent operations

## Testing

A test program `test_indexivfhnsw.cpp` is provided to verify the basic functionality of the C API. The test creates an index, trains it, adds vectors, and performs searches.