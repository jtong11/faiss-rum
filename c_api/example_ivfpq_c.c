/*
 * Example program demonstrating IndexIVFPQ C API usage
 * 
 * This example shows how to:
 * 1. Create an IndexIVFPQ with a quantizer
 * 2. Train the index
 * 3. Add vectors to the index
 * 4. Search for similar vectors
 * 5. Use various IndexIVFPQ-specific features
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "IndexIVFPQ_c.h"
#include "IndexFlat_c.h"
#include "Index_c.h"
#include "error_c.h"

#define FAISS_TRY(C)                                       \
    {                                                      \
        if (C) {                                           \
            fprintf(stderr, "Error: %s\n", faiss_get_last_error()); \
            exit(-1);                                      \
        }                                                  \
    }

int main() {
    printf("IndexIVFPQ C API Example\n");
    printf("========================\n\n");
    
    const int d = 128;           // dimension
    const int nlist = 100;       // number of clusters
    const int M = 8;             // number of subquantizers
    const int nbits = 8;         // bits per subquantizer
    const int n_vectors = 1000;  // number of vectors to add
    const int n_queries = 10;    // number of query vectors
    const int k = 5;             // number of nearest neighbors to search
    
    // Generate random data
    printf("Generating random data...\n");
    float* vectors = (float*)malloc(n_vectors * d * sizeof(float));
    float* queries = (float*)malloc(n_queries * d * sizeof(float));
    
    srand(42); // for reproducible results
    for (int i = 0; i < n_vectors * d; i++) {
        vectors[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < n_queries * d; i++) {
        queries[i] = (float)rand() / RAND_MAX;
    }
    
    // Create a quantizer
    printf("Creating quantizer...\n");
    FaissIndexFlat* quantizer = NULL;
    FAISS_TRY(faiss_IndexFlat_new_with(&quantizer, d, METRIC_L2));
    
    // Create an IndexIVFPQ
    printf("Creating IndexIVFPQ...\n");
    FaissIndexIVFPQ* index = NULL;
    FAISS_TRY(faiss_IndexIVFPQ_new(&index, (FaissIndex*)quantizer, d, nlist, M, nbits, METRIC_L2));
    
    // Display IndexIVFPQ properties
    printf("\nIndexIVFPQ Properties:\n");
    printf("  Dimension: %ld\n", faiss_Index_d(index));
    printf("  Number of vectors: %ld\n", faiss_Index_ntotal(index));
    printf("  Is trained: %d\n", faiss_Index_is_trained(index));
    printf("  PQ M: %zu\n", faiss_IndexIVFPQ_pq_M(index));
    printf("  PQ nbits: %zu\n", faiss_IndexIVFPQ_pq_nbits(index));
    printf("  PQ code_size: %zu\n", faiss_IndexIVFPQ_pq_code_size(index));
    printf("  PQ dsub: %zu\n", faiss_IndexIVFPQ_pq_dsub(index));
    printf("  PQ ksub: %zu\n", faiss_IndexIVFPQ_pq_ksub(index));
    
    // Configure IndexIVFPQ parameters
    printf("\nConfiguring IndexIVFPQ parameters...\n");
    faiss_IndexIVFPQ_set_do_polysemous_training(index, 0);
    faiss_IndexIVFPQ_set_scan_table_threshold(index, 0);
    faiss_IndexIVFPQ_set_polysemous_ht(index, 0);
    faiss_IndexIVFPQ_set_use_precomputed_table(index, 0);
    
    printf("  do_polysemous_training: %d\n", faiss_IndexIVFPQ_do_polysemous_training(index));
    printf("  scan_table_threshold: %zu\n", faiss_IndexIVFPQ_scan_table_threshold(index));
    printf("  polysemous_ht: %d\n", faiss_IndexIVFPQ_polysemous_ht(index));
    printf("  use_precomputed_table: %d\n", faiss_IndexIVFPQ_use_precomputed_table(index));
    
    // Train the index
    printf("\nTraining the index...\n");
    FAISS_TRY(faiss_Index_train(index, n_vectors, vectors));
    printf("  Training completed. Is trained: %d\n", faiss_Index_is_trained(index));
    
    // Add vectors to the index
    printf("\nAdding vectors to the index...\n");
    FAISS_TRY(faiss_Index_add(index, n_vectors, vectors));
    printf("  Added %d vectors. Total vectors: %ld\n", n_vectors, faiss_Index_ntotal(index));
    
    // Test encoding/decoding
    printf("\nTesting encoding/decoding...\n");
    if (n_vectors > 0) {
        uint8_t* codes = (uint8_t*)malloc(n_vectors * faiss_IndexIVFPQ_pq_code_size(index));
        float* decoded = (float*)malloc(n_vectors * d * sizeof(float));
        idx_t* keys = (idx_t*)malloc(n_vectors * sizeof(idx_t));
        
        // Encode vectors
        FAISS_TRY(faiss_IndexIVFPQ_encode_multiple(index, n_vectors, keys, vectors, codes, 1));
        printf("  Encoded %d vectors\n", n_vectors);
        
        // Decode vectors
        FAISS_TRY(faiss_IndexIVFPQ_decode_multiple(index, n_vectors, keys, codes, decoded));
        printf("  Decoded %d vectors\n", n_vectors);
        
        // Check reconstruction quality
        float mse = 0.0f;
        for (int i = 0; i < n_vectors * d; i++) {
            float diff = vectors[i] - decoded[i];
            mse += diff * diff;
        }
        mse /= (n_vectors * d);
        printf("  Reconstruction MSE: %f\n", mse);
        
        free(codes);
        free(decoded);
        free(keys);
    }
    
    // Search for similar vectors
    printf("\nSearching for similar vectors...\n");
    float* distances = (float*)malloc(n_queries * k * sizeof(float));
    idx_t* labels = (idx_t*)malloc(n_queries * k * sizeof(idx_t));
    
    FAISS_TRY(faiss_Index_search(index, n_queries, queries, k, distances, labels));
    
    printf("  Search completed. Results for first query:\n");
    for (int i = 0; i < k; i++) {
        printf("    %d: label=%ld, distance=%f\n", i, labels[i], distances[i]);
    }
    
    // Test downcast
    printf("\nTesting downcast...\n");
    FaissIndexIVFPQ* casted = faiss_IndexIVFPQ_cast((FaissIndex*)index);
    if (casted == index) {
        printf("  Downcast successful!\n");
    } else {
        printf("  Downcast failed!\n");
    }
    
    // Test statistics
    printf("\nIndexIVFPQ Statistics:\n");
    FaissIndexIVFPQStats* stats = faiss_get_indexIVFPQ_stats();
    printf("  nrefine: %zu\n", stats->nrefine);
    printf("  n_hamming_pass: %zu\n", stats->n_hamming_pass);
    printf("  search_cycles: %zu\n", stats->search_cycles);
    printf("  refine_cycles: %zu\n", stats->refine_cycles);
    
    // Test precomputed table settings
    printf("\nPrecomputed table settings:\n");
    size_t max_bytes = faiss_get_precomputed_table_max_bytes();
    printf("  Max bytes: %zu (%.2f GB)\n", max_bytes, max_bytes / (1024.0 * 1024.0 * 1024.0));
    
    // Test find_duplicates (if there are any)
    printf("\nTesting find_duplicates...\n");
    idx_t* dup_ids = (idx_t*)malloc(n_vectors * sizeof(idx_t));
    size_t* lims = (size_t*)malloc((n_vectors / 2 + 1) * sizeof(size_t));
    size_t n_duplicates = 0;
    
    FAISS_TRY(faiss_IndexIVFPQ_find_duplicates(index, dup_ids, lims, &n_duplicates));
    printf("  Found %zu duplicate groups\n", n_duplicates);
    
    // Test train_encoder_num_vectors
    printf("\nTraining requirements:\n");
    idx_t required_vectors = faiss_IndexIVFPQ_train_encoder_num_vectors(index);
    printf("  Required vectors for training: %ld\n", required_vectors);
    
    // Clean up
    printf("\nCleaning up...\n");
    free(vectors);
    free(queries);
    free(distances);
    free(labels);
    free(dup_ids);
    free(lims);
    
    faiss_IndexIVFPQ_free(index);
    faiss_IndexFlat_free(quantizer);
    
    printf("\nExample completed successfully!\n");
    return 0;
}