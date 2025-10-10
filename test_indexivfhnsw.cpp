/*
 * Test program for IndexIVFHNSW C API
 */

#include <iostream>
#include <vector>
#include <random>
#include <c_api/IndexIVFHNSW_c.h>
#include <c_api/Index_c.h>
#include <c_api/faiss_c.h>

int main() {
    std::cout << "Testing IndexIVFHNSW C API..." << std::endl;
    
    // Test parameters
    const int d = 128;           // dimension
    const size_t nlist = 100;    // number of clusters
    const int M = 32;            // HNSW parameter
    const int efConstruction = 200;
    const int efSearch = 50;
    const int n_vectors = 1000;  // number of vectors to add
    const int n_queries = 10;    // number of query vectors
    const int k = 5;             // number of nearest neighbors to search
    
    // Create random vectors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> vectors(n_vectors * d);
    std::vector<float> queries(n_queries * d);
    
    for (int i = 0; i < n_vectors * d; ++i) {
        vectors[i] = dis(gen);
    }
    
    for (int i = 0; i < n_queries * d; ++i) {
        queries[i] = dis(gen);
    }
    
    // Create IndexIVFHNSW
    FaissIndexIVFHNSW* index = nullptr;
    int ret = faiss_IndexIVFHNSW_new_with(
        &index, d, nlist, M, efConstruction, efSearch, METRIC_L2);
    
    if (ret != 0) {
        std::cerr << "Failed to create IndexIVFHNSW: " << ret << std::endl;
        return 1;
    }
    
    std::cout << "Created IndexIVFHNSW successfully" << std::endl;
    
    // Train the index
    ret = faiss_Index_train(index, n_vectors, vectors.data());
    if (ret != 0) {
        std::cerr << "Failed to train index: " << ret << std::endl;
        faiss_IndexIVFHNSW_free(index);
        return 1;
    }
    
    std::cout << "Trained index successfully" << std::endl;
    
    // Add vectors
    ret = faiss_Index_add(index, n_vectors, vectors.data());
    if (ret != 0) {
        std::cerr << "Failed to add vectors: " << ret << std::endl;
        faiss_IndexIVFHNSW_free(index);
        return 1;
    }
    
    std::cout << "Added " << n_vectors << " vectors successfully" << std::endl;
    
    // Search
    std::vector<float> distances(n_queries * k);
    std::vector<idx_t> labels(n_queries * k);
    
    ret = faiss_Index_search(index, n_queries, queries.data(), k, 
                            distances.data(), labels.data());
    if (ret != 0) {
        std::cerr << "Failed to search: " << ret << std::endl;
        faiss_IndexIVFHNSW_free(index);
        return 1;
    }
    
    std::cout << "Search completed successfully" << std::endl;
    
    // Print some results
    for (int i = 0; i < n_queries; ++i) {
        std::cout << "Query " << i << " results:" << std::endl;
        for (int j = 0; j < k; ++j) {
            int idx = i * k + j;
            std::cout << "  " << j << ": label=" << labels[idx] 
                     << ", distance=" << distances[idx] << std::endl;
        }
    }
    
    // Test getters
    size_t nlist_val = faiss_IndexIVFHNSW_nlist(index);
    size_t nprobe_val = faiss_IndexIVFHNSW_nprobe(index);
    int M_val = faiss_IndexIVFHNSW_M(index);
    int efConstruction_val = faiss_IndexIVFHNSW_efConstruction(index);
    int efSearch_val = faiss_IndexIVFHNSW_efSearch(index);
    
    std::cout << "Index parameters:" << std::endl;
    std::cout << "  nlist: " << nlist_val << std::endl;
    std::cout << "  nprobe: " << nprobe_val << std::endl;
    std::cout << "  M: " << M_val << std::endl;
    std::cout << "  efConstruction: " << efConstruction_val << std::endl;
    std::cout << "  efSearch: " << efSearch_val << std::endl;
    
    // Test setters
    faiss_IndexIVFHNSW_set_nprobe(index, 10);
    faiss_IndexIVFHNSW_set_efSearch(index, 100);
    
    std::cout << "Updated nprobe to: " << faiss_IndexIVFHNSW_nprobe(index) << std::endl;
    std::cout << "Updated efSearch to: " << faiss_IndexIVFHNSW_efSearch(index) << std::endl;
    
    // Clean up
    faiss_IndexIVFHNSW_free(index);
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}