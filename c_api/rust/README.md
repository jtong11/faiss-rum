# Rust IVF-RaBitQ example via Faiss C API

This crate demonstrates calling Faiss from Rust and creating an IVF-RaBitQ
index with an index factory string:

- `IVF{nlist},RaBitQ`

The implementation performs a runtime version check and requires:

- Faiss `>= 1.11.0` (IVF-RaBitQ support target)

## Build prerequisites

Build Faiss with C API enabled first:

```bash
cmake -B build -DFAISS_ENABLE_C_API=ON -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF
cmake --build build -j
```

Then run tests for this Rust crate:

```bash
cd c_api/rust
cargo test
```

If `libfaiss_c` is not found automatically, set:

```bash
export FAISS_C_LIB_DIR=/path/to/build/c_api
```
