# Rust index examples via Faiss C API

This crate demonstrates calling Faiss from Rust and creating indexes with
factory strings:

- `IVF{nlist},RaBitQ`
- `IVF{nlist},SQ8`
- `HNSW{m}`

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
FAISS_C_LIB_PATH=/absolute/path/to/libfaiss_c.so cargo test
```

If `libfaiss_c` is not found automatically, set:

```bash
export FAISS_C_LIB_PATH=/path/to/build/c_api/libfaiss_c.so
```

## Benchmark binary (100,000 embeddings by default)

Run:

```bash
cd c_api/rust
FAISS_C_LIB_PATH=/absolute/path/to/libfaiss_c.so cargo run --release --bin benchmark_all
```

The benchmark runs all three algorithms (IVF-RaBitQ, IVF-SQ8, HNSW) with
default settings:

- `--embeddings 100000`
- `--dimension 64`
- `--queries 1000`
- `--k 10`
- `--nlist 1024`
- `--hnsw-m 32`
- `--metric l2`

Optional CLI flags:

```bash
cargo run --release --bin benchmark_all -- \
  --embeddings 100000 \
  --dimension 64 \
  --queries 1000 \
  --k 10 \
  --nlist 1024 \
  --hnsw-m 32 \
  --metric l2
```
