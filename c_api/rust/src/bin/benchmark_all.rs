use faiss_rabitq::{HnswIndex, IvfRaBitQIndex, IvfSq8Index, MetricType};
use std::env;
use std::error::Error;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct BenchConfig {
    embeddings: usize,
    dimension: usize,
    queries: usize,
    k: usize,
    nlist: usize,
    hnsw_m: usize,
    metric: MetricType,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            embeddings: 100_000,
            dimension: 64,
            queries: 1_000,
            k: 10,
            nlist: 1_024,
            hnsw_m: 32,
            metric: MetricType::L2,
        }
    }
}

#[derive(Debug)]
struct BenchResult {
    algo: &'static str,
    build_ms: f64,
    search_ms: f64,
    qps: f64,
    valid_hits: usize,
}

fn synthetic_vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut values = Vec::with_capacity(n * d);
    let mut state = seed;
    for i in 0..n {
        for j in 0..d {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let random = ((state >> 33) as f32) / ((1_u32 << 31) as f32);
            values.push(random + i as f32 * 1e-3 + j as f32 * 1e-4);
        }
    }
    values
}

fn parse_metric(value: &str) -> Result<MetricType, String> {
    match value.to_ascii_lowercase().as_str() {
        "l2" => Ok(MetricType::L2),
        "ip" | "inner_product" => Ok(MetricType::InnerProduct),
        _ => Err(format!("unsupported metric '{value}'. Use 'l2' or 'ip'.")),
    }
}

fn parse_args() -> Result<BenchConfig, String> {
    let mut cfg = BenchConfig::default();
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;

    while i < args.len() {
        let flag = &args[i];
        let read_value = |idx: usize| -> Result<&str, String> {
            args.get(idx + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("missing value after {flag}"))
        };

        match flag.as_str() {
            "--embeddings" | "-n" => {
                cfg.embeddings = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid embeddings value: {e}"))?;
                i += 2;
            }
            "--dimension" | "-d" => {
                cfg.dimension = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid dimension value: {e}"))?;
                i += 2;
            }
            "--queries" | "-q" => {
                cfg.queries = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid queries value: {e}"))?;
                i += 2;
            }
            "--k" => {
                cfg.k = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid k value: {e}"))?;
                i += 2;
            }
            "--nlist" => {
                cfg.nlist = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid nlist value: {e}"))?;
                i += 2;
            }
            "--hnsw-m" => {
                cfg.hnsw_m = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid hnsw-m value: {e}"))?;
                i += 2;
            }
            "--metric" => {
                cfg.metric = parse_metric(read_value(i)?)?;
                i += 2;
            }
            "--help" | "-h" => {
                return Err(help_text());
            }
            unknown => {
                return Err(format!("unknown argument '{unknown}'\n\n{}", help_text()));
            }
        }
    }

    if cfg.embeddings == 0 {
        return Err("embeddings must be > 0".to_owned());
    }
    if cfg.dimension == 0 {
        return Err("dimension must be > 0".to_owned());
    }
    if cfg.queries == 0 {
        return Err("queries must be > 0".to_owned());
    }
    if cfg.k == 0 {
        return Err("k must be > 0".to_owned());
    }
    if cfg.nlist == 0 {
        return Err("nlist must be > 0".to_owned());
    }
    if cfg.hnsw_m == 0 {
        return Err("hnsw-m must be > 0".to_owned());
    }

    Ok(cfg)
}

fn help_text() -> String {
    [
        "Benchmark Faiss IVF-RaBitQ, IVF-SQ8, and HNSW from Rust",
        "",
        "Defaults:",
        "  --embeddings 100000 --dimension 64 --queries 1000 --k 10 --nlist 1024 --hnsw-m 32 --metric l2",
        "",
        "Flags:",
        "  -n, --embeddings <usize>",
        "  -d, --dimension <usize>",
        "  -q, --queries <usize>",
        "      --k <usize>",
        "      --nlist <usize>",
        "      --hnsw-m <usize>",
        "      --metric <l2|ip>",
        "  -h, --help",
    ]
    .join("\n")
}

fn valid_hits(labels: &[i64]) -> usize {
    labels.iter().filter(|&&id| id >= 0).count()
}

fn bench_ivf_rabitq(
    cfg: &BenchConfig,
    base: &[f32],
    train: &[f32],
    queries: &[f32],
) -> Result<BenchResult, Box<dyn Error>> {
    let build_start = Instant::now();
    let mut index = IvfRaBitQIndex::new(cfg.dimension, cfg.nlist, cfg.metric)?;
    index.train(train)?;
    index.add(base)?;
    index.set_nprobe((cfg.nlist / 16).max(1).min(64))?;
    let build_ms = build_start.elapsed().as_secs_f64() * 1_000.0;

    let search_start = Instant::now();
    let results = index.search(queries, cfg.k)?;
    let search_elapsed = search_start.elapsed();
    let search_ms = search_elapsed.as_secs_f64() * 1_000.0;
    let qps = cfg.queries as f64 / search_elapsed.as_secs_f64();

    Ok(BenchResult {
        algo: "IVF-RaBitQ",
        build_ms,
        search_ms,
        qps,
        valid_hits: valid_hits(&results.labels),
    })
}

fn bench_ivf_sq8(
    cfg: &BenchConfig,
    base: &[f32],
    train: &[f32],
    queries: &[f32],
) -> Result<BenchResult, Box<dyn Error>> {
    let build_start = Instant::now();
    let mut index = IvfSq8Index::new(cfg.dimension, cfg.nlist, cfg.metric)?;
    index.train(train)?;
    index.add(base)?;
    index.set_nprobe((cfg.nlist / 16).max(1).min(64))?;
    let build_ms = build_start.elapsed().as_secs_f64() * 1_000.0;

    let search_start = Instant::now();
    let results = index.search(queries, cfg.k)?;
    let search_elapsed = search_start.elapsed();
    let search_ms = search_elapsed.as_secs_f64() * 1_000.0;
    let qps = cfg.queries as f64 / search_elapsed.as_secs_f64();

    Ok(BenchResult {
        algo: "IVF-SQ8",
        build_ms,
        search_ms,
        qps,
        valid_hits: valid_hits(&results.labels),
    })
}

fn bench_hnsw(
    cfg: &BenchConfig,
    base: &[f32],
    queries: &[f32],
) -> Result<BenchResult, Box<dyn Error>> {
    let build_start = Instant::now();
    let mut index = HnswIndex::new(cfg.dimension, cfg.hnsw_m, cfg.metric)?;
    index.add(base)?;
    let build_ms = build_start.elapsed().as_secs_f64() * 1_000.0;

    let search_start = Instant::now();
    let results = index.search(queries, cfg.k)?;
    let search_elapsed = search_start.elapsed();
    let search_ms = search_elapsed.as_secs_f64() * 1_000.0;
    let qps = cfg.queries as f64 / search_elapsed.as_secs_f64();

    Ok(BenchResult {
        algo: "HNSW",
        build_ms,
        search_ms,
        qps,
        valid_hits: valid_hits(&results.labels),
    })
}

fn print_report(cfg: &BenchConfig, rows: &[BenchResult]) {
    println!(
        "Benchmark config: embeddings={} dim={} queries={} k={} nlist={} hnsw_m={} metric={:?}",
        cfg.embeddings, cfg.dimension, cfg.queries, cfg.k, cfg.nlist, cfg.hnsw_m, cfg.metric
    );
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12}",
        "algorithm", "build_ms", "search_ms", "qps", "valid_hits"
    );
    for row in rows {
        println!(
            "{:<12} {:>12.2} {:>12.2} {:>12.2} {:>12}",
            row.algo, row.build_ms, row.search_ms, row.qps, row.valid_hits
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = parse_args().map_err(|msg| -> Box<dyn Error> { msg.into() })?;

    let base = synthetic_vectors(cfg.embeddings, cfg.dimension, 0xdead_beef_u64);
    let train_size = cfg.embeddings.min(25_000);
    let train = &base[..train_size * cfg.dimension];
    let queries = synthetic_vectors(cfg.queries, cfg.dimension, 0xface_cafe_u64);

    let mut rows = Vec::with_capacity(3);
    rows.push(bench_ivf_rabitq(&cfg, &base, train, &queries)?);
    rows.push(bench_ivf_sq8(&cfg, &base, train, &queries)?);
    rows.push(bench_hnsw(&cfg, &base, &queries)?);

    print_report(&cfg, &rows);
    Ok(())
}
