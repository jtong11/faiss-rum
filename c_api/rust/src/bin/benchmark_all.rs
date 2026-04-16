use faiss_rabitq::{HnswIndex, IvfRaBitQIndex, IvfSq8Index, MetricType};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
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
    rabitq_nbits: u8,
    hnsw_m: usize,
    hnsw_ef_build: usize,
    hnsw_ef_search: usize,
    metric: MetricType,
    with_recall: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            embeddings: 100_000,
            dimension: 64,
            queries: 1_000,
            k: 10,
            nlist: 1_024,
            rabitq_nbits: 8,
            hnsw_m: 32,
            hnsw_ef_build: 200,
            hnsw_ef_search: 128,
            metric: MetricType::L2,
            with_recall: false,
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
    recall_at_k: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct HeapCandidate {
    score: f32,
    id: usize,
}

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.id == other.id
    }
}

impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.id.cmp(&other.id))
    }
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
            "--nbits" | "--rabitq-nbits" => {
                cfg.rabitq_nbits = read_value(i)?
                    .parse::<u8>()
                    .map_err(|e| format!("invalid rabitq nbits value: {e}"))?;
                i += 2;
            }
            "--hnsw-m" => {
                cfg.hnsw_m = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid hnsw-m value: {e}"))?;
                i += 2;
            }
            "--ef-build" => {
                cfg.hnsw_ef_build = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid ef-build value: {e}"))?;
                i += 2;
            }
            "--ef-search" => {
                cfg.hnsw_ef_search = read_value(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid ef-search value: {e}"))?;
                i += 2;
            }
            "--metric" => {
                cfg.metric = parse_metric(read_value(i)?)?;
                i += 2;
            }
            "--with-recall" => {
                cfg.with_recall = true;
                i += 1;
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
    if cfg.k > cfg.embeddings {
        return Err("k must be <= embeddings".to_owned());
    }
    if cfg.nlist == 0 {
        return Err("nlist must be > 0".to_owned());
    }
    if cfg.rabitq_nbits > 8 {
        return Err("rabitq nbits must be in [0, 8]".to_owned());
    }
    if cfg.hnsw_m == 0 {
        return Err("hnsw-m must be > 0".to_owned());
    }
    if cfg.hnsw_ef_build == 0 {
        return Err("ef-build must be > 0".to_owned());
    }
    if cfg.hnsw_ef_search == 0 {
        return Err("ef-search must be > 0".to_owned());
    }

    Ok(cfg)
}

fn help_text() -> String {
    [
        "Benchmark Faiss IVF-RaBitQ, IVF-SQ8, and HNSW from Rust",
        "",
        "Defaults:",
        "  --embeddings 100000 --dimension 64 --queries 1000 --k 10 --nlist 1024 --nbits 8 --hnsw-m 32 --ef-build 200 --ef-search 128 --metric l2",
        "",
        "Flags:",
        "  -n, --embeddings <usize>",
        "  -d, --dimension <usize>",
        "  -q, --queries <usize>",
        "      --k <usize>",
        "      --nlist <usize>",
        "      --nbits, --rabitq-nbits <u8 in [0,8]>",
        "      --hnsw-m <usize>",
        "      --ef-build <usize>",
        "      --ef-search <usize>",
        "      --metric <l2|ip>",
        "      --with-recall",
        "      --with-recall",
        "  -h, --help",
    ]
    .join("\n")
}

fn valid_hits(labels: &[i64]) -> usize {
    labels.iter().filter(|&&id| id >= 0).count()
}

fn metric_distance(metric: MetricType, x: &[f32], y: &[f32]) -> f32 {
    match metric {
        MetricType::L2 => x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum(),
        MetricType::InnerProduct => -x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>(),
    }
}

fn exact_topk_labels(
    base: &[f32],
    queries: &[f32],
    dimension: usize,
    k: usize,
    metric: MetricType,
) -> Vec<i64> {
    let nb = base.len() / dimension;
    let nq = queries.len() / dimension;
    let mut all_labels = vec![-1_i64; nq * k];

    for q in 0..nq {
        let qv = &queries[q * dimension..(q + 1) * dimension];
        let mut scored: Vec<(f32, i64)> = (0..nb)
            .map(|i| {
                let bv = &base[i * dimension..(i + 1) * dimension];
                (metric_distance(metric, qv, bv), i as i64)
            })
            .collect();

        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (rank, (_, id)) in scored.into_iter().take(k).enumerate() {
            all_labels[q * k + rank] = id;
        }
    }

    all_labels
}

fn recall_at_k(predicted: &[i64], ground_truth: &[i64], k: usize) -> f64 {
    let nq = predicted.len() / k;
    let mut hits = 0_usize;
    for q in 0..nq {
        let pred = &predicted[q * k..(q + 1) * k];
        let gt = &ground_truth[q * k..(q + 1) * k];
        for &id in pred {
            if id >= 0 && gt.contains(&id) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}

fn transformed_score(metric: MetricType, query: &[f32], base: &[f32]) -> f32 {
    match metric {
        MetricType::L2 => query
            .iter()
            .zip(base.iter())
            .map(|(q, b)| {
                let diff = q - b;
                diff * diff
            })
            .sum(),
        MetricType::InnerProduct => -query
            .iter()
            .zip(base.iter())
            .map(|(q, b)| q * b)
            .sum::<f32>(),
    }
}

fn compute_exact_topk_labels(cfg: &BenchConfig, base: &[f32], queries: &[f32]) -> Vec<i64> {
    let k = cfg.k;
    let mut labels = Vec::with_capacity(cfg.queries * k);

    for q_idx in 0..cfg.queries {
        let q_start = q_idx * cfg.dimension;
        let q = &queries[q_start..q_start + cfg.dimension];

        let mut heap = BinaryHeap::with_capacity(k);
        for b_idx in 0..cfg.embeddings {
            let b_start = b_idx * cfg.dimension;
            let b = &base[b_start..b_start + cfg.dimension];
            let score = transformed_score(cfg.metric, q, b);
            let cand = HeapCandidate { score, id: b_idx };

            if heap.len() < k {
                heap.push(cand);
                continue;
            }

            if let Some(worst) = heap.peek()
                && cand.score < worst.score
            {
                heap.pop();
                heap.push(cand);
            }
        }

        let mut ordered = heap.into_vec();
        ordered.sort_by(|a, b| a.score.total_cmp(&b.score));
        labels.extend(ordered.iter().map(|cand| cand.id as i64));
    }

    labels
}

fn recall_at_k(predicted: &[i64], exact: &[i64], k: usize) -> f64 {
    if predicted.len() != exact.len() || k == 0 {
        return 0.0;
    }
    let nq = predicted.len() / k;
    let mut sum = 0.0;

    for q in 0..nq {
        let start = q * k;
        let end = start + k;
        let gt: HashSet<i64> = exact[start..end].iter().copied().collect();
        let hits = predicted[start..end]
            .iter()
            .filter(|&&id| id >= 0 && gt.contains(&id))
            .count();
        sum += hits as f64 / k as f64;
    }

    sum / nq as f64
}

fn bench_ivf_rabitq(
    cfg: &BenchConfig,
    base: &[f32],
    train: &[f32],
    queries: &[f32],
    exact_labels: Option<&[i64]>,
) -> Result<BenchResult, Box<dyn Error>> {
    let build_start = Instant::now();
    let mut index = IvfRaBitQIndex::new(cfg.dimension, cfg.nlist, cfg.metric)?;
    index.train(train)?;
    index.add(base)?;
    index.set_nprobe((cfg.nlist / 16).max(1).min(64))?;
    index.set_nbits(cfg.rabitq_nbits)?;
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
        recall_at_k: exact_labels.map(|gt| recall_at_k(&results.labels, gt, cfg.k)),
    })
}

fn bench_ivf_sq8(
    cfg: &BenchConfig,
    base: &[f32],
    train: &[f32],
    queries: &[f32],
    exact_labels: Option<&[i64]>,
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
        recall_at_k: exact_labels.map(|gt| recall_at_k(&results.labels, gt, cfg.k)),
    })
}

fn bench_hnsw(
    cfg: &BenchConfig,
    base: &[f32],
    queries: &[f32],
    exact_labels: Option<&[i64]>,
) -> Result<BenchResult, Box<dyn Error>> {
    let build_start = Instant::now();
    let mut index = HnswIndex::new(cfg.dimension, cfg.hnsw_m, cfg.metric)?;
    index.set_ef_build(cfg.hnsw_ef_build)?;
    index.set_ef_search(cfg.hnsw_ef_search)?;
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
        recall_at_k: exact_labels.map(|gt| recall_at_k(&results.labels, gt, cfg.k)),
    })
}

fn print_report(cfg: &BenchConfig, rows: &[BenchResult]) {
    println!(
        "Benchmark config: embeddings={} dim={} queries={} k={} nlist={} rabitq_nbits={} hnsw_m={} ef_build={} ef_search={} metric={:?}",
        cfg.embeddings,
        cfg.dimension,
        cfg.queries,
        cfg.k,
        cfg.nlist,
        cfg.rabitq_nbits,
        cfg.hnsw_m,
        cfg.hnsw_ef_build,
        cfg.hnsw_ef_search,
        cfg.metric,
    );
    if cfg.with_recall {
        println!(
            "{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "algorithm", "build_ms", "search_ms", "qps", "valid_hits", "recall@k"
        );
    } else {
        println!(
            "{:<12} {:>12} {:>12} {:>12} {:>12}",
            "algorithm", "build_ms", "search_ms", "qps", "valid_hits"
        );
    }
    for row in rows {
        if let Some(recall) = row.recall_at_k {
            println!(
                "{:<12} {:>12.2} {:>12.2} {:>12.2} {:>12} {:>12.4}",
                row.algo, row.build_ms, row.search_ms, row.qps, row.valid_hits, recall
            );
        } else {
            println!(
                "{:<12} {:>12.2} {:>12.2} {:>12.2} {:>12}",
                row.algo, row.build_ms, row.search_ms, row.qps, row.valid_hits
            );
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = parse_args().map_err(|msg| -> Box<dyn Error> { msg.into() })?;

    let base = synthetic_vectors(cfg.embeddings, cfg.dimension, 0xdead_beef_u64);
    let train_size = cfg.embeddings.min(25_000);
    let train = &base[..train_size * cfg.dimension];
    let queries = synthetic_vectors(cfg.queries, cfg.dimension, 0xface_cafe_u64);
    let exact_labels = if cfg.with_recall {
        let exact_start = Instant::now();
        let labels = compute_exact_topk_labels(&cfg, &base, &queries);
        println!(
            "Computed exact ground truth in {:.2} ms",
            exact_start.elapsed().as_secs_f64() * 1_000.0
        );
        Some(labels)
    } else {
        None
    };

    let mut rows = Vec::with_capacity(3);
    rows.push(bench_ivf_rabitq(
        &cfg,
        &base,
        train,
        &queries,
        exact_labels.as_deref(),
    )?);
    rows.push(bench_ivf_sq8(
        &cfg,
        &base,
        train,
        &queries,
        exact_labels.as_deref(),
    )?);
    rows.push(bench_hnsw(&cfg, &base, &queries, exact_labels.as_deref())?);

    print_report(&cfg, &rows);
    Ok(())
}
