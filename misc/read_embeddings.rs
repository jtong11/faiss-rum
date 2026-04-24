use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct EmbeddingRecord {
    embedding: Vec<f32>,
}

/// Reads a JSONL file and collects the `embedding` field from each line.
///
/// Each line must be a valid JSON object with an `embedding` array, e.g.:
/// `{"id":"a1","embedding":[0.12,0.34,0.56]}`
pub fn read_embeddings_from_jsonl<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut embeddings = Vec::new();

    for (line_idx, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        let record: EmbeddingRecord = serde_json::from_str(trimmed).map_err(|err| {
            format!("failed to parse JSON on line {}: {}", line_idx + 1, err)
        })?;
        embeddings.push(record.embedding);
    }

    Ok(embeddings)
}
