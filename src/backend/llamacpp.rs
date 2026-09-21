//! `llama-server` backend: one `POST /completion` with `n_predict: 1` and
//! `n_probs: N`, reading the raw (pre-sampling) top-N log-probabilities of
//! the first generated position. `cache_prompt: true` makes every question
//! after the first on the same state cost one evaluated token.

use std::time::Instant;

use serde_json::{json, Value};

use super::{BackendError, ScoreCost, Scored, Scorer};

#[derive(Debug, Clone)]
pub struct LlamaServer {
    pub base_url: String,
    /// How many top tokens to request. Must comfortably exceed the number
    /// of options, since unrelated tokens compete for the slots.
    pub n_probs: usize,
    pub model_name: String,
}

impl LlamaServer {
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let model_name = probe_model(&base_url).unwrap_or_else(|| "llama-server".to_string());
        Self {
            base_url,
            n_probs: 40,
            model_name,
        }
    }
}

fn probe_model(base: &str) -> Option<String> {
    let v: Value = ureq::get(&format!("{base}/v1/models"))
        .call()
        .ok()?
        .into_json()
        .ok()?;
    let id = v.get("data")?.get(0)?.get("id")?.as_str()?;
    // llama-server reports the GGUF path; keep the file stem.
    let stem = std::path::Path::new(id).file_stem()?.to_str()?;
    Some(format!("llamacpp/{stem}"))
}

impl Scorer for LlamaServer {
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError> {
        let body = json!({
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": self.n_probs.max(candidates.len() + 8),
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "cache_prompt": true,
            "post_sampling_probs": false,
        });
        let t0 = Instant::now();
        let resp = ureq::post(&format!("{}/completion", self.base_url))
            .send_json(body)
            .map_err(|e| BackendError::Http(e.to_string()))?;
        let v: Value = resp
            .into_json()
            .map_err(|e| BackendError::Malformed(e.to_string()))?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1e3;

        let top = v
            .get("completion_probabilities")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("top_logprobs"))
            .and_then(Value::as_array)
            .ok_or_else(|| {
                BackendError::Malformed("no completion_probabilities[0].top_logprobs".into())
            })?;

        let mut logprobs = vec![f64::NEG_INFINITY; candidates.len()];
        for entry in top {
            let tok = entry.get("token").and_then(Value::as_str).unwrap_or("");
            let lp = entry
                .get("logprob")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NEG_INFINITY);
            if let Some(i) = candidates.iter().position(|c| c == tok) {
                if lp > logprobs[i] {
                    logprobs[i] = lp;
                }
            }
        }
        let timings = v.get("timings").cloned().unwrap_or(Value::Null);
        let cost = ScoreCost {
            prompt_evaluated: timings.get("prompt_n").and_then(Value::as_u64).unwrap_or(0),
            prompt_cached: v.get("tokens_cached").and_then(Value::as_u64).unwrap_or(0),
            latency_ms,
        };
        Ok(Scored { logprobs, cost })
    }

    fn model_name(&self) -> String {
        self.model_name.clone()
    }
}
