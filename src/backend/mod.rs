//! Backends. A [`Scorer`] answers one question: given a prompt, what are the
//! log-probabilities of each candidate next token? Everything else
//! (rendering, softmax, calibration, the wire format) is backend-independent.
//!
//! Implemented: [`llamacpp::LlamaServer`] (any GGUF via `llama-server`).
//! Planned: an in-process ds4-rs-metal session (`AttnStepState` fork per
//! question), an in-process llama.cpp / ggml-tilers context (`llama_kv_self_seq_cp`).

pub mod llamacpp;
pub mod typesafe;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend http: {0}")]
    Http(String),
    #[error("backend returned malformed output: {0}")]
    Malformed(String),
    #[error("request rejected: {0}")]
    Rejected(String),
}

/// What one scoring call cost.
#[derive(Debug, Clone, Default)]
pub struct ScoreCost {
    /// Prompt tokens the backend actually evaluated (cache misses).
    pub prompt_evaluated: u64,
    /// Prompt tokens served from the backend's cache.
    pub prompt_cached: u64,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct Scored {
    /// Natural-log probability of each candidate, `-inf` when the backend did
    /// not report the candidate (it fell outside its top-N).
    pub logprobs: Vec<f64>,
    pub cost: ScoreCost,
}

pub trait Scorer: Send + Sync {
    /// Log-probabilities of `candidates` as the *next* token after `prompt`.
    /// Candidate strings are exact token texts (e.g. `" A"`).
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError>;

    /// Human-readable identity for the `model` field.
    fn model_name(&self) -> String;
}
