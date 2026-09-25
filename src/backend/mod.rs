//! Backends. A [`Scorer`] answers one question: given a prompt, what are the
//! log-probabilities of each candidate next token? Everything else
//! (rendering, softmax, calibration, the wire format) is backend-independent.
//!
//! Implemented: [`llamacpp::LlamaServer`] (any GGUF via `llama-server`),
//! [`openai::OpenAiChat`], plus whole-request backends that already speak a
//! Jev-shaped protocol: [`systemone::SystemOne`] (Laya `laya-serve`, any
//! `/v1/systemone` host) and [`agentjev::AgentJev`].

pub mod agentjev;
pub mod llamacpp;
pub mod openai;
pub mod systemone;
pub mod typesafe;

use thiserror::Error;

use crate::protocol::Request;

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

impl Scorer for Box<dyn Scorer> {
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError> {
        (**self).score(prompt, candidates)
    }
    fn model_name(&self) -> String {
        (**self).model_name()
    }
}

/// A backend that answers a whole System One request in one call (Laya,
/// AgentJev, hosted Jev) rather than scoring next-token labels.
pub trait FullBackend: Send + Sync {
    /// Returns the Jev-shaped `answers` map and the wall latency in ms.
    fn evaluate(
        &self,
        req: &Request,
    ) -> Result<(serde_json::Map<String, Value>, f64), BackendError>;

    fn model_name(&self) -> String;
}

/// Re-export for FullBackend implementors.
pub use serde_json::Value;
