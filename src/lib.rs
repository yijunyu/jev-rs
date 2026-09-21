//! jev-rs: System One judgments (`noul` / `choice` / `score`) from any LLM
//! in one prefill, served on a Jev-compatible `POST /v1/systemone`.
//!
//! Pipeline: [`prompt`] renders the state once and each question as a suffix
//! whose next token is an option label; a [`backend::Scorer`] returns the
//! label log-probabilities; [`score`] normalises, calibrates and derives the
//! typed answer; [`judge::Judge`] ties it together; [`server`] speaks HTTP.

pub mod backend;
pub mod eval;
pub mod judge;
pub mod mcp;
pub mod prompt;
pub mod protocol;
pub mod score;
pub mod server;
