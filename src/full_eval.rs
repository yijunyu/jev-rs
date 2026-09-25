//! Eval path for whole-request backends (Laya, AgentJev): one HTTP call per
//! case answers every question; probabilities are converted to log-probs so
//! [`metrics`] / [`fit`] score them identically to a [`Scorer`] run.

use serde_json::Value;

use crate::backend::{BackendError, FullBackend};
use crate::eval::{self, Case, Row};
use crate::protocol::{parse_questions, Question, Request};
use crate::score::Calibration;

/// Run cases through a full-request backend. Latency recorded on each row is
/// the whole-case wall time (how long a Jev-style multi-question call took).
pub fn run(backend: &dyn FullBackend, cases: &[Case]) -> Result<(Vec<Row>, usize), BackendError> {
    let mut rows = Vec::new();
    let mut failed = 0usize;
    for (ci, c) in cases.iter().enumerate() {
        let req = Request {
            model: c.model.clone(),
            state: c.state.clone(),
            questions: c.questions.clone(),
        };
        let (answers, ms) = match backend.evaluate(&req) {
            Ok(x) => x,
            Err(BackendError::Rejected(m)) => return Err(BackendError::Rejected(m)),
            Err(e) => {
                eprintln!("case {ci}: {e} (counted as failed)");
                failed += 1;
                continue;
            }
        };
        let parsed = parse_questions(&req.questions).map_err(BackendError::Rejected)?;
        for (id, q) in &parsed {
            let Some(g) = c.gold.get(id) else { continue };
            let (kind, keys) = match q {
                Question::Noul { .. } => ("noul", vec!["yes".to_string(), "no".to_string()]),
                Question::Choice { criteria, .. } => {
                    ("choice", criteria.keys().cloned().collect::<Vec<_>>())
                }
                Question::Score { criteria, .. } => (
                    "score",
                    (0..criteria.len()).map(|i| i.to_string()).collect(),
                ),
            };
            let gold = match g {
                Value::Number(n) => n.as_u64().map(|x| x as usize),
                Value::String(s) => keys
                    .iter()
                    .position(|k| k == s)
                    .or_else(|| s.parse().ok())
                    .or(match s.as_str() {
                        "true" => Some(0),
                        "false" => Some(1),
                        _ => None,
                    }),
                Value::Bool(b) => Some(if *b { 0 } else { 1 }),
                _ => None,
            };
            let Some(gold) = gold else {
                return Err(BackendError::Rejected(format!(
                    "case {ci} question `{id}`: gold {g} is not an option"
                )));
            };
            let logprobs = answer_logprobs(kind, &keys, answers.get(id).unwrap_or(&Value::Null))?;
            rows.push(Row {
                case_index: ci,
                id: id.clone(),
                kind: kind.to_string(),
                n: keys.len(),
                raw_logprobs: logprobs,
                gold,
                latency_ms: ms,
                prompt_evaluated: 0,
                prompt_cached: 0,
            });
        }
    }
    Ok((rows, failed))
}

/// Convert a Jev answer object into ln(prob) over `keys` (same order the
/// Scorer path uses). Temperature stays 1 here; `metrics`/`fit` apply it.
fn answer_logprobs(kind: &str, keys: &[String], ans: &Value) -> Result<Vec<f64>, BackendError> {
    let bad = |m: &str| BackendError::Malformed(m.to_string());
    match kind {
        "noul" => {
            let p = ans
                .get("noul")
                .and_then(Value::as_f64)
                .ok_or_else(|| bad("noul answer missing `noul`"))?;
            let p = p.clamp(1e-12, 1.0 - 1e-12);
            Ok(vec![p.ln(), (1.0 - p).ln()])
        }
        "choice" => {
            let probs = ans
                .get("probabilities")
                .and_then(Value::as_object)
                .ok_or_else(|| bad("choice answer missing probabilities"))?;
            keys.iter()
                .map(|k| {
                    let p = probs
                        .get(k)
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        .clamp(1e-12, 1.0);
                    Ok(p.ln())
                })
                .collect()
        }
        "score" => {
            let probs = ans
                .get("probabilities")
                .and_then(Value::as_object)
                .ok_or_else(|| bad("score answer missing probabilities"))?;
            keys.iter()
                .map(|k| {
                    let p = probs
                        .get(k)
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        .clamp(1e-12, 1.0);
                    Ok(p.ln())
                })
                .collect()
        }
        other => Err(BackendError::Malformed(format!("unknown kind `{other}`"))),
    }
}

/// Convenience: metrics with the default (identity) calibration, then the
/// same after fitting temperatures on this run — both matter for comparing
/// against Scorer backends that report raw and calibrated views.
pub fn metrics_pair(rows: &[Row], failed: usize) -> (eval::Metrics, eval::Metrics, Calibration) {
    let raw = eval::metrics(rows, failed, &Calibration::default());
    let cal = eval::fit(rows);
    let fitted = eval::metrics(rows, failed, &cal);
    (raw, fitted, cal)
}
