//! The judge: render each question against the shared state prefix, score
//! the option labels with a [`Scorer`], calibrate, and build typed answers.

use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::backend::{BackendError, Scorer};
use crate::prompt::{self, Template, MAX_OPTIONS};
use crate::protocol::{
    parse_questions, render_state, Answer, Evaluation, Question, Request, Usage,
};
use crate::score::{argmax, confidence, expected_level, softmax, Calibration};

#[derive(Debug, Clone)]
pub struct JudgeConfig {
    pub template: Template,
    pub calibration: Calibration,
    /// For `choice`: score this many cyclic rotations of the option order and
    /// average the per-key probabilities. 1 = no position-bias averaging.
    pub permutations: usize,
    pub debug: bool,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            template: Template::ChatMl,
            calibration: Calibration::default(),
            permutations: 1,
            debug: false,
        }
    }
}

pub struct Judge<S: Scorer> {
    pub scorer: S,
    pub cfg: JudgeConfig,
}

/// Raw, uncalibrated result for one question (used by `eval`/`calibrate`).
#[derive(Debug, Clone)]
pub struct RawQuestion {
    pub id: String,
    pub kind: &'static str,
    pub keys: Vec<String>,
    /// Averaged over permutations, still uncalibrated (log of mean prob).
    pub logprobs: Vec<f64>,
    pub prompt_evaluated: u64,
    pub prompt_cached: u64,
    pub latency_ms: f64,
}

impl<S: Scorer> Judge<S> {
    pub fn new(scorer: S, cfg: JudgeConfig) -> Self {
        Self { scorer, cfg }
    }

    /// Score every question and return uncalibrated per-option logprobs.
    /// All prompts of the request (every question, every rotation) go to the
    /// backend in one `score_many` call so it can share the state prefix.
    pub fn raw(&self, req: &Request) -> Result<Vec<RawQuestion>, BackendError> {
        let questions = parse_questions(&req.questions).map_err(BackendError::Rejected)?;
        let prefix = prompt::prefix(self.cfg.template, &render_state(&req.state));

        // Plan: one (question index, rotation order) per prompt.
        struct Plan {
            q: usize,
            order: Vec<usize>,
        }
        let mut plans = Vec::new();
        let mut prompts = Vec::new();
        let mut cands = Vec::new();
        let mut meta = Vec::with_capacity(questions.len()); // (kind, n, keys)
        for (qi, (id, q)) in questions.iter().enumerate() {
            let (kind, n) = match q {
                Question::Noul { .. } => ("noul", 2),
                Question::Choice { criteria, .. } => ("choice", criteria.len()),
                Question::Score { criteria, .. } => ("score", criteria.len()),
            };
            if n > MAX_OPTIONS {
                return Err(BackendError::Rejected(format!(
                    "question `{id}`: {n} options; jev-rs currently supports at most {MAX_OPTIONS}"
                )));
            }
            let perms = if kind == "choice" {
                self.cfg.permutations.clamp(1, n)
            } else {
                1
            };
            let mut keys = Vec::new();
            for r in 0..perms {
                let order: Vec<usize> = (0..n).map(|i| (i + r) % n).collect();
                let rendered = prompt::render(self.cfg.template, &prefix, q, Some(&order));
                if r == 0 {
                    // rotation 0 is the identity: keys are in request order.
                    keys = rendered.keys.clone();
                }
                prompts.push(rendered.prompt());
                cands.push(rendered.labels.clone());
                plans.push(Plan { q: qi, order });
            }
            meta.push((kind, n, keys, perms));
        }

        let t0 = Instant::now();
        let scored = self.scorer.score_many(&prompts, &cands)?;
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;
        if scored.len() != prompts.len() {
            return Err(BackendError::Malformed(format!(
                "backend returned {} results for {} prompts",
                scored.len(),
                prompts.len()
            )));
        }

        let mut out: Vec<RawQuestion> = questions
            .iter()
            .zip(&meta)
            .map(|((id, _), (kind, n, keys, _))| RawQuestion {
                id: id.clone(),
                kind,
                keys: keys.clone(),
                logprobs: vec![0.0; *n], // holds mean probability until the end
                prompt_evaluated: 0,
                prompt_cached: 0,
                latency_ms: 0.0,
            })
            .collect();
        for (plan, s) in plans.iter().zip(&scored) {
            let r = &mut out[plan.q];
            let perms = meta[plan.q].3 as f64;
            let p = softmax(&s.logprobs, 1.0);
            // rendered position `pos` holds original option `order[pos]`.
            for (pos, &orig) in plan.order.iter().enumerate() {
                r.logprobs[orig] += p[pos] / perms;
            }
            r.prompt_evaluated += s.cost.prompt_evaluated;
            r.prompt_cached += s.cost.prompt_cached;
            r.latency_ms += s.cost.latency_ms;
        }
        // A batched backend reports one latency for the whole group (the
        // rest are 0); spread the wall time evenly so per-question numbers
        // stay meaningful.
        if scored.iter().any(|s| s.cost.latency_ms == 0.0) {
            let each = total_ms / out.len().max(1) as f64;
            for r in &mut out {
                r.latency_ms = each;
            }
        }
        for r in &mut out {
            r.logprobs = r.logprobs.iter().map(|p| p.max(1e-300).ln()).collect();
        }
        Ok(out)
    }

    pub fn evaluate(&self, req: &Request) -> Result<Evaluation, BackendError> {
        let raws = self.raw(req)?;
        let mut answers = Map::new();
        let mut usage = Usage::default();
        let mut debug = Vec::new();
        for r in &raws {
            let t = self.cfg.calibration.temperature(r.kind, r.keys.len());
            let probs = softmax(&r.logprobs, t);
            let answer = match r.kind {
                "noul" => Answer::Noul {
                    noul: round(probs[0]),
                },
                "choice" => {
                    let best = argmax(&probs);
                    Answer::Choice {
                        choice: r.keys[best].clone(),
                        probabilities: prob_map(&r.keys, &probs),
                        confidence: round(confidence(&probs)),
                    }
                }
                _ => {
                    let legend: Map<String, Value> = match req.questions.get(&r.id) {
                        Some(Value::Object(o)) => o
                            .get("criteria")
                            .and_then(Value::as_array)
                            .map(|a| {
                                a.iter()
                                    .enumerate()
                                    .map(|(i, v)| (i.to_string(), v.clone()))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        _ => Map::new(),
                    };
                    Answer::Score {
                        score: round(expected_level(&probs)),
                        legend,
                        probabilities: prob_map(&r.keys, &probs),
                        confidence: round(confidence(&probs)),
                    }
                }
            };
            answers.insert(r.id.clone(), serde_json::to_value(answer).unwrap());
            // Only tokens the backend actually evaluated; cached prefix
            // tokens cost nothing and are reported under `debug`.
            usage.input_tokens += r.prompt_evaluated;
            if self.cfg.debug {
                debug.push(json!({
                    "id": r.id, "kind": r.kind, "temperature": t,
                    "raw_logprobs": r.logprobs,
                    "prompt_evaluated": r.prompt_evaluated,
                    "prompt_cached": r.prompt_cached,
                    "latency_ms": round(r.latency_ms),
                }));
            }
        }
        Ok(Evaluation {
            model: self.scorer.model_name(),
            answers,
            usage,
            debug: if self.cfg.debug {
                Some(Value::Array(debug))
            } else {
                None
            },
        })
    }
}

fn prob_map(keys: &[String], probs: &[f64]) -> Map<String, Value> {
    keys.iter()
        .zip(probs)
        .map(|(k, p)| (k.clone(), json!(round(*p))))
        .collect()
}

fn round(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
