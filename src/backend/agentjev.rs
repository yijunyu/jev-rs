//! AgentJev HTTP adapter: maps a Jev-shaped `Request` onto AgentJev's
//! `POST /api/evaluate` (`boolean`/`choice`/`score`, question *array*) and
//! the response back onto Jev `answers`.

use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::protocol::{parse_questions, render_state, Question, Request};

use super::{BackendError, FullBackend};

#[derive(Debug, Clone)]
pub struct AgentJev {
    pub base_url: String,
    pub model: String,
}

impl AgentJev {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: "agentjev-0.6b".into(),
        }
    }

    fn post(&self, req: &Request) -> Result<(Value, f64), BackendError> {
        let questions = parse_questions(&req.questions).map_err(BackendError::Rejected)?;
        let state = render_state(&req.state);
        let mut arr = Vec::with_capacity(questions.len());
        for (id, q) in &questions {
            let o = match q {
                Question::Noul {
                    instructions,
                    criteria,
                } => {
                    let (t, f) = match criteria {
                        Some(c) => (
                            crate::protocol::text_of(&c.is_true),
                            crate::protocol::text_of(&c.is_false),
                        ),
                        None => ("true".into(), "false".into()),
                    };
                    // Prefer non-empty criteria; AgentJev wants distinct texts.
                    let t = if t.trim().is_empty() {
                        "the statement holds".into()
                    } else {
                        t
                    };
                    let f = if f.trim().is_empty() {
                        "the statement does not hold".into()
                    } else {
                        f
                    };
                    json!({
                        "id": id,
                        "type": "boolean",
                        "question": crate::protocol::text_of(instructions),
                        "criteria": {"true": t, "false": f},
                    })
                }
                Question::Choice {
                    instructions,
                    criteria,
                } => {
                    let mut options = Map::new();
                    for (k, v) in criteria {
                        // AgentJev rejects null: use the option key as text.
                        let d = crate::protocol::text_of(v);
                        let d = if d.trim().is_empty() { k.clone() } else { d };
                        options.insert(k.clone(), Value::String(d));
                    }
                    json!({
                        "id": id,
                        "type": "choice",
                        "question": crate::protocol::text_of(instructions),
                        "options": options,
                    })
                }
                Question::Score {
                    instructions,
                    criteria,
                } => {
                    let levels: Vec<Value> = criteria
                        .iter()
                        .map(|v| {
                            let t = crate::protocol::text_of(v);
                            Value::String(if t.trim().is_empty() {
                                "level".into()
                            } else {
                                t
                            })
                        })
                        .collect();
                    json!({
                        "id": id,
                        "type": "score",
                        "question": crate::protocol::text_of(instructions),
                        "levels": levels,
                    })
                }
            };
            arr.push(o);
        }
        let body = json!({ "state": state, "questions": arr });
        let t0 = Instant::now();
        let resp = ureq::post(&format!("{}/api/evaluate", self.base_url))
            .send_json(body)
            .map_err(|e| match e {
                ureq::Error::Status(code, r) => {
                    let text = r.into_string().unwrap_or_default();
                    BackendError::Rejected(format!(
                        "{code}: {}",
                        text.chars().take(400).collect::<String>()
                    ))
                }
                other => BackendError::Http(other.to_string()),
            })?;
        let v: Value = resp
            .into_json()
            .map_err(|e| BackendError::Malformed(e.to_string()))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let answers = map_answers(&v)?;
        let usage = v.get("usage").cloned().unwrap_or(Value::Null);
        let out = json!({
            "model": self.model,
            "answers": answers,
            "usage": {
                "input_tokens": usage.get("input_path_tokens").and_then(Value::as_u64).unwrap_or(0),
                "output_tokens": usage.get("generated_tokens").and_then(Value::as_u64).unwrap_or(0),
            },
            "agentjev": usage,
        });
        Ok((out, ms))
    }
}

impl FullBackend for AgentJev {
    fn evaluate(&self, req: &Request) -> Result<(Map<String, Value>, f64), BackendError> {
        let (v, ms) = self.post(req)?;
        let answers = v
            .get("answers")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| BackendError::Malformed("adapter produced no answers".into()))?;
        Ok((answers, ms))
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }
}

fn map_answers(v: &Value) -> Result<Map<String, Value>, BackendError> {
    let list = v
        .pointer("/results/0/answers")
        .and_then(Value::as_array)
        .ok_or_else(|| BackendError::Malformed("no results[0].answers".into()))?;
    let mut out = Map::new();
    for a in list {
        let id = a
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| BackendError::Malformed("answer missing id".into()))?
            .to_string();
        let ty = a.get("type").and_then(Value::as_str).unwrap_or("");
        let dist = a.get("distribution").cloned().unwrap_or(Value::Null);
        let mapped = match ty {
            "boolean" | "noul" => {
                let p_true = dist
                    .get("true")
                    .and_then(Value::as_f64)
                    .or_else(|| a.get("probability").and_then(Value::as_f64))
                    .unwrap_or(0.5);
                // AgentJev `probability` is P(value); distribution has both.
                let p = if dist.get("true").is_some() {
                    p_true
                } else {
                    // value boolean + probability of that value
                    let val = a.get("value").and_then(Value::as_bool).unwrap_or(false);
                    let p = a.get("probability").and_then(Value::as_f64).unwrap_or(0.5);
                    if val {
                        p
                    } else {
                        1.0 - p
                    }
                };
                json!({"type": "noul", "noul": round(p)})
            }
            "choice" => {
                let probs = dist.as_object().cloned().unwrap_or_default();
                let best = a
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let pmax = a
                    .get("top_probability")
                    .and_then(Value::as_f64)
                    .or_else(|| probs.get(&best).and_then(Value::as_f64))
                    .unwrap_or(0.0);
                // confidence: TypeSafe formula needs full dist; use top prob
                // rescaled later by caller — expose raw top as confidence here.
                let mut probabilities = Map::new();
                for (k, p) in &probs {
                    probabilities.insert(k.clone(), p.clone());
                }
                if !probabilities.contains_key(&best) && pmax > 0.0 {
                    probabilities.insert(best.clone(), json!(pmax));
                }
                json!({
                    "type": "choice",
                    "choice": best,
                    "probabilities": probabilities,
                    "confidence": round(pmax),
                })
            }
            "score" => {
                let probs = dist.as_object().cloned().unwrap_or_default();
                let score = a.get("score").and_then(Value::as_f64).unwrap_or(0.0);
                let legend: Map<String, Value> = a
                    .get("legend")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .enumerate()
                            .map(|(i, v)| (i.to_string(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                json!({
                    "type": "score",
                    "score": round(score),
                    "legend": legend,
                    "probabilities": probs,
                    "confidence": round(confidence_of(probs.values())),
                })
            }
            other => {
                return Err(BackendError::Malformed(format!(
                    "unknown AgentJev answer type `{other}`"
                )))
            }
        };
        out.insert(id, mapped);
    }
    Ok(out)
}

fn confidence_of<'a, I: Iterator<Item = &'a Value>>(probs: I) -> f64 {
    let p: Vec<f64> = probs.filter_map(Value::as_f64).collect();
    let n = p.len();
    if n == 0 {
        return 0.0;
    }
    let pmax = p.iter().cloned().fold(0.0f64, f64::max);
    ((n as f64 * pmax - 1.0) / (n as f64 - 1.0)).clamp(0.0, 1.0)
}

fn round(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
