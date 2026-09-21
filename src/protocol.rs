//! Wire types for the System One request/response, byte-compatible with
//! TypeSafe's `POST /v1/systemone` so the official SDKs (`typesafe-sdk`,
//! `@typesafe-ai/sdk`, the `jev` crate) work unchanged with
//! `TYPESAFE_BASE_URL` pointed at a jev-rs server.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A request: the situation (`state`) plus named typed questions.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Request {
    /// Model alias. Accepted and echoed; jev-rs serves whatever backend it was
    /// started with.
    #[serde(default)]
    pub model: Option<String>,
    /// Free text, or any JSON object/array (rendered as pretty JSON).
    pub state: Value,
    /// Question id -> question.
    pub questions: Map<String, Value>,
}

/// One typed judgment requested from the model.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no. `noul` is TypeSafe's name for the primitive.
    Noul {
        instructions: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// One of N named options. Values are optional descriptions.
    Choice {
        instructions: Value,
        criteria: Map<String, Value>,
    },
    /// An ordered scale; `criteria[i]` describes level `i`.
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    pub is_true: Value,
    #[serde(rename = "false")]
    pub is_false: Value,
}

/// One typed answer.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        /// P(yes) in [0, 1].
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: Map<String, Value>,
        confidence: f64,
    },
    Score {
        /// Probability-weighted mean level index.
        score: f64,
        legend: Map<String, Value>,
        probabilities: Map<String, Value>,
        confidence: f64,
    },
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// The response envelope.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Evaluation {
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: Usage,
    /// jev-rs extension: raw per-question diagnostics (logprobs, latency).
    /// Absent unless the server was started with `--debug`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<Value>,
}

impl Evaluation {
    pub fn answer(&self, id: &str) -> Option<Answer> {
        self.answers
            .get(id)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

/// Render an `instructions`/`criteria` value as prompt text.
pub fn text_of(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Render `state` as prompt text: strings verbatim, structures as pretty JSON.
pub fn render_state(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

pub fn parse_questions(raw: &Map<String, Value>) -> Result<Vec<(String, Question)>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for (id, v) in raw {
        let q: Question = serde_json::from_value(v.clone())
            .map_err(|e| format!("question `{id}`: {e}"))?;
        match &q {
            Question::Choice { criteria, .. } if criteria.is_empty() => {
                return Err(format!("question `{id}`: choice needs at least one option"))
            }
            Question::Score { criteria, .. } if criteria.len() < 2 || criteria.len() > 10 => {
                return Err(format!("question `{id}`: score needs 2 to 10 levels"))
            }
            _ => {}
        }
        out.push((id.clone(), q));
    }
    if out.is_empty() {
        return Err("no questions".into());
    }
    Ok(out)
}
