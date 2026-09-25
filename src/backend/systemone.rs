//! Passthrough to any Jev-compatible `POST /v1/systemone` host (Laya's
//! `laya-serve`, TypeSafe, jev-rs itself). The host owns the whole request;
//! this backend only re-labels probabilities into the shared log-prob
//! space so `eval` can score it with the same metrics as a Scorer.

use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::protocol::{parse_questions, Request};

use super::{BackendError, FullBackend};

#[derive(Debug, Clone)]
pub struct SystemOne {
    pub endpoint: String,
    pub api_key: Option<String>,
    pub model: String,
    pub name: String,
}

impl SystemOne {
    /// `base` is a host root (`http://127.0.0.1:8010`); the path
    /// `/v1/systemone` is appended. `model` is sent as the request's model
    /// when the request itself does not pin one.
    pub fn new(base: impl Into<String>, model: impl Into<String>, name: impl Into<String>) -> Self {
        let base = base.into().trim_end_matches('/').to_string();
        Self {
            endpoint: format!("{base}/v1/systemone"),
            api_key: std::env::var("JEV_API_KEY").ok(),
            model: model.into(),
            name: name.into(),
        }
    }

    fn post(&self, req: &Request) -> Result<(Value, f64), BackendError> {
        // Sanitise null option descriptions: some hosts reject JSON nulls.
        let questions = sanitise_questions(&req.questions)?;
        let body = json!({
            "model": req.model.clone().unwrap_or_else(|| self.model.clone()),
            "state": req.state,
            "questions": questions,
        });
        let t0 = Instant::now();
        let mut r = ureq::post(&self.endpoint);
        if let Some(k) = &self.api_key {
            r = r.set("Authorization", &format!("Bearer {k}"));
        }
        let resp = r.send_json(body);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let v: Value = match resp {
            Ok(resp) => resp
                .into_json()
                .map_err(|e| BackendError::Malformed(e.to_string()))?,
            Err(ureq::Error::Status(code, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                return Err(BackendError::Rejected(format!(
                    "{code}: {}",
                    text.chars().take(400).collect::<String>()
                )));
            }
            Err(e) => return Err(BackendError::Http(e.to_string())),
        };
        Ok((v, ms))
    }
}

impl FullBackend for SystemOne {
    fn evaluate(&self, req: &Request) -> Result<(Map<String, Value>, f64), BackendError> {
        let (v, ms) = self.post(req)?;
        let answers = v
            .get("answers")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| BackendError::Malformed("response has no answers object".into()))?;
        Ok((answers, ms))
    }

    fn model_name(&self) -> String {
        self.name.clone()
    }
}

/// Validate questions and replace `null` option descriptions with `""` so
/// strict hosts (AgentJev) accept the body; Laya accepts nulls but this is
/// harmless there too.
fn sanitise_questions(q: &Map<String, Value>) -> Result<Value, BackendError> {
    parse_questions(q).map_err(BackendError::Rejected)?;
    let mut out = q.clone();
    for (_qid, qv) in out.iter_mut() {
        if let Some(obj) = qv.get_mut("criteria").and_then(Value::as_object_mut) {
            for (_k, v) in obj.iter_mut() {
                if v.is_null() {
                    *v = Value::String(String::new());
                }
            }
        }
    }
    Ok(Value::Object(out))
}
