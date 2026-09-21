//! Passthrough to the hosted TypeSafe API, so the same CLI/eval harness can
//! A/B a local backend against `jev-latest` with identical requests.

use std::time::Instant;

use serde_json::{json, Value};

use crate::protocol::{Evaluation, Request};

use super::BackendError;

pub const ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";

#[derive(Debug, Clone)]
pub struct TypeSafe {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
}

impl TypeSafe {
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("TYPESAFE_API_KEY").ok()?;
        let base =
            std::env::var("TYPESAFE_BASE_URL").unwrap_or_else(|_| "https://api.typesafe.ai".into());
        Some(Self {
            endpoint: format!("{}/v1/systemone", base.trim_end_matches('/')),
            api_key,
            model: std::env::var("TYPESAFE_DEFAULT_MODEL").unwrap_or_else(|_| "jev-latest".into()),
        })
    }

    pub fn evaluate(&self, req: &Request) -> Result<(Evaluation, f64), BackendError> {
        let body = json!({
            "model": req.model.clone().unwrap_or_else(|| self.model.clone()),
            "state": req.state,
            "questions": req.questions,
        });
        let t0 = Instant::now();
        let resp = ureq::post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(body);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let v: Value = match resp {
            Ok(r) => r
                .into_json()
                .map_err(|e| BackendError::Malformed(e.to_string()))?,
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                return Err(BackendError::Rejected(format!("{code}: {text}")));
            }
            Err(e) => return Err(BackendError::Http(e.to_string())),
        };
        let ev: Evaluation =
            serde_json::from_value(v).map_err(|e| BackendError::Malformed(e.to_string()))?;
        Ok((ev, ms))
    }
}
