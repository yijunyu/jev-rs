//! OpenAI-compatible `chat/completions` backend with `logprobs`. Works with
//! the DeepSeek API, vLLM, SGLang, llama-server's `/v1/chat/completions`
//! and any server that returns `top_logprobs` for the first generated
//! token. The server applies its own chat template, so the prompt is sent
//! as a system message (judge instructions), a user message (situation +
//! question) and the answer cue; the first assistant token is the label.

use std::time::Instant;

use serde_json::{json, Value};

use super::{BackendError, ScoreCost, Scored, Scorer};

#[derive(Debug, Clone)]
pub struct OpenAiChat {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    /// `top_logprobs` to request (DeepSeek/OpenAI cap this at 20).
    pub top_logprobs: usize,
    /// Extra body fields, e.g. `{"thinking": {"type": "disabled"}}`.
    pub extra: Value,
}

impl OpenAiChat {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key,
            top_logprobs: 20,
            extra: Value::Null,
        }
    }
}

/// Split a rendered prompt back into (system, user) text. The prompt module
/// renders with `Template::Raw` for this backend, so the layout is
/// `SYSTEM\n\nSituation:...\nAnswer:`.
fn split_prompt(prompt: &str) -> (&str, &str) {
    match prompt.split_once("\n\nSituation:\n") {
        Some((sys, rest)) => (sys, rest),
        None => ("", prompt),
    }
}

impl Scorer for OpenAiChat {
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError> {
        let (system, rest) = split_prompt(prompt);
        // Drop the trailing "\nAnswer:" cue: the assistant turn is the answer.
        let user = format!(
            "Situation:\n{}\n\nReply with the option letter only, nothing else.",
            rest.trim_end_matches("Answer:").trim_end()
        );
        let mut body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": true,
            "top_logprobs": self.top_logprobs.clamp(1, 20),
            "stream": false
        });
        if let (Value::Object(b), Value::Object(e)) = (&mut body, &self.extra) {
            for (k, v) in e {
                b.insert(k.clone(), v.clone());
            }
        }
        let t0 = Instant::now();
        let mut req = ureq::post(&format!("{}/chat/completions", self.base_url));
        if let Some(k) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        let v: Value = match req.send_json(body) {
            Ok(r) => r
                .into_json()
                .map_err(|e| BackendError::Malformed(e.to_string()))?,
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                return Err(BackendError::Http(format!(
                    "{code}: {}",
                    text.chars().take(300).collect::<String>()
                )));
            }
            Err(e) => return Err(BackendError::Http(e.to_string())),
        };
        let latency_ms = t0.elapsed().as_secs_f64() * 1e3;

        let content = v
            .pointer("/choices/0/logprobs/content")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                BackendError::Malformed(
                    "no choices[0].logprobs.content: the server did not return logprobs".into(),
                )
            })?;
        let first = content
            .first()
            .ok_or_else(|| BackendError::Malformed("empty logprobs.content".into()))?;
        let mut entries: Vec<(String, f64)> = first
            .get("top_logprobs")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|e| {
                        Some((
                            e.get("token")?.as_str()?.to_string(),
                            e.get("logprob")?.as_f64()?,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if let (Some(t), Some(lp)) = (
            first.get("token").and_then(Value::as_str),
            first.get("logprob").and_then(Value::as_f64),
        ) {
            entries.push((t.to_string(), lp));
        }
        // Candidates are " A"-style; chat models may answer "A", " A", "**A".
        let mut logprobs = vec![f64::NEG_INFINITY; candidates.len()];
        for (tok, lp) in entries {
            let norm = tok
                .trim()
                .trim_matches(|c: char| !c.is_ascii_alphanumeric());
            if let Some(i) = candidates.iter().position(|c| c.trim() == norm) {
                if lp > logprobs[i] {
                    logprobs[i] = lp;
                }
            }
        }
        let usage = v.get("usage").cloned().unwrap_or(Value::Null);
        let cost = ScoreCost {
            prompt_evaluated: usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                - usage
                    .pointer("/prompt_cache_hit_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            prompt_cached: usage
                .pointer("/prompt_cache_hit_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            latency_ms,
        };
        Ok(Scored { logprobs, cost })
    }

    fn model_name(&self) -> String {
        format!("openai/{}", self.model)
    }
}
