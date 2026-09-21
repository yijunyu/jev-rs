//! A tiny synchronous HTTP server exposing `POST /v1/systemone`.
//! Compatible with the TypeSafe SDKs via `TYPESAFE_BASE_URL`.

use std::sync::Arc;

use serde_json::{json, Value};
use tiny_http::{Header, Method, Request as HttpRequest, Response, Server};

use crate::backend::{BackendError, Scorer};
use crate::judge::Judge;
use crate::protocol::Request;

pub struct ServerConfig {
    pub bind: String,
    /// Accepted bearer tokens; empty = no auth.
    pub api_keys: Vec<String>,
}

pub fn serve<S: Scorer + 'static>(judge: Judge<S>, cfg: ServerConfig) -> Result<(), String> {
    let server = Server::http(&cfg.bind).map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    let judge = Arc::new(judge);
    let keys = Arc::new(cfg.api_keys);
    eprintln!(
        "jev-rs listening on http://{}  (model {})",
        cfg.bind,
        judge.scorer.model_name()
    );
    for req in server.incoming_requests() {
        let judge = judge.clone();
        let keys = keys.clone();
        std::thread::spawn(move || handle(req, &judge, &keys));
    }
    Ok(())
}

fn handle<S: Scorer>(mut req: HttpRequest, judge: &Judge<S>, keys: &[String]) {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    let result: (u16, Value) = match (&method, path.as_str()) {
        (Method::Get, "/health") => (200, json!({"status": "ok"})),
        (Method::Get, "/v1/models") => (
            200,
            json!({"object": "list", "data": [
                {"id": judge.scorer.model_name(), "object": "model", "owned_by": "jev-rs"},
                {"id": "jev-latest", "object": "model", "owned_by": "jev-rs", "alias_of": judge.scorer.model_name()}
            ]}),
        ),
        (Method::Post, "/v1/systemone") => {
            if !keys.is_empty() && !authorized(&req, keys) {
                (401, json!({"message": "invalid or missing API key"}))
            } else {
                let mut body = String::new();
                if req.as_reader().read_to_string(&mut body).is_err() {
                    (400, json!({"message": "unreadable body"}))
                } else {
                    match serde_json::from_str::<Request>(&body) {
                        Err(e) => (422, json!({"message": format!("invalid request: {e}")})),
                        Ok(r) => match judge.evaluate(&r) {
                            Ok(ev) => (200, serde_json::to_value(ev).unwrap()),
                            Err(BackendError::Rejected(m)) => (422, json!({"message": m})),
                            Err(e) => (502, json!({"message": e.to_string()})),
                        },
                    }
                }
            }
        }
        _ => (
            404,
            json!({"message": format!("no route {} {}", method, path)}),
        ),
    };
    let json_hdr = Header::from_bytes("Content-Type", "application/json").unwrap();
    let resp = Response::from_string(result.1.to_string())
        .with_status_code(result.0)
        .with_header(json_hdr);
    let _ = req.respond(resp);
}

fn authorized(req: &HttpRequest, keys: &[String]) -> bool {
    req.headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| {
            h.value
                .as_str()
                .trim_start_matches("Bearer ")
                .trim()
                .to_string()
        })
        .map(|k| keys.iter().any(|x| x == &k))
        .unwrap_or(false)
}
