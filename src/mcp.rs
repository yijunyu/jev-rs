//! Minimal MCP (Model Context Protocol) server over stdio, so coding agents
//! (Claude Code, Codex, Grok Build, OpenCode, …) can call jev-rs as a tool.
//!
//! JSON-RPC 2.0, one message per line. Implements `initialize`, `ping`,
//! `tools/list` and `tools/call`; notifications are accepted and ignored.

use std::io::{self, BufRead, Write};

use serde_json::{json, Value};

use crate::backend::{BackendError, Scorer};
use crate::judge::Judge;
use crate::protocol::Request;

const PROTOCOL_VERSION: &str = "2025-06-18";

fn tool_schema() -> Value {
    json!({
        "name": "judge",
        "title": "Typed judgment (System One)",
        "description": "Ask typed questions about a piece of state and get calibrated probabilities \
    back, with no generated text. Question types: \
    noul {instructions} -> P(yes); \
    choice {instructions, criteria:{key:description|null}} -> chosen key + probabilities + confidence; \
    score {instructions, criteria:[level0,...]} -> probability-weighted level + probabilities. \
    Use it for routing, triage, gating, ranking and yes/no checks where you would otherwise \
    prompt-and-parse. Put facts in `state`, keep each question to one judgment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "description": "The situation: free text or a JSON object/array.",
                    "type": ["string", "object", "array"]
                },
                "questions": {
                    "type": "object",
                    "description": "Map of question id -> {type: noul|choice|score, instructions, criteria}.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {"enum": ["noul", "choice", "score"]},
                            "instructions": {},
                            "criteria": {}
                        },
                        "required": ["type", "instructions"]
                    }
                }
            },
            "required": ["state", "questions"]
        }
    })
}

pub fn serve<S: Scorer>(judge: &Judge<S>) -> io::Result<()> {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                write_msg(
                    &mut out,
                    &error(Value::Null, -32700, &format!("parse error: {e}")),
                )?;
                continue;
            }
        };
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let Some(id) = id else {
            continue; // notification
        };
        let reply = match method {
            "initialize" => result(
                id,
                json!({
                    "protocolVersion": params.get("protocolVersion").cloned().unwrap_or(json!(PROTOCOL_VERSION)),
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "jev-rs", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": "Call `judge` with a state and typed questions instead of asking an LLM to classify and parse."
                }),
            ),
            "ping" => result(id, json!({})),
            "tools/list" => result(id, json!({"tools": [tool_schema()]})),
            "tools/call" => call(judge, id, &params),
            _ => error(id, -32601, &format!("method not found: {method}")),
        };
        write_msg(&mut out, &reply)?;
    }
    Ok(())
}

fn call<S: Scorer>(judge: &Judge<S>, id: Value, params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    if name != "judge" {
        return error(id, -32602, &format!("unknown tool: {name}"));
    }
    let args = params.get("arguments").cloned().unwrap_or(Value::Null);
    let req: Request = match serde_json::from_value(args) {
        Ok(r) => r,
        Err(e) => return tool_error(id, &format!("invalid arguments: {e}")),
    };
    match judge.evaluate(&req) {
        Ok(ev) => {
            let text = serde_json::to_string_pretty(&ev.answers).unwrap_or_default();
            result(
                id,
                json!({
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": serde_json::to_value(&ev).unwrap_or(Value::Null),
                    "isError": false
                }),
            )
        }
        Err(BackendError::Rejected(m)) => tool_error(id, &m),
        Err(e) => tool_error(
            id,
            &format!("backend unavailable: {e}. Is llama-server running? See README."),
        ),
    }
}

fn tool_error(id: Value, msg: &str) -> Value {
    result(
        id,
        json!({"content": [{"type": "text", "text": msg}], "isError": true}),
    )
}

fn result(id: Value, r: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": r})
}

fn error(id: Value, code: i64, msg: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}})
}

fn write_msg(out: &mut impl Write, v: &Value) -> io::Result<()> {
    out.write_all(v.to_string().as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()
}
