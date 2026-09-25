use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde_json::{json, Map, Value};

use jev_rs::backend::agentjev::AgentJev;
use jev_rs::backend::llamacpp::LlamaServer;
use jev_rs::backend::openai::OpenAiChat;
use jev_rs::backend::systemone::SystemOne;
use jev_rs::backend::typesafe::TypeSafe;
use jev_rs::backend::{FullBackend, Scorer};
use jev_rs::eval;
use jev_rs::full_eval;
use jev_rs::judge::{Judge, JudgeConfig};
use jev_rs::prompt::Template;
use jev_rs::protocol::Request;
use jev_rs::score::Calibration;
use jev_rs::server::{serve, ServerConfig};

/// Which engine path the CLI drives: next-token Scorer or whole-request FullBackend.
enum Engine {
    Scored(Judge<Box<dyn Scorer>>),
    Full(Box<dyn FullBackend>),
}

#[derive(Parser)]
#[command(
    name = "jev",
    version,
    about = "System One judgments from any LLM, in one prefill"
)]
struct Cli {
    /// Backend base URL: a llama-server root, or an OpenAI-compatible `/v1`
    /// root (DeepSeek API, vLLM, SGLang) with --backend-kind openai.
    #[arg(long, env = "JEV_BACKEND_URL", default_value = "http://127.0.0.1:8080")]
    backend: String,
    /// llamacpp (raw next-token logprobs via /completion) | openai
    /// (chat/completions with logprobs) | laya/systemone (Jev wire host) |
    /// agentjev (AgentJev /api/evaluate adapter)
    #[arg(long, env = "JEV_BACKEND_KIND", default_value = "llamacpp")]
    backend_kind: String,
    /// Model id for --backend-kind openai (e.g. deepseek-chat), or the model
    /// field sent to a systemone host (default: typed-decisions for laya).
    #[arg(long, env = "JEV_MODEL")]
    model: Option<String>,
    /// Environment variable holding the API key for --backend-kind openai.
    #[arg(long, default_value = "JEV_API_KEY")]
    api_key_env: String,
    /// Extra JSON merged into openai requests, e.g. '{"thinking":{"type":"disabled"}}'.
    #[arg(long, env = "JEV_EXTRA")]
    extra: Option<String>,
    /// Chat template of the backend model: chatml | gemma | llama3 | raw
    #[arg(long, env = "JEV_TEMPLATE", default_value = "chatml")]
    template: String,
    /// JSON file with per-bucket temperatures (from `jev calibrate`).
    #[arg(long, env = "JEV_CALIBRATION")]
    calibration: Option<PathBuf>,
    /// Cyclic option rotations to average for `choice` (position-bias control).
    #[arg(long, default_value_t = 1)]
    permutations: usize,
    /// Include raw logprobs and per-question latency in responses.
    #[arg(long)]
    debug: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve POST /v1/systemone (TypeSafe-compatible).
    Serve {
        #[arg(long, default_value = "127.0.0.1:8090")]
        bind: String,
        /// Comma-separated bearer tokens; default none (also JEV_API_KEYS).
        #[arg(long, env = "JEV_API_KEYS", default_value = "")]
        api_keys: String,
    },
    /// Ask one request. Reads a JSON request from --file or stdin, or builds
    /// one from --state and --noul/--choice/--score flags.
    Ask {
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long)]
        state: Option<String>,
        /// id=instructions
        #[arg(long)]
        noul: Vec<String>,
        /// id=instructions|key1:desc,key2:desc
        #[arg(long)]
        choice: Vec<String>,
        /// id=instructions|level0,level1,level2
        #[arg(long)]
        score: Vec<String>,
        /// Send the same request to the hosted TypeSafe API too (needs TYPESAFE_API_KEY).
        #[arg(long)]
        compare: bool,
    },
    /// Run a JSONL case file and report accuracy / Brier / ECE / latency.
    Eval {
        file: PathBuf,
        /// Also write the raw rows here (for `calibrate`).
        #[arg(long)]
        rows: Option<PathBuf>,
    },
    /// Run as an MCP server over stdio (for Claude Code, Codex, Grok Build, OpenCode).
    Mcp,
    /// Fit per-bucket temperatures on a JSONL case file and write a calibration JSON.
    Calibrate {
        file: PathBuf,
        #[arg(long, default_value = "calibration.json")]
        out: PathBuf,
    },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let template: Template = cli.template.parse()?;
    let calibration = match &cli.calibration {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?)
            .map_err(|e| format!("calibration: {e}"))?,
        None => Calibration::default(),
    };
    let engine = match cli.backend_kind.as_str() {
        "llamacpp" | "llama" => {
            let cfg = JudgeConfig {
                template,
                calibration,
                permutations: cli.permutations,
                debug: cli.debug,
            };
            Engine::Scored(Judge::new(
                Box::new(LlamaServer::new(cli.backend.clone())) as Box<dyn Scorer>,
                cfg,
            ))
        }
        "openai" | "chat" => {
            let model = cli
                .model
                .clone()
                .ok_or("--model is required with --backend-kind openai")?;
            let mut s = OpenAiChat::new(
                cli.backend.clone(),
                model,
                std::env::var(&cli.api_key_env).ok(),
            );
            if let Some(x) = &cli.extra {
                s.extra = serde_json::from_str(x).map_err(|e| format!("--extra: {e}"))?;
            }
            let cfg = JudgeConfig {
                // The chat server applies its own template; render plain text.
                template: Template::Raw,
                calibration,
                permutations: cli.permutations,
                debug: cli.debug,
            };
            Engine::Scored(Judge::new(Box::new(s) as Box<dyn Scorer>, cfg))
        }
        "laya" | "systemone" => {
            let model = cli
                .model
                .clone()
                .unwrap_or_else(|| "typed-decisions".into());
            let name = cli.model.clone().unwrap_or_else(|| "systemone".into());
            Engine::Full(Box::new(SystemOne::new(cli.backend.clone(), model, name)))
        }
        "agentjev" => Engine::Full(Box::new(AgentJev::new(cli.backend.clone()))),
        other => {
            return Err(format!(
                "unknown --backend-kind `{other}` (llamacpp|openai|laya|systemone|agentjev)"
            ))
        }
    };

    match cli.cmd {
        Cmd::Serve { bind, api_keys } => {
            let api_keys: Vec<String> = api_keys
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            match engine {
                Engine::Scored(judge) => serve(judge, ServerConfig { bind, api_keys }),
                Engine::Full(b) => {
                    // Thin reverse proxy: accept /v1/systemone, forward whole.
                    serve_full(b, ServerConfig { bind, api_keys })
                }
            }
        }
        Cmd::Ask {
            file,
            state,
            noul,
            choice,
            score,
            compare,
        } => {
            let req = build_request(file, state, noul, choice, score)?;
            match engine {
                Engine::Scored(judge) => {
                    let t0 = std::time::Instant::now();
                    let ev = judge.evaluate(&req).map_err(|e| e.to_string())?;
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    println!("{}", serde_json::to_string_pretty(&ev).unwrap());
                    eprintln!("local: {:.1} ms end-to-end", ms);
                }
                Engine::Full(b) => {
                    let t0 = std::time::Instant::now();
                    let (answers, ms) = b.evaluate(&req).map_err(|e| e.to_string())?;
                    let ev = serde_json::json!({
                        "model": b.model_name(),
                        "answers": answers,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    });
                    println!("{}", serde_json::to_string_pretty(&ev).unwrap());
                    eprintln!(
                        "{}: {:.1} ms end-to-end",
                        b.model_name(),
                        ms.max(t0.elapsed().as_secs_f64() * 1e3)
                    );
                }
            }
            if compare {
                let ts = TypeSafe::from_env().ok_or("TYPESAFE_API_KEY not set")?;
                let (remote, rms) = ts.evaluate(&req).map_err(|e| e.to_string())?;
                println!("{}", serde_json::to_string_pretty(&remote).unwrap());
                eprintln!("typesafe: {:.1} ms end-to-end", rms);
            }
            Ok(())
        }
        Cmd::Eval { file, rows } => {
            let cases = eval::load_cases(&file)?;
            let (r, failed) = match &engine {
                Engine::Scored(judge) => eval::run(judge, &cases).map_err(|e| e.to_string())?,
                Engine::Full(b) => full_eval::run(&**b, &cases).map_err(|e| e.to_string())?,
            };
            let cal = match &engine {
                Engine::Scored(judge) => judge.cfg.calibration.clone(),
                Engine::Full(_) => Calibration::default(),
            };
            let m = eval::metrics(&r, failed, &cal);
            // For full backends also report post-fit metrics (same harness as calibrate).
            let extra = if matches!(engine, Engine::Full(_)) {
                let fitted_cal = eval::fit(&r);
                let fitted = eval::metrics(&r, failed, &fitted_cal);
                Some(
                    serde_json::json!({"fitted_calibration": fitted_cal, "metrics_fitted": fitted}),
                )
            } else {
                None
            };
            let mut out = serde_json::to_value(&m).unwrap();
            if let Some(e) = extra {
                if let (Some(obj), Some(val)) = (out.as_object_mut(), Some(e)) {
                    for (k, v) in val.as_object().unwrap() {
                        obj.insert(k.clone(), v.clone());
                    }
                }
            }
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
            if let Some(p) = rows {
                let text: String = r
                    .iter()
                    .map(|x| serde_json::to_string(x).unwrap() + "\n")
                    .collect();
                std::fs::write(&p, text).map_err(|e| e.to_string())?;
            }
            Ok(())
        }
        Cmd::Mcp => match engine {
            Engine::Scored(judge) => jev_rs::mcp::serve(&judge).map_err(|e| e.to_string()),
            Engine::Full(_) => Err("--backend-kind mcp only supports llamacpp|openai".into()),
        },
        Cmd::Calibrate { file, out } => {
            let cases = eval::load_cases(&file)?;
            let (r, failed) = match &engine {
                Engine::Scored(judge) => eval::run(judge, &cases).map_err(|e| e.to_string())?,
                Engine::Full(b) => full_eval::run(&**b, &cases).map_err(|e| e.to_string())?,
            };
            let before = eval::metrics(&r, failed, &Calibration::default());
            let cal = eval::fit(&r);
            let after = eval::metrics(&r, failed, &cal);
            std::fs::write(&out, serde_json::to_string_pretty(&cal).unwrap())
                .map_err(|e| e.to_string())?;
            println!(
                "{}",
                json!({"calibration": cal, "before": before, "after": after, "written": out})
            );
            Ok(())
        }
    }
}

/// Minimal reverse proxy for whole-request backends on `jev serve`.
fn serve_full(b: Box<dyn FullBackend>, cfg: ServerConfig) -> Result<(), String> {
    use tiny_http::{Header, Method, Server};
    let server = Server::http(&cfg.bind).map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    eprintln!(
        "jev-rs (full) listening on http://{}  (model {})",
        cfg.bind,
        b.model_name()
    );
    for mut req in server.incoming_requests() {
        let path = req.url().split('?').next().unwrap_or("").to_string();
        let method = req.method().clone();
        let result: (u16, String) = match (&method, path.as_str()) {
            (Method::Get, "/health") => (200, r#"{"status":"ok"}"#.into()),
            (Method::Get, "/v1/models") => (
                200,
                json!({"object":"list","data":[{"id": b.model_name(), "object":"model"}]})
                    .to_string(),
            ),
            (Method::Post, "/v1/systemone") => {
                let authed = cfg.api_keys.is_empty()
                    || req
                        .headers()
                        .iter()
                        .find(|h| h.field.equiv("Authorization"))
                        .and_then(|h| h.value.as_str().strip_prefix("Bearer ").map(String::from))
                        .map(|t| cfg.api_keys.contains(&t))
                        .unwrap_or(false);
                if !authed {
                    (
                        401,
                        json!({"message":"invalid or missing API key"}).to_string(),
                    )
                } else {
                    let mut body = String::new();
                    if req.as_reader().read_to_string(&mut body).is_err() {
                        (400, json!({"message":"unreadable body"}).to_string())
                    } else {
                        match serde_json::from_str::<Request>(&body) {
                            Err(e) => (
                                422,
                                json!({"message": format!("invalid request: {e}")}).to_string(),
                            ),
                            Ok(r) => match b.evaluate(&r) {
                                Ok((answers, ms)) => (
                                    200,
                                    json!({
                                        "model": b.model_name(),
                                        "answers": answers,
                                        "usage": {"input_tokens":0,"output_tokens":0},
                                        "latency_ms": ms,
                                    })
                                    .to_string(),
                                ),
                                Err(e) => (502, json!({"message": e.to_string()}).to_string()),
                            },
                        }
                    }
                }
            }
            _ => (
                404,
                json!({"message": format!("no route {} {}", method, path)}).to_string(),
            ),
        };
        let json_hdr = Header::from_bytes("Content-Type", "application/json").unwrap();
        let resp = tiny_http::Response::from_string(result.1)
            .with_status_code(result.0)
            .with_header(json_hdr);
        let _ = req.respond(resp);
    }
    Ok(())
}

fn build_request(
    file: Option<PathBuf>,
    state: Option<String>,
    noul: Vec<String>,
    choice: Vec<String>,
    score: Vec<String>,
) -> Result<Request, String> {
    if let Some(p) = file {
        let text = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
        return serde_json::from_str(&text).map_err(|e| e.to_string());
    }
    if state.is_none() && noul.is_empty() && choice.is_empty() && score.is_empty() {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)
            .map_err(|e| e.to_string())?;
        return serde_json::from_str(&text).map_err(|e| e.to_string());
    }
    let state = state.ok_or("--state is required with --noul/--choice/--score")?;
    let mut questions = Map::new();
    for s in noul {
        let (id, instr) = split_once(&s, '=')?;
        questions.insert(id.into(), json!({"type": "noul", "instructions": instr}));
    }
    for s in choice {
        let (id, rest) = split_once(&s, '=')?;
        let (instr, opts) = split_once(rest, '|')?;
        let mut criteria = Map::new();
        for o in opts.split(',') {
            let (k, d) = o.split_once(':').unwrap_or((o, ""));
            criteria.insert(
                k.trim().into(),
                if d.trim().is_empty() {
                    Value::Null
                } else {
                    json!(d.trim())
                },
            );
        }
        questions.insert(
            id.into(),
            json!({"type": "choice", "instructions": instr, "criteria": criteria}),
        );
    }
    for s in score {
        let (id, rest) = split_once(&s, '=')?;
        let (instr, levels) = split_once(rest, '|')?;
        let levels: Vec<&str> = levels.split(',').map(str::trim).collect();
        questions.insert(
            id.into(),
            json!({"type": "score", "instructions": instr, "criteria": levels}),
        );
    }
    Ok(Request {
        model: None,
        state: Value::String(state),
        questions,
    })
}

fn split_once(s: &str, c: char) -> Result<(&str, &str), String> {
    s.split_once(c)
        .map(|(a, b)| (a.trim(), b.trim()))
        .ok_or_else(|| format!("expected `{c}` in `{s}`"))
}
