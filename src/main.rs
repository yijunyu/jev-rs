use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde_json::{json, Map, Value};

use jev_rs::backend::agentjev::AgentJev;
use jev_rs::backend::llamacpp::LlamaServer;
use jev_rs::backend::openai::OpenAiChat;
use jev_rs::backend::systemone::SystemOne;
use jev_rs::backend::typesafe::TypeSafe;
use jev_rs::backend::{BackendError, FullBackend, Scorer};
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
    /// Chat template of the backend model: chatml | gemma | gemma4 | llama3 | raw
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
    /// Fitted stacker JSON (from `jev ensemble --out`): fan every request out
    /// to all --member backends in parallel and answer from the meta-model.
    #[arg(long = "ensemble", env = "JEV_ENSEMBLE", global = true)]
    stacker: Option<PathBuf>,
    /// Live ensemble member: `name=kind,url[,model]` (name/order must match
    /// the model's members; repeat once per member).
    #[arg(long = "member", action = clap::ArgAction::Append)]
    member: Vec<String>,
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
    /// Stack several backends' `eval --rows` files: fit a meta-calibrator on
    /// a case-level train split and report it against each member on test.
    Ensemble {
        /// Row files from `jev eval --rows`, one per member backend.
        #[arg(long = "rows", required = true)]
        rows: Vec<PathBuf>,
        /// Write the fitted stacker here (JSON).
        #[arg(long, default_value = "ensemble.json")]
        out: PathBuf,
        /// Apply this saved stacker instead of fitting.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Fraction of cases held out for testing (case-level split).
        #[arg(long, default_value_t = 0.3)]
        test: f64,
        /// Gradient-descent iterations for the meta-model.
        #[arg(long, default_value_t = 600)]
        iterations: usize,
        /// L2 regularization on the meta-model weights.
        #[arg(long, default_value_t = 1e-3)]
        lambda: f64,
    },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Build the single-backend engine from kind/url/model.
fn build_engine(
    cli: &Cli,
    kind: &str,
    url: &str,
    model: Option<&str>,
    template: Template,
    calibration: Calibration,
) -> Result<Engine, String> {
    match kind {
        "llamacpp" | "llama" => {
            let cfg = JudgeConfig {
                template,
                calibration,
                permutations: cli.permutations,
                debug: cli.debug,
            };
            Ok(Engine::Scored(Judge::new(
                Box::new(LlamaServer::new(url.to_string())) as Box<dyn Scorer>,
                cfg,
            )))
        }
        "openai" | "chat" => {
            let model = model
                .map(str::to_string)
                .or_else(|| cli.model.clone())
                .ok_or("--model is required with --backend-kind openai")?;
            let mut s =
                OpenAiChat::new(url.to_string(), model, std::env::var(&cli.api_key_env).ok());
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
            Ok(Engine::Scored(Judge::new(
                Box::new(s) as Box<dyn Scorer>,
                cfg,
            )))
        }
        "laya" | "systemone" => {
            let model = model
                .map(str::to_string)
                .or_else(|| cli.model.clone())
                .unwrap_or_else(|| "typed-decisions".into());
            let name = model.clone();
            Ok(Engine::Full(Box::new(SystemOne::new(
                url.to_string(),
                model,
                name,
            ))))
        }
        "agentjev" => Ok(Engine::Full(Box::new(AgentJev::new(url.to_string())))),
        other => Err(format!(
            "unknown backend kind `{other}` (llamacpp|openai|laya|systemone|agentjev)"
        )),
    }
}

/// Build one live ensemble member from a `name=kind,url[,model]` spec.
fn build_member(
    cli: &Cli,
    spec: &str,
    template: Template,
    calibration: Calibration,
) -> Result<Box<dyn jev_rs::ensemble::Member>, String> {
    let (name, rest) = spec
        .split_once('=')
        .ok_or_else(|| format!("--member `{spec}`: expected name=kind,url[,model]"))?;
    let mut parts = rest.splitn(3, ',');
    let kind = parts.next().unwrap_or("");
    let url = parts
        .next()
        .ok_or_else(|| format!("--member `{spec}`: missing url"))?;
    let model = parts.next();
    match build_engine(cli, kind, url, model, template, calibration)? {
        Engine::Scored(judge) => Ok(Box::new(jev_rs::ensemble::JudgeMember {
            name: name.to_string(),
            judge,
        })),
        Engine::Full(backend) => Ok(Box::new(jev_rs::ensemble::FullMember {
            name: name.to_string(),
            backend,
        })),
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
    let engine = match &cli.stacker {
        Some(path) => {
            let stacker = jev_rs::ensemble::Stacker::load(path)?;
            if cli.member.is_empty() {
                return Err("--ensemble requires one --member per member backend".into());
            }
            let members = cli
                .member
                .iter()
                .map(|spec| build_member(&cli, spec, template, calibration.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            Engine::Full(Box::new(jev_rs::ensemble::LiveStacker::new(
                stacker, members,
            )?))
        }
        None => build_engine(
            &cli,
            &cli.backend_kind,
            &cli.backend,
            cli.model.as_deref(),
            template,
            calibration,
        )?,
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
            Engine::Full(b) => jev_rs::mcp::serve_with(&move |req: &Request| {
                let t0 = std::time::Instant::now();
                let (answers, ms) = b.evaluate(req)?;
                Ok(json!({
                    "model": b.model_name(),
                    "answers": answers,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "latency_ms": ms.max(t0.elapsed().as_secs_f64() * 1e3),
                }))
            })
            .map_err(|e| e.to_string()),
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
        Cmd::Ensemble {
            rows,
            out,
            model,
            test,
            iterations,
            lambda,
        } => {
            let mut members = Vec::new();
            for p in &rows {
                let name = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.display().to_string());
                members.push((name, jev_rs::ensemble::load_rows(p)?));
            }
            match model {
                Some(mp) => {
                    let v = jev_rs::ensemble::apply_report(&mp, &members)?;
                    println!("{}", serde_json::to_string_pretty(&v).unwrap());
                }
                None => {
                    let (mut report, stacker) =
                        jev_rs::ensemble::fit_report(&members, test, iterations, lambda)?;
                    stacker.save(&out)?;
                    report.model = Some(out.display().to_string());
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                }
            }
            Ok(())
        }
    }
}

/// Minimal reverse proxy for whole-request backends on `jev serve`.
fn serve_full(b: Box<dyn FullBackend>, cfg: ServerConfig) -> Result<(), String> {
    use tiny_http::Server;
    let server = Server::http(&cfg.bind).map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    eprintln!(
        "jev-rs (full) listening on http://{}  (model {})",
        cfg.bind,
        b.model_name()
    );
    let b = std::sync::Arc::new(b);
    let keys = std::sync::Arc::new(cfg.api_keys);
    for req in server.incoming_requests() {
        // One thread per request: a live ensemble's fan-out would otherwise
        // serialize every client behind the slowest member.
        let b = b.clone();
        let keys = keys.clone();
        std::thread::spawn(move || handle_full(req, &**b, keys.as_slice()));
    }
    Ok(())
}

fn handle_full(mut req: tiny_http::Request, b: &dyn FullBackend, keys: &[String]) {
    use tiny_http::{Header, Method};
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    let result: (u16, String) = match (&method, path.as_str()) {
        (Method::Get, "/health") => (200, r#"{"status":"ok"}"#.into()),
        (Method::Get, "/v1/models") => (
            200,
            json!({"object":"list","data":[{"id": b.model_name(), "object":"model"}]}).to_string(),
        ),
        (Method::Post, "/v1/systemone") => {
            let authed = keys.is_empty()
                || req
                    .headers()
                    .iter()
                    .find(|h| h.field.equiv("Authorization"))
                    .and_then(|h| h.value.as_str().strip_prefix("Bearer ").map(String::from))
                    .map(|t| keys.contains(&t))
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
                            // Rejected is a client error (bad question, too many
                            // options) — 422, exactly like the scored server.
                            Err(BackendError::Rejected(m)) => {
                                (422, json!({"message": m}).to_string())
                            }
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
