use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde_json::{json, Map, Value};

use jev_rs::backend::llamacpp::LlamaServer;
use jev_rs::backend::typesafe::TypeSafe;
use jev_rs::eval;
use jev_rs::judge::{Judge, JudgeConfig};
use jev_rs::prompt::Template;
use jev_rs::protocol::Request;
use jev_rs::score::Calibration;
use jev_rs::server::{serve, ServerConfig};

#[derive(Parser)]
#[command(
    name = "jev",
    version,
    about = "System One judgments from any LLM, in one prefill"
)]
struct Cli {
    /// llama-server base URL (any GGUF model).
    #[arg(long, env = "JEV_BACKEND_URL", default_value = "http://127.0.0.1:8080")]
    backend: String,
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
    let cfg = JudgeConfig {
        template,
        calibration,
        permutations: cli.permutations,
        debug: cli.debug,
    };
    let judge = Judge::new(LlamaServer::new(cli.backend.clone()), cfg);

    match cli.cmd {
        Cmd::Serve { bind, api_keys } => {
            let api_keys = api_keys
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            serve(judge, ServerConfig { bind, api_keys })
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
            let t0 = std::time::Instant::now();
            let ev = judge.evaluate(&req).map_err(|e| e.to_string())?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!("{}", serde_json::to_string_pretty(&ev).unwrap());
            eprintln!("local: {:.1} ms end-to-end", ms);
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
            let r = eval::run(&judge, &cases).map_err(|e| e.to_string())?;
            let m = eval::metrics(&r, &judge.cfg.calibration);
            println!("{}", serde_json::to_string_pretty(&m).unwrap());
            if let Some(p) = rows {
                let text: String = r
                    .iter()
                    .map(|x| serde_json::to_string(x).unwrap() + "\n")
                    .collect();
                std::fs::write(&p, text).map_err(|e| e.to_string())?;
            }
            Ok(())
        }
        Cmd::Mcp => jev_rs::mcp::serve(&judge).map_err(|e| e.to_string()),
        Cmd::Calibrate { file, out } => {
            let cases = eval::load_cases(&file)?;
            let r = eval::run(&judge, &cases).map_err(|e| e.to_string())?;
            let before = eval::metrics(&r, &Calibration::default());
            let cal = eval::fit(&r);
            let after = eval::metrics(&r, &cal);
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
