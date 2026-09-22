//! In-process llama.cpp backend (feature `llamacpp`): load any GGUF into this
//! process and score answer labels straight from the logits, with no server
//! in between. The shared state prefix stays in the KV cache between
//! questions: each call re-decodes only the tokens after the longest common
//! prefix with the previous call, exactly what a prefix cache gives a server
//! but without the HTTP hop or the JSON round trip of the logprobs.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use super::{BackendError, ScoreCost, Scored, Scorer};

struct State {
    ctx: LlamaContext<'static>,
    /// Tokens currently held in sequence 0 of the KV cache, in order.
    cached: Vec<LlamaToken>,
}

pub struct LlamaInProcess {
    // Field order is drop order: the context must go before the model it
    // borrows. `model` is boxed so its address is stable for the `'static`
    // borrow the context holds; see `load`.
    state: Mutex<State>,
    model: Box<LlamaModel>,
    _backend: LlamaBackend,
    name: String,
    n_ctx: usize,
    n_batch: usize,
}

// SAFETY: every use of the llama.cpp context goes through the mutex, so at
// most one thread touches it at a time; llama.cpp permits that. The model is
// read-only after load.
unsafe impl Send for LlamaInProcess {}
unsafe impl Sync for LlamaInProcess {}

impl LlamaInProcess {
    /// Load a GGUF. `n_gpu_layers` = 0 keeps the model on the CPU.
    pub fn load(path: &Path, n_ctx: u32, n_gpu_layers: u32) -> Result<Self, BackendError> {
        let backend = LlamaBackend::init().map_err(|e| BackendError::Http(e.to_string()))?;
        let mparams = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = Box::new(
            LlamaModel::load_from_file(&backend, path, &mparams)
                .map_err(|e| BackendError::Http(format!("load {}: {e}", path.display())))?,
        );
        let n_batch = 512u32;
        let cparams = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch);
        // SAFETY: `model` is boxed and lives as long as `self`; the context is
        // dropped first (field order). The lifetime is extended only to store
        // both in one struct.
        let model_ref: &'static LlamaModel = unsafe { &*(&*model as *const LlamaModel) };
        let ctx = model_ref
            .new_context(&backend, cparams)
            .map_err(|e| BackendError::Http(format!("context: {e}")))?;
        let name = PathBuf::from(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("gguf")
            .to_string();
        Ok(Self {
            state: Mutex::new(State {
                ctx,
                cached: Vec::new(),
            }),
            model,
            _backend: backend,
            name: format!("llamacpp-inproc/{name}"),
            n_ctx: n_ctx as usize,
            n_batch: n_batch as usize,
        })
    }

    fn first_token(&self, text: &str) -> Option<LlamaToken> {
        self.model
            .str_to_token(text, AddBos::Never)
            .ok()
            .and_then(|v| v.first().copied())
    }
}

impl Scorer for LlamaInProcess {
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError> {
        let t0 = Instant::now();
        let toks = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| BackendError::Malformed(format!("tokenize: {e}")))?;
        if toks.len() >= self.n_ctx {
            return Err(BackendError::Rejected(format!(
                "prompt is {} tokens; context is {}",
                toks.len(),
                self.n_ctx
            )));
        }
        let mut st = self
            .state
            .lock()
            .map_err(|_| BackendError::Http("poisoned".into()))?;

        // Longest common prefix with what sequence 0 already holds. Always
        // re-decode at least the last token so its logits are fresh.
        let mut k = st
            .cached
            .iter()
            .zip(&toks)
            .take_while(|(a, b)| a == b)
            .count();
        k = k.min(toks.len() - 1);
        if k < st.cached.len() {
            st.ctx
                .kv_cache_seq_rm(0, Some(k as u32), None)
                .map_err(|e| BackendError::Http(format!("kv rm: {e}")))?;
        }
        let mut batch = LlamaBatch::new(self.n_batch, 1);
        let mut last_idx = 0;
        let n = toks.len();
        let mut i = k;
        while i < n {
            batch.clear();
            let end = (i + self.n_batch).min(n);
            for (j, tok) in toks[i..end].iter().enumerate() {
                let pos = i + j;
                batch
                    .add(*tok, pos as i32, &[0], pos + 1 == n)
                    .map_err(|e| BackendError::Http(format!("batch: {e}")))?;
            }
            last_idx = batch.n_tokens() - 1;
            st.ctx
                .decode(&mut batch)
                .map_err(|e| BackendError::Http(format!("decode: {e}")))?;
            i = end;
        }
        st.cached = toks.clone();

        // Log-softmax over the full vocabulary, then read the label tokens.
        let logits = st.ctx.get_logits_ith(last_idx);
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = logits.iter().map(|&x| (x - m).exp()).sum::<f32>().ln() + m;
        let logprobs = candidates
            .iter()
            .map(|c| {
                let mut best = f64::NEG_INFINITY;
                for text in [c.as_str(), c.trim()] {
                    if let Some(t) = self.first_token(text) {
                        let id = t.0 as usize;
                        if id < logits.len() {
                            best = best.max((logits[id] - lse) as f64);
                        }
                    }
                }
                best
            })
            .collect();
        Ok(Scored {
            logprobs,
            cost: ScoreCost {
                prompt_evaluated: (n - k) as u64,
                prompt_cached: k as u64,
                latency_ms: t0.elapsed().as_secs_f64() * 1e3,
            },
        })
    }

    fn model_name(&self) -> String {
        self.name.clone()
    }
}
