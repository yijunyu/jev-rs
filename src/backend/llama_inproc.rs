//! In-process llama.cpp backend (feature `llamacpp`): load any GGUF into this
//! process and score answer labels straight from the logits, with no server
//! in between.
//!
//! Two paths share one context:
//! - [`Scorer::score`]: one prompt; re-decodes only the tokens after the
//!   longest common prefix with the previous call (a prefix cache without a
//!   server).
//! - [`Scorer::score_many`]: the prompts of one request. The common token
//!   prefix (the state) is decoded once into sequence 0, forked to one KV
//!   sequence per prompt with `kv_cache_seq_cp` (a bookkeeping copy in the
//!   unified cache, not a data copy), and every question tail is decoded in
//!   one batched step. A four-question request costs one prefill plus one
//!   decode call.

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

/// KV sequences the context is created with: sequence 0 holds the shared
/// prefix, 1..MAX_SEQ hold question tails. Larger requests are chunked.
const MAX_SEQ: usize = 16;

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

fn err(what: &str, e: impl std::fmt::Display) -> BackendError {
    BackendError::Http(format!("{what}: {e}"))
}

impl LlamaInProcess {
    /// Load a GGUF. `n_gpu_layers` = 0 keeps the model on the CPU.
    pub fn load(path: &Path, n_ctx: u32, n_gpu_layers: u32) -> Result<Self, BackendError> {
        let backend = LlamaBackend::init().map_err(|e| err("backend", e))?;
        let mparams = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = Box::new(
            LlamaModel::load_from_file(&backend, path, &mparams)
                .map_err(|e| err(&format!("load {}", path.display()), e))?,
        );
        let n_batch = 1024u32;
        let cparams = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch)
            .with_n_seq_max(MAX_SEQ as u32)
            .with_kv_unified(true);
        // SAFETY: `model` is boxed and lives as long as `self`; the context is
        // dropped first (field order). The lifetime is extended only to store
        // both in one struct.
        let model_ref: &'static LlamaModel = unsafe { &*(&*model as *const LlamaModel) };
        let ctx = model_ref
            .new_context(&backend, cparams)
            .map_err(|e| err("context", e))?;
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

    fn tokenize(&self, prompt: &str) -> Result<Vec<LlamaToken>, BackendError> {
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
        Ok(toks)
    }

    fn first_token(&self, text: &str) -> Option<LlamaToken> {
        self.model
            .str_to_token(text, AddBos::Never)
            .ok()
            .and_then(|v| v.first().copied())
    }

    /// Log-probabilities of the candidate label tokens from one logits row.
    fn label_logprobs(&self, logits: &[f32], candidates: &[String]) -> Vec<f64> {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = logits.iter().map(|&x| (x - m).exp()).sum::<f32>().ln() + m;
        candidates
            .iter()
            .map(|c| {
                let mut best = f64::NEG_INFINITY;
                // A label renders as " A"; some templates make the model
                // emit "A" without the space. Keep the larger.
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
            .collect()
    }

    /// Make sequence 0 hold exactly `toks[..len]`, reusing what it already
    /// has. Returns how many tokens were already cached.
    fn ensure_prefix(
        &self,
        st: &mut State,
        toks: &[LlamaToken],
        len: usize,
    ) -> Result<usize, BackendError> {
        let k = st
            .cached
            .iter()
            .zip(&toks[..len])
            .take_while(|(a, b)| a == b)
            .count();
        if k < st.cached.len() {
            st.ctx
                .kv_cache_seq_rm(0, Some(k as u32), None)
                .map_err(|e| err("kv rm", e))?;
            st.cached.truncate(k);
        }
        let mut i = k;
        let mut batch = LlamaBatch::new(self.n_batch, 1);
        while i < len {
            batch.clear();
            let end = (i + self.n_batch).min(len);
            for (j, tok) in toks[i..end].iter().enumerate() {
                batch
                    .add(*tok, (i + j) as i32, &[0], false)
                    .map_err(|e| err("batch", e))?;
            }
            st.ctx.decode(&mut batch).map_err(|e| err("decode", e))?;
            i = end;
        }
        st.cached = toks[..len].to_vec();
        Ok(k)
    }
}

impl Scorer for LlamaInProcess {
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError> {
        let t0 = Instant::now();
        let toks = self.tokenize(prompt)?;
        let n = toks.len();
        let mut st = self.state.lock().map_err(|_| err("lock", "poisoned"))?;
        // Everything but the last token can live in the cache; the last one
        // is decoded here so its logits are fresh.
        let k = self.ensure_prefix(&mut st, &toks, n - 1)?;
        let mut batch = LlamaBatch::new(self.n_batch, 1);
        batch
            .add(toks[n - 1], (n - 1) as i32, &[0], true)
            .map_err(|e| err("batch", e))?;
        st.ctx.decode(&mut batch).map_err(|e| err("decode", e))?;
        st.cached = toks.clone();
        let logprobs = self.label_logprobs(st.ctx.get_logits_ith(0), candidates);
        Ok(Scored {
            logprobs,
            cost: ScoreCost {
                prompt_evaluated: (n - k) as u64,
                prompt_cached: k as u64,
                latency_ms: t0.elapsed().as_secs_f64() * 1e3,
            },
        })
    }

    fn score_many(
        &self,
        prompts: &[String],
        candidates: &[Vec<String>],
    ) -> Result<Vec<Scored>, BackendError> {
        if prompts.len() <= 1 {
            return prompts
                .iter()
                .zip(candidates)
                .map(|(p, c)| self.score(p, c))
                .collect();
        }
        let t0 = Instant::now();
        let toks: Vec<Vec<LlamaToken>> = prompts
            .iter()
            .map(|p| self.tokenize(p))
            .collect::<Result<_, _>>()?;
        // Shared prefix: longest common token prefix of all prompts, leaving
        // every prompt at least one tail token to decode with logits.
        let mut shared = toks[0].len();
        for t in &toks {
            let common = toks[0].iter().zip(t).take_while(|(a, b)| a == b).count();
            shared = shared.min(common).min(t.len() - 1);
        }
        let mut st = self.state.lock().map_err(|_| err("lock", "poisoned"))?;
        let k = self.ensure_prefix(&mut st, &toks[0], shared)?;
        let tails: usize = toks.iter().map(|t| t.len() - shared).sum();
        // The context must hold prefix + every tail at once.
        if shared + tails >= self.n_ctx {
            drop(st);
            return prompts
                .iter()
                .zip(candidates)
                .map(|(p, c)| self.score(p, c))
                .collect();
        }

        let mut out = Vec::with_capacity(prompts.len());
        let mut batch = LlamaBatch::new(self.n_batch.max(tails), MAX_SEQ as i32);
        for group in (0..prompts.len()).collect::<Vec<_>>().chunks(MAX_SEQ - 1) {
            // Fork the prefix into one sequence per prompt of this group.
            for (slot, _) in group.iter().enumerate() {
                let seq = (slot + 1) as i32;
                st.ctx
                    .kv_cache_seq_rm(seq, None, None)
                    .map_err(|e| err("kv rm", e))?;
                st.ctx
                    .kv_cache_seq_cp(0, seq, None, Some(shared as u32))
                    .map_err(|e| err("kv cp", e))?;
            }
            batch.clear();
            let mut last_idx = Vec::with_capacity(group.len());
            for (slot, &pi) in group.iter().enumerate() {
                let seq = (slot + 1) as i32;
                let t = &toks[pi];
                for (j, tok) in t[shared..].iter().enumerate() {
                    let pos = shared + j;
                    batch
                        .add(*tok, pos as i32, &[seq], pos + 1 == t.len())
                        .map_err(|e| err("batch", e))?;
                }
                last_idx.push(batch.n_tokens() - 1);
            }
            st.ctx.decode(&mut batch).map_err(|e| err("decode", e))?;
            for (slot, &pi) in group.iter().enumerate() {
                let logits = st.ctx.get_logits_ith(last_idx[slot]);
                let logprobs = self.label_logprobs(logits, &candidates[pi]);
                out.push(Scored {
                    logprobs,
                    cost: ScoreCost {
                        prompt_evaluated: (toks[pi].len() - shared) as u64
                            + if pi == 0 { (shared - k) as u64 } else { 0 },
                        prompt_cached: if pi == 0 { k as u64 } else { shared as u64 },
                        latency_ms: 0.0, // reported once for the batch by the judge
                    },
                });
            }
            // Drop the forks; sequence 0 keeps the prefix for the next request.
            for slot in 0..group.len() {
                st.ctx
                    .kv_cache_seq_rm((slot + 1) as i32, None, None)
                    .map_err(|e| err("kv rm", e))?;
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if let Some(first) = out.first_mut() {
            first.cost.latency_ms = ms;
        }
        Ok(out)
    }

    fn model_name(&self) -> String {
        self.name.clone()
    }
}
