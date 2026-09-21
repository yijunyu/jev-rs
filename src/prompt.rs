//! Turn (state, question) into a prompt whose *next token* is the answer.
//!
//! The state goes into a shared prefix and every question into a suffix, so
//! all questions of one request share an identical prefix. Backends with a
//! prompt cache (llama-server `cache_prompt`, SGLang radix, vLLM APC, the
//! ds4-rs `AttnStepState` fork) prefill the state once.

use crate::protocol::{text_of, Question};

/// Which chat template wraps the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// `<|im_start|>` ChatML (Qwen, DeepSeek-V4 via ChatML, many fine-tunes).
    /// Includes an empty `<think>` block so Qwen3 does not reason first.
    ChatMl,
    /// Gemma `<start_of_turn>` template.
    Gemma,
    /// Llama 3 header template.
    Llama3,
    /// No template: plain text, for base models.
    Raw,
}

impl std::str::FromStr for Template {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "chatml" | "qwen" | "deepseek" => Ok(Self::ChatMl),
            "gemma" => Ok(Self::Gemma),
            "llama3" | "llama" => Ok(Self::Llama3),
            "raw" | "base" | "none" => Ok(Self::Raw),
            other => Err(format!("unknown template `{other}` (chatml|gemma|llama3|raw)")),
        }
    }
}

const SYSTEM: &str = "You are a fast, calibrated judge. You read a situation and answer one \
typed question about it with exactly one option label. Judge only from the situation; \
do not explain.";

/// A prompt ready for next-token scoring.
#[derive(Debug, Clone)]
pub struct Rendered {
    /// Everything up to and including the state (shared across questions).
    pub prefix: String,
    /// The question, its options, and the answer cue.
    pub suffix: String,
    /// Candidate answer tokens, in option order (e.g. `" A"`, `" B"`).
    pub labels: Vec<String>,
    /// Option keys in the same order as `labels`.
    pub keys: Vec<String>,
}

impl Rendered {
    pub fn prompt(&self) -> String {
        format!("{}{}", self.prefix, self.suffix)
    }
}

fn letter(i: usize) -> String {
    // Single-token labels for every tokenizer we know: " A" .. " Z".
    format!(" {}", (b'A' + i as u8) as char)
}

/// Render the shared prefix for a state.
pub fn prefix(template: Template, state_text: &str) -> String {
    let body = format!("Situation:\n{state_text}\n");
    match template {
        Template::ChatMl => format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{body}"
        ),
        Template::Gemma => format!("<start_of_turn>user\n{SYSTEM}\n\n{body}"),
        Template::Llama3 => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}<|eot_id|>\
<|start_header_id|>user<|end_header_id|>\n\n{body}"
        ),
        Template::Raw => format!("{SYSTEM}\n\n{body}"),
    }
}

fn close(template: Template) -> &'static str {
    match template {
        Template::ChatMl => "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:",
        Template::Gemma => "<end_of_turn>\n<start_of_turn>model\nAnswer:",
        Template::Llama3 => "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer:",
        Template::Raw => "\nAnswer:",
    }
}

/// Render one question. `order` optionally permutes choice options (for
/// position-bias averaging); it must be a permutation of `0..n`.
pub fn render(
    template: Template,
    state_prefix: &str,
    q: &Question,
    order: Option<&[usize]>,
) -> Rendered {
    let mut s = String::new();
    let mut labels = Vec::new();
    let mut keys = Vec::new();
    match q {
        Question::Noul { instructions, criteria } => {
            s.push_str("\nQuestion: ");
            s.push_str(&text_of(instructions));
            s.push_str("\nOptions:\n");
            let (yes, no) = match criteria {
                Some(c) => (text_of(&c.is_true), text_of(&c.is_false)),
                None => (String::new(), String::new()),
            };
            s.push_str(&format!("A) yes{}\n", desc(&yes)));
            s.push_str(&format!("B) no{}\n", desc(&no)));
            labels = vec![letter(0), letter(1)];
            keys = vec!["yes".into(), "no".into()];
        }
        Question::Choice { instructions, criteria } => {
            s.push_str("\nQuestion: ");
            s.push_str(&text_of(instructions));
            s.push_str("\nOptions:\n");
            let opts: Vec<(&String, &serde_json::Value)> = criteria.iter().collect();
            let n = opts.len();
            let idx: Vec<usize> = match order {
                Some(o) => o.to_vec(),
                None => (0..n).collect(),
            };
            for (pos, &i) in idx.iter().enumerate() {
                let (k, v) = opts[i];
                s.push_str(&format!("{}) {}{}\n", letter(pos).trim(), k, desc(&text_of(v))));
                labels.push(letter(pos));
                keys.push(k.clone());
            }
        }
        Question::Score { instructions, criteria } => {
            s.push_str("\nQuestion: ");
            s.push_str(&text_of(instructions));
            s.push_str("\nRate on this ordered scale (lowest first):\n");
            for (i, level) in criteria.iter().enumerate() {
                s.push_str(&format!("{}) level {}: {}\n", letter(i).trim(), i, text_of(level)));
                labels.push(letter(i));
                keys.push(i.to_string());
            }
        }
    }
    s.push_str("\nAnswer with the option letter only.");
    s.push_str(close(template));
    Rendered {
        prefix: state_prefix.to_string(),
        suffix: s,
        labels,
        keys,
    }
}

fn desc(d: &str) -> String {
    if d.trim().is_empty() {
        String::new()
    } else {
        format!(": {}", d.trim())
    }
}

/// Maximum options a single-token letter label can address.
pub const MAX_OPTIONS: usize = 26;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn choice_labels_follow_order() {
        let q: Question = serde_json::from_value(json!({
            "type":"choice","instructions":"Which?","criteria":{"x":null,"y":"why","z":null}
        }))
        .unwrap();
        let p = prefix(Template::Raw, "s");
        let r = render(Template::Raw, &p, &q, Some(&[2, 0, 1]));
        assert_eq!(r.keys, vec!["z", "x", "y"]);
        assert_eq!(r.labels, vec![" A", " B", " C"]);
        assert!(r.suffix.contains("B) x"));
        assert!(r.suffix.contains("C) y: why"));
        assert!(r.prompt().ends_with("Answer:"));
    }
}
