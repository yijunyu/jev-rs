//! Turn next-token log-probabilities over the option labels into typed
//! answers: restricted softmax, temperature calibration, confidence, and the
//! probability-weighted score.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Per-bucket temperatures, keyed like Laya's `temperature_by_options`:
/// `noul:2`, `choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`,
/// `score:2`, `score:3-5`, `score:6-10`. Missing bucket -> `default`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    #[serde(default = "one")]
    pub default: f64,
    #[serde(default)]
    pub by_bucket: HashMap<String, f64>,
}

fn one() -> f64 {
    1.0
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            default: 1.0,
            by_bucket: HashMap::new(),
        }
    }
}

impl Calibration {
    pub fn bucket(kind: &str, n: usize) -> String {
        let range = match n {
            0..=2 => "2".to_string(),
            3..=5 => "3-5".to_string(),
            6..=10 => "6-10".to_string(),
            _ => "11+".to_string(),
        };
        format!("{kind}:{range}")
    }

    pub fn temperature(&self, kind: &str, n: usize) -> f64 {
        let t = self
            .by_bucket
            .get(&Self::bucket(kind, n))
            .copied()
            .unwrap_or(self.default);
        if t > 0.0 && t.is_finite() {
            t
        } else {
            1.0
        }
    }

    /// Fit one temperature per bucket by grid search on negative
    /// log-likelihood. `samples` are `(bucket, raw_logprobs, gold_index)`.
    pub fn fit(samples: &[(String, Vec<f64>, usize)]) -> Self {
        let mut by_bucket: HashMap<String, Vec<(&Vec<f64>, usize)>> = HashMap::new();
        for (b, lp, g) in samples {
            by_bucket.entry(b.clone()).or_default().push((lp, *g));
        }
        // 0.1 .. 30.0: instruct models are often so peaked that the optimum
        // sits far above the 1-2 range encoders need.
        let grid: Vec<f64> = (1..=300).map(|i| 0.1 * i as f64).collect();
        let mut out = Calibration::default();
        for (b, rows) in by_bucket {
            let mut best = (f64::INFINITY, 1.0);
            for &t in &grid {
                let nll: f64 = rows
                    .iter()
                    .map(|(lp, g)| -softmax(lp, t)[*g].max(1e-12).ln())
                    .sum();
                if nll < best.0 {
                    best = (nll, t);
                }
            }
            out.by_bucket.insert(b, best.1);
        }
        out
    }
}

/// Softmax of `logprobs / t`. `-inf` entries get probability 0.
pub fn softmax(logprobs: &[f64], t: f64) -> Vec<f64> {
    let m = logprobs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        let n = logprobs.len().max(1);
        return vec![1.0 / n as f64; logprobs.len()];
    }
    let e: Vec<f64> = logprobs.iter().map(|&l| ((l - m) / t).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

/// TypeSafe's documented confidence: how far the top probability sits above
/// uniform, rescaled to [0, 1]. For 3 options: `(3·p_max − 1) / 2`.
pub fn confidence(probs: &[f64]) -> f64 {
    let n = probs.len();
    if n < 2 {
        return 1.0;
    }
    let pmax = probs.iter().cloned().fold(0.0, f64::max);
    ((n as f64 * pmax - 1.0) / (n as f64 - 1.0)).clamp(0.0, 1.0)
}

/// Probability-weighted mean level index.
pub fn expected_level(probs: &[f64]) -> f64 {
    probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()
}

pub fn argmax(probs: &[f64]) -> usize {
    let mut best = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p > probs[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_matches_docs() {
        let c = confidence(&[0.6, 0.3, 0.1]);
        assert!((c - 0.4).abs() < 1e-9);
        assert_eq!(confidence(&[1.0, 0.0]), 1.0);
        assert_eq!(confidence(&[0.5, 0.5]), 0.0);
    }

    #[test]
    fn softmax_handles_missing() {
        let p = softmax(&[0.0, f64::NEG_INFINITY], 1.0);
        assert!((p[0] - 1.0).abs() < 1e-12 && p[1] == 0.0);
        let p = softmax(&[-1.0, -1.0], 2.0);
        assert!((p[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn fit_flattens_overconfident() {
        // Model always says option 0 with logprob gap 5; gold is 0 only half
        // the time -> the fitted temperature must be well above 1.
        let mut s = Vec::new();
        for i in 0..20 {
            s.push(("choice:2".to_string(), vec![0.0, -5.0], i % 2));
        }
        let c = Calibration::fit(&s);
        assert!(c.temperature("choice", 2) > 3.0);
    }
}
