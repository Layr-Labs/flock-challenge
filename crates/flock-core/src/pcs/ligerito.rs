// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::build_eq_table;
use crate::merkle::{self, Hash, HashKind};
use crate::ntt::additive_ntt_f128::AdditiveNttF128;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LigeritoProfile {
    #[default]
    Fast,
    Slim,
    Secure,
}

impl LigeritoProfile {
    pub fn log_inv_rate(self) -> usize {
        match self {
            Self::Fast | Self::Secure => 1,
            Self::Slim => 2,
        }
    }

    pub fn security_bits(self) -> usize {
        match self {
            Self::Fast | Self::Slim => 100,
            Self::Secure => 120,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Slim => "slim",
            Self::Secure => "secure",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "slim" => Some(Self::Slim),
            "secure" => Some(Self::Secure),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProverConfig {
    pub log_inv_rates: Vec<usize>,
    pub recursive_steps: usize,
    pub initial_log_msg_cols: usize,
    pub initial_log_num_interleaved: usize,
    pub initial_k: usize,
    pub recursive_log_msg_cols: Vec<usize>,
    pub recursive_ks: Vec<usize>,

    pub queries: Vec<usize>,

    pub grinding_bits: Vec<usize>,

    pub fold_grinding_bits: Vec<usize>,

    pub ood_samples: Vec<usize>,

    pub merkle_hash: HashKind,
}

#[derive(Clone, Debug)]
pub struct VerifierConfig {
    pub log_inv_rates: Vec<usize>,
    pub recursive_steps: usize,
    pub initial_log_msg_cols: usize,
    pub initial_log_num_interleaved: usize,
    pub initial_k: usize,
    pub recursive_log_msg_cols: Vec<usize>,
    pub recursive_ks: Vec<usize>,

    pub queries: Vec<usize>,

    pub grinding_bits: Vec<usize>,

    pub fold_grinding_bits: Vec<usize>,

    pub ood_samples: Vec<usize>,

    pub merkle_hash: HashKind,
}

pub const UDR_PROXIMITY_LOSS: f64 = 0.0;

const UDR_TARGET_BITS: f64 = 100.0;

pub fn udr_queries(log_inv_rate: usize) -> usize {
    assert!(log_inv_rate > 0, "log_inv_rate=0 (rate 1) has no soundness");
    let per_q = udr_per_query_bits_asymptotic(log_inv_rate);
    (UDR_TARGET_BITS / per_q).ceil() as usize
}

pub fn default_config(
    log_n: usize,
    log_batch_size: usize,
    log_inv_rate: usize,
) -> Result<ProverConfig, &'static str> {
    let initial_k = log_batch_size;
    if log_n <= initial_k {
        return Err("log_n must be > initial_k");
    }

    let mut log_inv_rates = vec![log_inv_rate];
    let mut recursive_ks = Vec::new();
    let mut recursive_log_msg_cols = Vec::new();

    let mut n_running = log_n - initial_k;
    let mut rate_running = log_inv_rate;

    {
        let block_len_log = n_running + rate_running;
        let qs = udr_queries(rate_running);
        if (1usize << block_len_log) < qs {
            return Err("L0 block_len < udr_queries — log_n too small for chosen rate");
        }
    }

    while n_running > 5 {
        let k = 3.min(n_running);
        let log_msg_cols_next = n_running - k;

        let mut next_rate = rate_running + 1;
        loop {
            let bl = 1usize << (log_msg_cols_next + next_rate);
            let qs = udr_queries(next_rate);
            if bl >= qs {
                break;
            }
            next_rate += 1;
            if next_rate > 20 {
                return Err("could not find feasible recursive rate (level too deep)");
            }
        }
        recursive_log_msg_cols.push(log_msg_cols_next);
        recursive_ks.push(k);
        log_inv_rates.push(next_rate);
        n_running -= k;
        rate_running = next_rate;
    }

    if recursive_ks.is_empty() {
        return Err("log_n too small — no recursive levels for the Ligerito recursion");
    }

    let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
    let n_levels = log_inv_rates.len();
    let grinding_bits = vec![0usize; n_levels];

    Ok(ProverConfig {
        log_inv_rates: log_inv_rates.clone(),
        recursive_steps: recursive_ks.len(),
        initial_log_msg_cols: log_n - initial_k,
        initial_log_num_interleaved: initial_k,
        initial_k,
        recursive_log_msg_cols,
        recursive_ks,
        queries,
        grinding_bits,
        fold_grinding_bits: vec![0usize; n_levels],
        ood_samples: vec![0usize; n_levels],
        merkle_hash: HashKind::default(),
    })
}

struct LadderShape {
    log_inv_rates: Vec<usize>,
    log_msg_cols: Vec<usize>,
    log_num_interleaved: Vec<usize>,
    k_recursive: Vec<usize>,
    yr_log_n: usize,
}

fn derive_ladder_shape(
    log_n: usize,
    initial_k: usize,
    log_inv_rate: usize,
    queries_at_rate: &dyn Fn(usize) -> usize,
) -> Result<LadderShape, String> {
    if log_n <= initial_k {
        return Err("log_n must be > initial_k".into());
    }
    let mut shape = LadderShape {
        log_inv_rates: vec![log_inv_rate],
        log_msg_cols: vec![log_n - initial_k],
        log_num_interleaved: vec![initial_k],
        k_recursive: vec![initial_k],
        yr_log_n: 0,
    };
    let mut n_running = log_n - initial_k;
    let mut rate_running = log_inv_rate;
    if (1usize << (n_running + rate_running)) < queries_at_rate(rate_running) {
        return Err("L0 block_len < queries — log_n too small for chosen rate".into());
    }
    while n_running > 5 {
        let k = 3.min(n_running);
        let log_msg_cols_next = n_running - k;
        let mut next_rate = rate_running + 1;
        loop {
            if (1usize << (log_msg_cols_next + next_rate)) >= queries_at_rate(next_rate) {
                break;
            }
            next_rate += 1;
            if next_rate > 20 {
                return Err("could not find feasible recursive rate (level too deep)".into());
            }
        }
        shape.log_inv_rates.push(next_rate);
        shape.log_msg_cols.push(log_msg_cols_next);
        shape.log_num_interleaved.push(k);
        shape.k_recursive.push(k);
        n_running -= k;
        rate_running = next_rate;
    }
    if shape.k_recursive.len() < 2 {
        return Err("log_n too small — no recursive levels for the Ligerito recursion".into());
    }
    shape.yr_log_n = n_running;
    Ok(shape)
}

macro_rules! profile_configs {
    ($($m:literal),+ $(,)?) => {
        &[
            $(
                (($m, LigeritoProfile::Fast),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_fast.toml"))),
                (($m, LigeritoProfile::Slim),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_slim.toml"))),
                (($m, LigeritoProfile::Secure),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_secure.toml"))),
            )+
        ]
    };
}
const EMBEDDED_CONFIGS: &[((usize, LigeritoProfile), &str)] =
    profile_configs!(22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35);

pub fn embedded_security_config(m: usize, profile: LigeritoProfile) -> Option<&'static str> {
    EMBEDDED_CONFIGS.iter().find_map(|&(key, toml)| {
        if key == (m, profile) {
            Some(toml)
        } else {
            None
        }
    })
}

pub fn prover_config_for(
    log_n: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Result<ProverConfig, String> {
    let m = log_n + crate::pcs::LOG_PACKING;
    let toml = embedded_security_config(m, profile).ok_or_else(|| {
        format!(
            "no security config registered for (m={m}, profile={}). \
             Add a TOML at configs/ligerito/m{m}_{}.toml and register it in \
             EMBEDDED_CONFIGS, or call default_config explicitly for ad-hoc shapes.",
            profile.as_str(),
            profile.as_str(),
        )
    })?;
    let sec = LigeritoSecurityConfig::from_toml_str(toml)?;
    if sec.initial_k != log_batch_size {
        return Err(format!(
            "embedded config for (m={m}, profile={}) has \
             initial_k={} but caller requested log_batch_size={log_batch_size}",
            profile.as_str(),
            sec.initial_k
        ));
    }
    let (pv, _) = sec.to_prover_verifier_configs()?;
    Ok(pv)
}

pub fn verifier_config_for(
    log_n: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Result<VerifierConfig, String> {
    let m = log_n + crate::pcs::LOG_PACKING;
    let toml = embedded_security_config(m, profile).ok_or_else(|| {
        format!(
            "no security config registered for (m={m}, profile={})",
            profile.as_str()
        )
    })?;
    let sec = LigeritoSecurityConfig::from_toml_str(toml)?;
    if sec.initial_k != log_batch_size {
        return Err(format!(
            "embedded config for (m={m}, profile={}) has \
             initial_k={} but caller requested log_batch_size={log_batch_size}",
            profile.as_str(),
            sec.initial_k
        ));
    }
    let (_, vc) = sec.to_prover_verifier_configs()?;
    Ok(vc)
}

pub fn default_verifier_config(
    log_n: usize,
    log_batch_size: usize,
    log_inv_rate: usize,
) -> Result<VerifierConfig, &'static str> {
    let p = default_config(log_n, log_batch_size, log_inv_rate)?;
    Ok(VerifierConfig {
        log_inv_rates: p.log_inv_rates,
        recursive_steps: p.recursive_steps,
        initial_log_msg_cols: p.initial_log_msg_cols,
        initial_log_num_interleaved: p.initial_log_num_interleaved,
        initial_k: p.initial_k,
        recursive_log_msg_cols: p.recursive_log_msg_cols,
        recursive_ks: p.recursive_ks,
        queries: p.queries,
        grinding_bits: p.grinding_bits,
        fold_grinding_bits: p.fold_grinding_bits,
        ood_samples: p.ood_samples,
        merkle_hash: p.merkle_hash,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoundnessRegime {
    Udr,

    JohnsonOod,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrindingStep {
    PostCommitPreQueries,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeritoLevelConfig {
    pub log_inv_rate: usize,

    pub log_msg_cols: usize,

    pub log_num_interleaved: usize,

    pub k_recursive: usize,

    pub regime: SoundnessRegime,

    pub eta: Option<f64>,

    #[serde(default)]
    pub proximity_loss: Option<f64>,

    pub queries: usize,

    pub grinding_bits: usize,

    #[serde(default)]
    pub fold_grinding_bits: usize,

    #[serde(default)]
    pub ood_samples: usize,

    pub target_security_bits: usize,

    pub expected_eps_pg_bits: f64,

    pub expected_eps_query_bits: f64,

    #[serde(default)]
    pub expected_eps_ood_bits: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalBlockConfig {
    pub yr_log_n: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeritoSecurityConfig {
    pub m: usize,

    pub log_n: usize,

    pub initial_k: usize,

    pub target_security_bits: usize,

    pub analysis_version: String,

    pub field: String,

    pub hash: String,

    pub grinding_step: GrindingStep,

    pub levels: Vec<LigeritoLevelConfig>,

    pub final_block: FinalBlockConfig,
}

const ANALYSIS_LOG_Q: f64 = 128.0;

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

const PAPER_COMPAT_TOL_BITS: f64 = 0.6;

fn paper_thm_ca_johnson_log_a(log_inv_rate: usize, eta: f64, log_msg_cols: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let sqrt_rho = rho.sqrt();
    let gamma = 1.0 - sqrt_rho - eta;

    let m_param = ((sqrt_rho / (2.0 * eta)).ceil() as usize).max(3) as f64;
    let half = m_param + 0.5;
    let half5 = half.powi(5);
    let numerator = 2.0 * half5 + 3.0 * half * gamma * rho;
    let denominator = 3.0 * rho.powf(1.5);
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    let a = (numerator / denominator) * n + half / sqrt_rho;
    a.log2()
}

fn paper_johnson_log_a(
    log_inv_rate: usize,
    eta: f64,
    log_msg_cols: usize,
    log_num_interleaved: usize,
) -> f64 {
    let base = paper_thm_ca_johnson_log_a(log_inv_rate, eta, log_msg_cols);

    let row_union_penalty = (log_num_interleaved as f64 - 1.0).max(0.0);
    base + row_union_penalty
}

fn paper_per_query_bits(log_inv_rate: usize, eta: f64) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = 1.0 - rho.sqrt() - eta;
    (1.0 / (1.0 - gamma)).log2()
}

fn udr_gamma(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let delta = 1.0 - rho;
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    delta / 2.0 - 3.0 / (delta * n) - proximity_loss
}

fn udr_per_query_bits(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let gamma = udr_gamma(log_inv_rate, log_msg_cols, proximity_loss);
    (1.0 / (1.0 - gamma)).log2()
}

fn udr_per_query_bits_asymptotic(log_inv_rate: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = (1.0 - rho) / 2.0;
    (1.0 / (1.0 - gamma)).log2()
}

fn paper_thm_1_4_log_a(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let gamma = udr_gamma(log_inv_rate, log_msg_cols, proximity_loss);
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    (gamma * n + 1.0).log2()
}

fn johnson_interleaved_list_log2(log_inv_rate: usize, eta: f64) -> f64 {
    debug_assert!(
        eta > 0.0,
        "η must be > 0 to stay strictly below the Johnson radius"
    );
    let rho = (-(log_inv_rate as f64)).exp2();
    let sqrt_rho = rho.sqrt();
    let l_base = 1.0 / (2.0 * eta * sqrt_rho);
    l_base.log2()
}

fn paper_ood_bits(log_inv_rate: usize, eta: f64, mu_vars: usize, ood_samples: usize) -> f64 {
    let log2_l = johnson_interleaved_list_log2(log_inv_rate, eta);
    let log2_mu = (mu_vars as f64).log2();
    if ood_samples == 0 {
        ANALYSIS_LOG_Q - log2_l - log2_mu
    } else {
        ood_samples as f64 * (ANALYSIS_LOG_Q - log2_mu) - (2.0 * log2_l - 1.0)
    }
}

impl LigeritoLevelConfig {
    pub fn paper_predicted_bits(&self) -> (f64, f64) {
        match self.regime {
            SoundnessRegime::JohnsonOod => {
                let eta = self.eta.expect("JohnsonOod must have eta");

                let log_a = paper_johnson_log_a(
                    self.log_inv_rate,
                    eta,
                    self.log_msg_cols,
                    self.log_num_interleaved,
                );
                let eps_pg = ANALYSIS_LOG_Q - log_a;

                let per_q = paper_per_query_bits(self.log_inv_rate, eta);
                let eps_query = self.queries as f64 * per_q;
                (eps_pg, eps_query)
            }
            SoundnessRegime::Udr => {
                let proximity_loss = self
                    .proximity_loss
                    .expect("Udr regime must carry proximity_loss");

                let log_a =
                    paper_thm_1_4_log_a(self.log_inv_rate, self.log_msg_cols, proximity_loss);
                let eps_pg = ANALYSIS_LOG_Q - log_a;
                let per_q =
                    udr_per_query_bits(self.log_inv_rate, self.log_msg_cols, proximity_loss);
                let eps_query = self.queries as f64 * per_q;
                (eps_pg, eps_query)
            }
        }
    }

    pub fn paper_predicted_ood_bits(&self) -> Option<f64> {
        match self.regime {
            SoundnessRegime::JohnsonOod => {
                let eta = self.eta.expect("JohnsonOod must have eta");
                let mu = self.log_msg_cols + self.log_num_interleaved;
                Some(paper_ood_bits(self.log_inv_rate, eta, mu, self.ood_samples))
            }
            SoundnessRegime::Udr => None,
        }
    }
}

impl LigeritoSecurityConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.log_n + 7 != self.m {
            return Err(format!(
                "log_n ({}) + LOG_PACKING (7) != m ({})",
                self.log_n, self.m
            ));
        }

        self.merkle_hash()?;

        let levels_recursive_sum: usize = self.levels.iter().skip(1).map(|lv| lv.k_recursive).sum();
        let yr_log_n = self.final_block.yr_log_n;
        if self.initial_k + levels_recursive_sum + yr_log_n != self.log_n {
            return Err(format!(
                "shape mismatch: initial_k ({}) + Σ k_recursive ({}) + yr_log_n ({}) = {} ≠ log_n ({})",
                self.initial_k,
                levels_recursive_sum,
                yr_log_n,
                self.initial_k + levels_recursive_sum + yr_log_n,
                self.log_n,
            ));
        }

        let l0 = self
            .levels
            .first()
            .ok_or_else(|| "empty levels".to_string())?;
        if l0.k_recursive != self.initial_k {
            return Err(format!(
                "L0.k_recursive ({}) must equal initial_k ({})",
                l0.k_recursive, self.initial_k
            ));
        }
        if l0.log_num_interleaved != self.initial_k {
            return Err(format!(
                "L0.log_num_interleaved ({}) must equal initial_k ({})",
                l0.log_num_interleaved, self.initial_k
            ));
        }

        let mut dim_in = self.log_n;
        for (i, lv) in self.levels.iter().enumerate() {
            if lv.log_msg_cols + lv.log_num_interleaved != dim_in {
                return Err(format!(
                    "L{i}: log_msg_cols ({}) + log_num_interleaved ({}) ≠ input dim ({dim_in})",
                    lv.log_msg_cols, lv.log_num_interleaved
                ));
            }

            match (lv.regime, lv.eta) {
                (SoundnessRegime::Udr, Some(_)) => {
                    return Err(format!("L{i}: regime=udr but eta is set"));
                }
                (SoundnessRegime::JohnsonOod, None) => {
                    return Err(format!("L{i}: regime requires eta but eta is None"));
                }
                _ => {}
            }

            match (lv.regime, lv.proximity_loss) {
                (SoundnessRegime::Udr, None) => {
                    return Err(format!("L{i}: regime=udr but proximity_loss is missing"));
                }
                (SoundnessRegime::Udr, Some(eps)) if eps < 0.0 => {
                    return Err(format!("L{i}: proximity_loss must be ≥ 0, got {eps}"));
                }
                (SoundnessRegime::JohnsonOod, Some(_)) => {
                    return Err(format!("L{i}: proximity_loss is only valid for regime=udr"));
                }
                _ => {}
            }

            match lv.regime {
                SoundnessRegime::Udr if lv.ood_samples != 0 => {
                    return Err(format!(
                        "L{i}: regime=udr but ood_samples={} (unique decoding \
                         has list size 1 — no OOD binding step exists)",
                        lv.ood_samples
                    ));
                }
                SoundnessRegime::JohnsonOod if i == 0 && lv.ood_samples != 0 => {
                    return Err(format!(
                        "L0: ood_samples={} but L0 is bound by the opening's \
                         own evaluation claim (must be 0)",
                        lv.ood_samples
                    ));
                }
                SoundnessRegime::JohnsonOod if i > 0 && lv.ood_samples == 0 => {
                    return Err(format!(
                        "L{i}: regime=johnson_ood requires ood_samples ≥ 1 \
                         past L0 (the query counts assume single-codeword \
                         binding)"
                    ));
                }
                _ => {}
            }

            match (lv.regime, lv.expected_eps_ood_bits) {
                (SoundnessRegime::Udr, Some(_)) => {
                    return Err(format!("L{i}: regime=udr but expected_eps_ood_bits is set"));
                }
                (SoundnessRegime::JohnsonOod, None) => {
                    return Err(format!(
                        "L{i}: regime=johnson_ood requires expected_eps_ood_bits"
                    ));
                }
                (SoundnessRegime::JohnsonOod, Some(declared)) => {
                    let pred = lv
                        .paper_predicted_ood_bits()
                        .expect("JohnsonOod has an OOD prediction");
                    if (declared - pred).abs() > PAPER_COMPAT_TOL_BITS {
                        return Err(format!(
                            "L{i}: expected_eps_ood_bits ({declared:.2}) doesn't \
                             match prediction ({pred:.2}); tolerance ±{:.2} bits.",
                            PAPER_COMPAT_TOL_BITS
                        ));
                    }
                }
                _ => {}
            }

            let (pg_pred, q_pred) = lv.paper_predicted_bits();
            if (lv.expected_eps_pg_bits - pg_pred).abs() > PAPER_COMPAT_TOL_BITS {
                return Err(format!(
                    "L{i}: expected_eps_pg_bits ({:.2}) doesn't match \
                     {analysis} prediction ({:.2}); tolerance ±{:.2} bits. \
                     Re-derive Q, eta, or grinding so the declared diagnostic \
                     matches the formula.",
                    lv.expected_eps_pg_bits,
                    pg_pred,
                    PAPER_COMPAT_TOL_BITS,
                    analysis = self.analysis_version,
                ));
            }
            if (lv.expected_eps_query_bits - q_pred).abs() > PAPER_COMPAT_TOL_BITS {
                return Err(format!(
                    "L{i}: expected_eps_query_bits ({:.2}) doesn't match \
                     {analysis} prediction ({:.2}); tolerance ±{:.2} bits.",
                    lv.expected_eps_query_bits,
                    q_pred,
                    PAPER_COMPAT_TOL_BITS,
                    analysis = self.analysis_version,
                ));
            }

            if lv.target_security_bits > lv.grinding_bits
                && lv.expected_eps_query_bits + 1e-3
                    < (lv.target_security_bits - lv.grinding_bits) as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_query_bits ({:.2}) < target ({}) - grinding ({}) = {}",
                    lv.expected_eps_query_bits,
                    lv.target_security_bits,
                    lv.grinding_bits,
                    lv.target_security_bits - lv.grinding_bits
                ));
            }

            if lv.expected_eps_pg_bits + lv.fold_grinding_bits as f64 + 1e-3
                < lv.target_security_bits as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_pg_bits ({:.2}) + fold_grinding ({}) < target ({})",
                    lv.expected_eps_pg_bits, lv.fold_grinding_bits, lv.target_security_bits
                ));
            }

            if let Some(ood) = lv.expected_eps_ood_bits
                && ood + 1e-3 < lv.target_security_bits as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_ood_bits ({ood:.2}) < target ({}); \
                         increase ood_samples",
                    lv.target_security_bits
                ));
            }

            if lv.target_security_bits < self.target_security_bits {
                return Err(format!(
                    "L{i}: target_security_bits ({}) < global target ({})",
                    lv.target_security_bits, self.target_security_bits
                ));
            }

            dim_in -= lv.k_recursive;
        }

        if dim_in != yr_log_n {
            return Err(format!(
                "after consuming all levels, dim_in ({dim_in}) ≠ yr_log_n ({yr_log_n})"
            ));
        }

        Ok(())
    }

    pub fn derive_paper_compatible(
        m: usize,
        log_inv_rate: usize,
        target_security_bits: usize,
    ) -> Result<Self, String> {
        let log_n = m
            .checked_sub(crate::pcs::LOG_PACKING)
            .ok_or_else(|| format!("m ({m}) < LOG_PACKING (7)"))?;
        let initial_k = 6usize;
        let prover = default_config(log_n, initial_k, log_inv_rate).map_err(|e| e.to_string())?;
        let r = prover.recursive_steps;
        let mut levels = Vec::with_capacity(r + 1);

        let mut log_msg_cols_per_level = Vec::with_capacity(r + 1);
        let mut log_num_interleaved_per_level = Vec::with_capacity(r + 1);
        let mut k_recursive_per_level = Vec::with_capacity(r + 1);

        log_msg_cols_per_level.push(log_n - initial_k);
        log_num_interleaved_per_level.push(initial_k);
        k_recursive_per_level.push(initial_k);
        for i in 0..r {
            log_msg_cols_per_level.push(prover.recursive_log_msg_cols[i]);
            log_num_interleaved_per_level.push(prover.recursive_ks[i]);
            k_recursive_per_level.push(prover.recursive_ks[i]);
        }
        for i in 0..=r {
            let rate = prover.log_inv_rates[i];

            let proximity_loss = UDR_PROXIMITY_LOSS;
            let per_q = udr_per_query_bits(rate, log_msg_cols_per_level[i], proximity_loss);
            let queries = ((target_security_bits as f64) / per_q).ceil() as usize;

            let log_a = paper_thm_1_4_log_a(rate, log_msg_cols_per_level[i], proximity_loss);
            let eps_pg = ANALYSIS_LOG_Q - log_a;

            let fold_grinding_bits =
                ((target_security_bits as f64) - eps_pg).ceil().max(0.0) as usize;
            let eps_query = queries as f64 * per_q;
            levels.push(LigeritoLevelConfig {
                log_inv_rate: rate,
                log_msg_cols: log_msg_cols_per_level[i],
                log_num_interleaved: log_num_interleaved_per_level[i],
                k_recursive: k_recursive_per_level[i],
                regime: SoundnessRegime::Udr,
                eta: None,
                proximity_loss: Some(proximity_loss),
                queries,
                grinding_bits: 0,
                fold_grinding_bits,
                ood_samples: 0,
                target_security_bits,
                expected_eps_pg_bits: round1(eps_pg),
                expected_eps_query_bits: round1(eps_query),
                expected_eps_ood_bits: None,
            });
        }

        let total_recursive: usize = prover.recursive_ks.iter().sum();
        let yr_log_n = log_n - initial_k - total_recursive;
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits,
            analysis_version: "no_row_union_over_ben_sasson_2025_cor_1_4".into(),
            field: "f128".into(),
            hash: "sha256".into(),
            grinding_step: GrindingStep::PostCommitPreQueries,
            levels,
            final_block: FinalBlockConfig { yr_log_n },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn derive_profile(m: usize, profile: LigeritoProfile) -> Result<Self, String> {
        const JOHNSON_ETA: f64 = 0.02;
        let target_bits = profile.security_bits();
        let log_inv_rate = profile.log_inv_rate();
        let query_grind: usize = match profile {
            LigeritoProfile::Slim => 16,
            LigeritoProfile::Fast | LigeritoProfile::Secure => 0,
        };
        let log_n = m
            .checked_sub(crate::pcs::LOG_PACKING)
            .ok_or_else(|| format!("m ({m}) < LOG_PACKING (7)"))?;
        let initial_k = 6usize;

        let per_query_bits_feas = |rate: usize| -> f64 {
            match profile {
                LigeritoProfile::Secure => udr_per_query_bits_asymptotic(rate),
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    paper_per_query_bits(rate, JOHNSON_ETA)
                }
            }
        };

        let t_feas = target_bits as f64 + 5.0;
        let queries_feas = |rate: usize| -> usize {
            ((t_feas - query_grind as f64).max(1.0) / per_query_bits_feas(rate)).ceil() as usize
        };
        let shape = derive_ladder_shape(log_n, initial_k, log_inv_rate, &queries_feas)?;
        let n_levels = shape.log_inv_rates.len();

        let t = target_bits as f64;

        let mut levels = Vec::with_capacity(n_levels);
        for i in 0..n_levels {
            let rate = shape.log_inv_rates[i];
            let cols = shape.log_msg_cols[i];
            let ilv = shape.log_num_interleaved[i];

            let per_q = match profile {
                LigeritoProfile::Secure => udr_per_query_bits(rate, cols, UDR_PROXIMITY_LOSS),
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    paper_per_query_bits(rate, JOHNSON_ETA)
                }
            };
            let queries = ((t - query_grind as f64).max(1.0) / per_q).ceil() as usize;
            if queries > (1usize << (cols + rate)) {
                return Err(format!(
                    "L{i}: {queries} queries exceed block length 2^{}",
                    cols + rate
                ));
            }
            let eps_query = queries as f64 * per_q;

            let (regime, eta, proximity_loss, eps_pg, ood_samples, eps_ood) = match profile {
                LigeritoProfile::Secure => {
                    let eps_pg =
                        ANALYSIS_LOG_Q - paper_thm_1_4_log_a(rate, cols, UDR_PROXIMITY_LOSS);
                    (
                        SoundnessRegime::Udr,
                        None,
                        Some(UDR_PROXIMITY_LOSS),
                        eps_pg,
                        0usize,
                        None,
                    )
                }
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    let eps_pg = ANALYSIS_LOG_Q - paper_johnson_log_a(rate, JOHNSON_ETA, cols, ilv);
                    let mu = cols + ilv;
                    let ood_samples = if i == 0 {
                        0
                    } else {
                        (1..=8usize)
                            .find(|&s| paper_ood_bits(rate, JOHNSON_ETA, mu, s) >= t)
                            .ok_or_else(|| {
                                format!("L{i}: no OOD sample count reaches {t:.1} bits")
                            })?
                    };
                    let eps_ood = paper_ood_bits(rate, JOHNSON_ETA, mu, ood_samples);
                    (
                        SoundnessRegime::JohnsonOod,
                        Some(JOHNSON_ETA),
                        None,
                        eps_pg,
                        ood_samples,
                        Some(round1(eps_ood)),
                    )
                }
            };
            let fold_grinding_bits = (t - eps_pg).ceil().max(0.0) as usize;

            levels.push(LigeritoLevelConfig {
                log_inv_rate: rate,
                log_msg_cols: cols,
                log_num_interleaved: ilv,
                k_recursive: shape.k_recursive[i],
                regime,
                eta,
                proximity_loss,
                queries,
                grinding_bits: query_grind,
                fold_grinding_bits,
                ood_samples,
                target_security_bits: target_bits,
                expected_eps_pg_bits: round1(eps_pg),
                expected_eps_query_bits: round1(eps_query),
                expected_eps_ood_bits: eps_ood,
            });
        }

        let analysis_version = match profile {
            LigeritoProfile::Secure => "no_row_union_over_ben_sasson_2025_cor_1_4",
            LigeritoProfile::Fast | LigeritoProfile::Slim => {
                "johnson_ood_row_union_over_bchks25_thm_4_6"
            }
        };
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits: target_bits,
            analysis_version: analysis_version.into(),
            field: "f128".into(),
            hash: "sha256".into(),
            grinding_step: GrindingStep::PostCommitPreQueries,
            levels,
            final_block: FinalBlockConfig {
                yr_log_n: shape.yr_log_n,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        let cfg: Self = toml::from_str(s).map_err(|e| format!("toml parse: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml_string(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("toml serialize: {e}"))
    }

    pub fn to_prover_verifier_configs(&self) -> Result<(ProverConfig, VerifierConfig), String> {
        self.validate()?;
        let merkle_hash = self.merkle_hash()?;
        let log_inv_rates: Vec<usize> = self.levels.iter().map(|lv| lv.log_inv_rate).collect();
        let recursive_ks: Vec<usize> = self
            .levels
            .iter()
            .skip(1)
            .map(|lv| lv.k_recursive)
            .collect();
        let recursive_log_msg_cols: Vec<usize> = self
            .levels
            .iter()
            .skip(1)
            .map(|lv| lv.log_msg_cols)
            .collect();
        let queries: Vec<usize> = self.levels.iter().map(|lv| lv.queries).collect();
        let grinding_bits: Vec<usize> = self.levels.iter().map(|lv| lv.grinding_bits).collect();
        let fold_grinding_bits: Vec<usize> =
            self.levels.iter().map(|lv| lv.fold_grinding_bits).collect();
        let ood_samples: Vec<usize> = self.levels.iter().map(|lv| lv.ood_samples).collect();
        let prover = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: self.levels[0].log_msg_cols,
            initial_log_num_interleaved: self.initial_k,
            initial_k: self.initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: recursive_ks.clone(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: fold_grinding_bits.clone(),
            ood_samples: ood_samples.clone(),
            merkle_hash,
        };
        let verifier = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: self.levels[0].log_msg_cols,
            initial_log_num_interleaved: self.initial_k,
            initial_k: self.initial_k,
            recursive_log_msg_cols,
            recursive_ks,
            queries,
            grinding_bits,
            fold_grinding_bits,
            ood_samples,
            merkle_hash,
        };
        Ok((prover, verifier))
    }

    pub fn merkle_hash(&self) -> Result<HashKind, String> {
        HashKind::parse(&self.hash).map_err(|e| format!("security config `hash`: {e}"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursiveProof {
    pub opened_rows: Vec<Vec<F128>>,

    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalProof {
    pub yr: Vec<F128>,

    pub opened_rows: Vec<Vec<F128>>,
    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LigeritoProof {
    pub initial_root: Hash,
    pub initial_proof: RecursiveProof,
    pub recursive_roots: Vec<Hash>,
    pub recursive_proofs: Vec<RecursiveProof>,
    pub final_proof: FinalProof,
    pub sumcheck_transcript: Vec<SumcheckMessage>,

    #[serde(default)]
    pub grinding_nonces: Vec<u64>,

    #[serde(default)]
    pub ood_values: Vec<F128>,

    #[serde(default)]
    pub fold_grinding_nonces: Vec<u64>,
}

impl LigeritoProof {
    pub fn size_bytes(&self) -> usize {
        const ELEM: usize = core::mem::size_of::<F128>();
        let level_bytes = |p: &RecursiveProof| -> usize {
            p.opened_rows.iter().map(|r| r.len() * ELEM).sum::<usize>() + p.merkle_proof.len() * 32
        };
        let mut total = 32;
        total += self.recursive_roots.len() * 32;
        total += level_bytes(&self.initial_proof);
        for p in &self.recursive_proofs {
            total += level_bytes(p);
        }
        total += self.final_proof.yr.len() * ELEM
            + self
                .final_proof
                .opened_rows
                .iter()
                .map(|r| r.len() * ELEM)
                .sum::<usize>()
            + self.final_proof.merkle_proof.len() * 32;
        total += self.sumcheck_transcript.len() * 2 * ELEM;
        total += self.ood_values.len() * ELEM;
        total += (self.grinding_nonces.len() + self.fold_grinding_nonces.len()) * 8;
        total
    }

    pub fn print_size_breakdown(&self) {
        const ELEM: usize = core::mem::size_of::<F128>();
        let kb = |b: usize| {
            if b >= 1024 * 1024 {
                format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
            } else if b >= 1024 {
                format!("{:.1} KB", b as f64 / 1024.0)
            } else {
                format!("{} B", b)
            }
        };

        let roots_b = 32 * (1 + self.recursive_roots.len());
        let init_opened: usize = self
            .initial_proof
            .opened_rows
            .iter()
            .map(|r| r.len() * ELEM)
            .sum();
        let init_merkle: usize = self.initial_proof.merkle_proof.len() * 32;
        eprintln!(
            "  L0 (initial): opened={} ({}q × {}lanes × {}B)  merkle={}",
            kb(init_opened),
            self.initial_proof.opened_rows.len(),
            self.initial_proof
                .opened_rows
                .first()
                .map_or(0, |r| r.len()),
            ELEM,
            kb(init_merkle),
        );
        let mut total_opened = init_opened;
        let mut total_merkle = init_merkle;
        for (i, rp) in self.recursive_proofs.iter().enumerate() {
            let opened: usize = rp.opened_rows.iter().map(|r| r.len() * ELEM).sum();
            let merkle: usize = rp.merkle_proof.len() * 32;
            eprintln!(
                "  L{} (recursive): opened={} ({}q × {}lanes × {}B)  merkle={}",
                i + 1,
                kb(opened),
                rp.opened_rows.len(),
                rp.opened_rows.first().map_or(0, |r| r.len()),
                ELEM,
                kb(merkle),
            );
            total_opened += opened;
            total_merkle += merkle;
        }
        let final_opened: usize = self
            .final_proof
            .opened_rows
            .iter()
            .map(|r| r.len() * ELEM)
            .sum();
        let final_merkle: usize = self.final_proof.merkle_proof.len() * 32;
        let yr_b = self.final_proof.yr.len() * ELEM;
        eprintln!(
            "  L{} (final):  opened={} ({}q × {}lanes × {}B)  merkle={}  yr={} ({}×{}B)",
            self.recursive_proofs.len() + 1,
            kb(final_opened),
            self.final_proof.opened_rows.len(),
            self.final_proof.opened_rows.first().map_or(0, |r| r.len()),
            ELEM,
            kb(final_merkle),
            kb(yr_b),
            self.final_proof.yr.len(),
            ELEM,
        );
        total_opened += final_opened;
        total_merkle += final_merkle;
        let tx_b = self.sumcheck_transcript.len() * 2 * ELEM;
        eprintln!(
            "  TOTALS: roots={}  opened={}  merkle={}  yr={}  transcript={} ({}×2×{}B)  GRAND={}",
            kb(roots_b),
            kb(total_opened),
            kb(total_merkle),
            kb(yr_b),
            kb(tx_b),
            self.sumcheck_transcript.len(),
            ELEM,
            kb(self.size_bytes()),
        );
    }
}

pub(crate) fn partial_eval_lsb(evals: &[F128], rs: &[F128]) -> Vec<F128> {
    let mut cur = evals.to_vec();
    for &r in rs {
        let half = cur.len() / 2;

        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            let v0 = cur[2 * i];
            let v1 = cur[2 * i + 1];
            next.push(v0 + (v0 + v1) * r);
        }
        cur = next;
    }
    cur
}

#[cfg(test)]
pub(crate) fn eval_mle_lsb(evals: &[F128], point: &[F128]) -> F128 {
    let folded = partial_eval_lsb(evals, point);
    debug_assert_eq!(folded.len(), 1);
    folded[0]
}

#[inline]
fn next_s(s: F128, s_at_root: F128) -> F128 {
    s * s + s_at_root * s
}

pub(crate) fn eval_sk_at_vks(log_n: usize) -> Vec<F128> {
    let mut sks_vks = vec![F128::ZERO; log_n + 1];
    sks_vks[0] = F128::ONE;
    if log_n == 0 {
        return sks_vks;
    }
    let mut layer: Vec<F128> = (1..=log_n).map(|i| F128::new(1u64 << i, 0)).collect();
    let mut cur_len = log_n;
    for i in 0..log_n {
        for j in 0..cur_len {
            let sk_at_vk = next_s(layer[j], sks_vks[i]);
            if j == 0 {
                sks_vks[i + 1] = sk_at_vk;
            } else {
                layer[j - 1] = sk_at_vk;
            }
        }
        cur_len -= 1;
    }
    sks_vks
}

#[cfg(test)]
#[allow(dead_code)]
fn evaluate_scaled_basis_inplace(
    sks_at_x: &mut [F128],
    basis: &mut [F128],
    sks_vks: &[F128],
    inv_sks_vks: &[F128],
    x: F128,
    alpha: F128,
) {
    let log_n = basis.len().trailing_zeros() as usize;
    debug_assert_eq!(basis.len(), 1 << log_n);
    debug_assert!(sks_at_x.len() >= log_n);
    debug_assert!(inv_sks_vks.len() > log_n);

    if log_n > 0 {
        sks_at_x[0] = x;
        for i in 1..log_n {
            sks_at_x[i] = next_s(sks_at_x[i - 1], sks_vks[i - 1]);
        }

        for i in 0..log_n {
            sks_at_x[i] *= inv_sks_vks[i];
        }
    }

    basis[0] = alpha;
    for k in 0..log_n {
        let s_at_x = sks_at_x[k];
        let current_len = 1 << k;
        let (src, dst) = basis[..2 * current_len].split_at_mut(current_len);
        crate::field::f128_slice::scalar_mul_into(dst, src, s_at_x);
    }
}

pub(crate) fn induce_sumcheck_enforced_sum(
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> F128 {
    assert_eq!(opened_rows.len(), queries.len());
    let eq = build_eq_table(v_challenges);
    let n_queries = queries.len();
    let alpha_weights: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        build_eq_table(alpha).into_iter().take(n_queries).collect()
    };
    let mut sum_unred = crate::field::F256Unreduced::ZERO;
    for (i, row) in opened_rows.iter().enumerate() {
        debug_assert_eq!(row.len(), eq.len());
        let dot = crate::field::f128_slice::inner_product(row, &eq);
        sum_unred ^= alpha_weights[i].mul_unreduced(dot);
    }
    sum_unred.reduce()
}

#[inline]
pub(crate) fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (n - 1).ilog2() as usize + 1
    }
}

pub(crate) fn induce_sumcheck_evaluate_at_residual(
    log_msg_cols: usize,
    sks_vks: &[F128],
    queries: &[usize],
    alpha: &[F128],
    ris_for_basis: &[F128],
    yr_log_n: usize,
) -> Vec<F128> {
    use crate::lincheck::build_eq_table;
    use rayon::prelude::*;
    assert_eq!(ris_for_basis.len() + yr_log_n, log_msg_cols);
    let n_queries = queries.len();
    let yr_len = 1usize << yr_log_n;

    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    let inv_sks_vks: Vec<F128> = sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
        .collect();

    let prefix_len = ris_for_basis.len();

    struct PerQuery {
        prefix_prod: F128,
        suffix_w: Vec<F128>,
    }
    let compute_query = |&q: &usize| -> PerQuery {
        let q_field = F128::new(q as u64, 0);

        let mut sks_at_x = Vec::with_capacity(log_msg_cols.max(1));
        if log_msg_cols > 0 {
            sks_at_x.push(q_field);
            for k in 1..log_msg_cols {
                sks_at_x.push(next_s(sks_at_x[k - 1], sks_vks[k - 1]));
            }
            for k in 0..log_msg_cols {
                sks_at_x[k] *= inv_sks_vks[k];
            }
        }

        let mut prefix_prod = F128::ONE;
        for k in 0..prefix_len {
            prefix_prod *= F128::ONE + ris_for_basis[k] * (F128::ONE + sks_at_x[k]);
        }
        let suffix_w = if log_msg_cols > prefix_len {
            sks_at_x[prefix_len..].to_vec()
        } else {
            Vec::new()
        };
        PerQuery {
            prefix_prod,
            suffix_w,
        }
    };

    const PAR_FLOOR: usize = 1024;
    let per_query: Vec<PerQuery> = if n_queries > PAR_FLOOR {
        queries.par_iter().map(compute_query).collect()
    } else {
        queries.iter().map(compute_query).collect()
    };

    let compute_y = |y: usize| -> F128 {
        let mut sum = F128::ZERO;
        for i in 0..n_queries {
            let pq = &per_query[i];
            let mut suffix_prod = F128::ONE;
            for j in 0..yr_log_n {
                let p_j = if (y >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                };
                suffix_prod *= F128::ONE + p_j * (F128::ONE + pq.suffix_w[j]);
            }
            sum += alpha_pows[i] * pq.prefix_prod * suffix_prod;
        }
        sum
    };
    if yr_len > PAR_FLOOR {
        (0..yr_len).into_par_iter().map(compute_y).collect()
    } else {
        (0..yr_len).map(compute_y).collect()
    }
}

pub(crate) fn induce_sumcheck_poly(
    log_msg_cols: usize,
    sks_vks: &[F128],
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    use rayon::prelude::*;
    let n = 1usize << log_msg_cols;
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);
    debug_assert_eq!(
        v_challenges.len(),
        opened_rows
            .first()
            .map(|r| r.len().trailing_zeros() as usize)
            .unwrap_or(0)
    );

    let eq = build_eq_table(v_challenges);

    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    let inv_sks_vks: Vec<F128> = sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
        .collect();

    let active_threads = rayon::current_num_threads().max(1).min(n_queries.max(1));
    let chunk_size = n_queries.div_ceil(active_threads);

    let partials: Vec<(Vec<F128>, F128)> = (0..active_threads)
        .into_par_iter()
        .filter_map(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(n_queries);
            if start >= end {
                return None;
            }
            let mut accum_basis = vec![F128::ZERO; n];
            let half = (n >> 1).max(1);

            let mut local_basis = vec![F128::ZERO; half];
            let mut sks_at_x = vec![F128::ZERO; log_msg_cols.max(1)];
            let mut local_sum = F128::ZERO;

            for i in start..end {
                let row = &opened_rows[i];
                let q = queries[i];
                let ap = alpha_pows[i];

                let dot = crate::field::f128_slice::inner_product(row, &eq);
                local_sum += dot * ap;

                if log_msg_cols == 0 {
                    accum_basis[0] += ap;
                } else {
                    let q_field = F128::new(q as u64, 0);
                    sks_at_x[0] = q_field;
                    for k in 1..log_msg_cols {
                        sks_at_x[k] = next_s(sks_at_x[k - 1], sks_vks[k - 1]);
                    }
                    for k in 0..log_msg_cols {
                        sks_at_x[k] *= inv_sks_vks[k];
                    }
                    local_basis[0] = ap;
                    for k in 0..(log_msg_cols - 1) {
                        let s_at_x = sks_at_x[k];
                        let current_len = 1 << k;
                        let (src, dst) = local_basis[..2 * current_len].split_at_mut(current_len);
                        crate::field::f128_slice::scalar_mul_into(dst, src, s_at_x);
                    }
                    let top_s = sks_at_x[log_msg_cols - 1];
                    let (acc_lo, acc_hi) = accum_basis.split_at_mut(half);
                    crate::field::f128_slice::xor_assign(acc_lo, &local_basis);
                    crate::field::f128_slice::scalar_mul_add_assign(acc_hi, &local_basis, top_s);
                }
            }
            Some((accum_basis, local_sum))
        })
        .collect();

    let mut iter = partials.into_iter();
    let Some((mut basis_poly, mut enforced_sum)) = iter.next() else {
        return (vec![F128::ZERO; n], F128::ZERO);
    };
    for (lb, ls) in iter {
        crate::field::f128_slice::xor_assign(&mut basis_poly, &lb);
        enforced_sum += ls;
    }

    (basis_poly, enforced_sum)
}

#[inline]
fn transpose_butterfly_slice(top: &mut [F128], bot: &mut [F128], t: F128) {
    debug_assert_eq!(top.len(), bot.len());
    let len = top.len();

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let vt = _mm_set_epi64x(t.hi as i64, t.lo as i64);
        let (tr0, tr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vt);
        let top_ptr = top.as_mut_ptr() as *mut __m128i;
        let bot_ptr = bot.as_mut_ptr() as *mut __m128i;
        let mut i = 0usize;
        while i + 2 <= len {
            let a0 = _mm_loadu_si128(top_ptr.add(i));
            let b0 = _mm_loadu_si128(bot_ptr.add(i));
            let a1 = _mm_loadu_si128(top_ptr.add(i + 1));
            let b1 = _mm_loadu_si128(bot_ptr.add(i + 1));
            let s0 = _mm_xor_si128(a0, b0);
            let s1 = _mm_xor_si128(a1, b1);
            let (p0, p1) =
                crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(s0, s1, tr0, tr1, tr0, tr1);
            _mm_storeu_si128(top_ptr.add(i), s0);
            _mm_storeu_si128(top_ptr.add(i + 1), s1);
            _mm_storeu_si128(bot_ptr.add(i), _mm_xor_si128(p0, b0));
            _mm_storeu_si128(bot_ptr.add(i + 1), _mm_xor_si128(p1, b1));
            i += 2;
        }
        while i < len {
            let a = _mm_loadu_si128(top_ptr.add(i));
            let b = _mm_loadu_si128(bot_ptr.add(i));
            let s = _mm_xor_si128(a, b);
            let p = crate::gf2_kernels::k0_f128_mul_const_m128i(s, tr0, tr1);
            _mm_storeu_si128(top_ptr.add(i), s);
            _mm_storeu_si128(bot_ptr.add(i), _mm_xor_si128(p, b));
            i += 1;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let t_arr = [t.lo, t.hi];
        let vt = vld1q_u64(t_arr.as_ptr());
        let (tr0, tr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vt);
        let top_ptr = top.as_mut_ptr() as *mut u64;
        let bot_ptr = bot.as_mut_ptr() as *mut u64;
        let mut i = 0usize;
        while i + 2 <= len {
            let idx = 2 * i;
            let a0 = vld1q_u64(top_ptr.add(idx));
            let b0 = vld1q_u64(bot_ptr.add(idx));
            let a1 = vld1q_u64(top_ptr.add(idx + 2));
            let b1 = vld1q_u64(bot_ptr.add(idx + 2));
            let s0 = veorq_u64(a0, b0);
            let s1 = veorq_u64(a1, b1);
            let (p0, p1) =
                crate::gf2_kernels::k0_f128_mul_const_vec2_neon(s0, s1, tr0, tr1, tr0, tr1);
            vst1q_u64(top_ptr.add(idx), s0);
            vst1q_u64(top_ptr.add(idx + 2), s1);
            vst1q_u64(bot_ptr.add(idx), veorq_u64(p0, b0));
            vst1q_u64(bot_ptr.add(idx + 2), veorq_u64(p1, b1));
            i += 2;
        }
        while i < len {
            let idx = 2 * i;
            let a = vld1q_u64(top_ptr.add(idx));
            let b = vld1q_u64(bot_ptr.add(idx));
            let s = veorq_u64(a, b);
            let p = crate::gf2_kernels::k0_f128_mul_const_neon(s, tr0, tr1);
            vst1q_u64(top_ptr.add(idx), s);
            vst1q_u64(bot_ptr.add(idx), veorq_u64(p, b));
            i += 1;
        }
        return;
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes"),
    )))]
    for (a_ref, b_ref) in top.iter_mut().zip(bot.iter_mut()) {
        let a = *a_ref;
        let b = *b_ref;
        let s = a + b;
        *a_ref = s;
        *b_ref = t * s + b;
    }
}

#[inline]
fn transpose_butterfly_fused_2layer_slices(
    q0: &mut [F128],
    q1: &mut [F128],
    q2: &mut [F128],
    q3: &mut [F128],
    t_o: F128,
    t_ia: F128,
    t_ib: F128,
) {
    let len = q0.len();
    debug_assert_eq!(q1.len(), len);
    debug_assert_eq!(q2.len(), len);
    debug_assert_eq!(q3.len(), len);

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let vto = _mm_set_epi64x(t_o.hi as i64, t_o.lo as i64);
        let vtia = _mm_set_epi64x(t_ia.hi as i64, t_ia.lo as i64);
        let vtib = _mm_set_epi64x(t_ib.hi as i64, t_ib.lo as i64);
        let (to0, to1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vto);
        let (tia0, tia1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vtia);
        let (tib0, tib1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vtib);
        let p0 = q0.as_mut_ptr() as *mut __m128i;
        let p1 = q1.as_mut_ptr() as *mut __m128i;
        let p2 = q2.as_mut_ptr() as *mut __m128i;
        let p3 = q3.as_mut_ptr() as *mut __m128i;
        for i in 0..len {
            let x0 = _mm_loadu_si128(p0.add(i));
            let x1 = _mm_loadu_si128(p1.add(i));
            let x2 = _mm_loadu_si128(p2.add(i));
            let x3 = _mm_loadu_si128(p3.add(i));
            let s01 = _mm_xor_si128(x0, x1);
            let s23 = _mm_xor_si128(x2, x3);
            let (m01, m23) =
                crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(s01, s23, tia0, tia1, tib0, tib1);
            let y1 = _mm_xor_si128(m01, x1);
            let y3 = _mm_xor_si128(m23, x3);
            let z0 = _mm_xor_si128(s01, s23);
            let z1 = _mm_xor_si128(y1, y3);
            let (m02, m13) =
                crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(z0, z1, to0, to1, to0, to1);
            let z2 = _mm_xor_si128(m02, s23);
            let z3 = _mm_xor_si128(m13, y3);
            _mm_storeu_si128(p0.add(i), z0);
            _mm_storeu_si128(p1.add(i), z1);
            _mm_storeu_si128(p2.add(i), z2);
            _mm_storeu_si128(p3.add(i), z3);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let to_arr = [t_o.lo, t_o.hi];
        let tia_arr = [t_ia.lo, t_ia.hi];
        let tib_arr = [t_ib.lo, t_ib.hi];
        let vto = vld1q_u64(to_arr.as_ptr());
        let vtia = vld1q_u64(tia_arr.as_ptr());
        let vtib = vld1q_u64(tib_arr.as_ptr());
        let (to0, to1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vto);
        let (tia0, tia1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vtia);
        let (tib0, tib1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vtib);
        let p0 = q0.as_mut_ptr() as *mut u64;
        let p1 = q1.as_mut_ptr() as *mut u64;
        let p2 = q2.as_mut_ptr() as *mut u64;
        let p3 = q3.as_mut_ptr() as *mut u64;
        for i in 0..len {
            let idx = 2 * i;
            let x0 = vld1q_u64(p0.add(idx));
            let x1 = vld1q_u64(p1.add(idx));
            let x2 = vld1q_u64(p2.add(idx));
            let x3 = vld1q_u64(p3.add(idx));
            let s01 = veorq_u64(x0, x1);
            let s23 = veorq_u64(x2, x3);
            let (m01, m23) =
                crate::gf2_kernels::k0_f128_mul_const_vec2_neon(s01, s23, tia0, tia1, tib0, tib1);
            let y1 = veorq_u64(m01, x1);
            let y3 = veorq_u64(m23, x3);
            let z0 = veorq_u64(s01, s23);
            let z1 = veorq_u64(y1, y3);
            let (m02, m13) =
                crate::gf2_kernels::k0_f128_mul_const_vec2_neon(z0, z1, to0, to1, to0, to1);
            let z2 = veorq_u64(m02, s23);
            let z3 = veorq_u64(m13, y3);
            vst1q_u64(p0.add(idx), z0);
            vst1q_u64(p1.add(idx), z1);
            vst1q_u64(p2.add(idx), z2);
            vst1q_u64(p3.add(idx), z3);
        }
        return;
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes"),
    )))]
    for i in 0..len {
        let x0 = q0[i];
        let x1 = q1[i];
        let x2 = q2[i];
        let x3 = q3[i];
        let s01 = x0 + x1;
        let s23 = x2 + x3;
        let y1 = t_ia * s01 + x1;
        let y3 = t_ib * s23 + x3;
        let z0 = s01 + s23;
        let z1 = y1 + y3;
        q0[i] = z0;
        q1[i] = z1;
        q2[i] = t_o * z0 + s23;
        q3[i] = t_o * z1 + y3;
    }
}

fn transpose_forward_ntt_layers_range(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    log_d: usize,
    start_layer_inclusive: usize,
    min_layer_inclusive: usize,
) {
    use rayon::prelude::*;
    if start_layer_inclusive < min_layer_inclusive {
        return;
    }
    let n_threads = rayon::current_num_threads().max(1);
    const PAR_SLICE_CHUNK: usize = 4096;
    let mut layer = start_layer_inclusive;
    loop {
        if layer > min_layer_inclusive {
            let outer_layer = layer - 1;
            let num_outer_blocks = 1usize << outer_layer;
            let outer_block_size = 1usize << (log_d - outer_layer);
            let q_len = outer_block_size >> 2;
            if num_outer_blocks >= n_threads || q_len <= PAR_SLICE_CHUNK {
                data.par_chunks_mut(outer_block_size).enumerate().for_each(
                    |(block_outer, chunk)| {
                        let t_o = ntt.twiddle(outer_layer, block_outer);
                        let t_ia = ntt.twiddle(layer, 2 * block_outer);
                        let t_ib = ntt.twiddle(layer, 2 * block_outer + 1);
                        let (h0, h1) = chunk.split_at_mut(2 * q_len);
                        let (q0, q1) = h0.split_at_mut(q_len);
                        let (q2, q3) = h1.split_at_mut(q_len);
                        transpose_butterfly_fused_2layer_slices(q0, q1, q2, q3, t_o, t_ia, t_ib);
                    },
                );
            } else {
                for block_outer in 0..num_outer_blocks {
                    let t_o = ntt.twiddle(outer_layer, block_outer);
                    let t_ia = ntt.twiddle(layer, 2 * block_outer);
                    let t_ib = ntt.twiddle(layer, 2 * block_outer + 1);
                    let chunk = &mut data
                        [block_outer * outer_block_size..(block_outer + 1) * outer_block_size];
                    let (h0, h1) = chunk.split_at_mut(2 * q_len);
                    let (q0, q1) = h0.split_at_mut(q_len);
                    let (q2, q3) = h1.split_at_mut(q_len);
                    q0.par_chunks_mut(PAR_SLICE_CHUNK)
                        .zip(q1.par_chunks_mut(PAR_SLICE_CHUNK))
                        .zip(q2.par_chunks_mut(PAR_SLICE_CHUNK))
                        .zip(q3.par_chunks_mut(PAR_SLICE_CHUNK))
                        .for_each(|(((c0, c1), c2), c3)| {
                            transpose_butterfly_fused_2layer_slices(
                                c0, c1, c2, c3, t_o, t_ia, t_ib,
                            );
                        });
                }
            }
            if layer < min_layer_inclusive + 2 {
                break;
            }
            layer -= 2;
        } else {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let bsh = block_size >> 1;
            if num_blocks >= n_threads || bsh <= PAR_SLICE_CHUNK {
                data.par_chunks_mut(block_size)
                    .enumerate()
                    .for_each(|(block, chunk)| {
                        let t = ntt.twiddle(layer, block);
                        let (top, bot) = chunk.split_at_mut(bsh);
                        transpose_butterfly_slice(top, bot, t);
                    });
            } else {
                for block in 0..num_blocks {
                    let t = ntt.twiddle(layer, block);
                    let chunk = &mut data[block * block_size..(block + 1) * block_size];
                    let (top, bot) = chunk.split_at_mut(bsh);
                    top.par_chunks_mut(PAR_SLICE_CHUNK)
                        .zip(bot.par_chunks_mut(PAR_SLICE_CHUNK))
                        .for_each(|(tc, bc)| {
                            transpose_butterfly_slice(tc, bc, t);
                        });
                }
            }
            if layer == min_layer_inclusive {
                break;
            }
            layer -= 1;
        }
    }
}

#[cfg(test)]
fn transpose_forward_ntt(ntt: &AdditiveNttF128, data: &mut [F128], log_d: usize) {
    debug_assert_eq!(data.len(), 1usize << log_d);
    debug_assert!(log_d <= ntt.log_domain_size());
    if log_d == 0 {
        return;
    }
    transpose_forward_ntt_layers_range(ntt, data, log_d, log_d - 1, 0);
}

pub(crate) fn induce_sumcheck_poly_via_ntt(
    log_msg_cols: usize,
    log_inv_rate: usize,
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    let n = 1usize << log_msg_cols;
    let log_block = log_msg_cols + log_inv_rate;
    let block_len = 1usize << log_block;
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);

    let eq = build_eq_table(v_challenges);
    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    let mut enforced_sum_unred = crate::field::F256Unreduced::ZERO;
    for i in 0..n_queries {
        let dot = crate::field::f128_slice::inner_product(&opened_rows[i], &eq);
        enforced_sum_unred ^= dot.mul_unreduced(alpha_pows[i]);
    }
    let enforced_sum = enforced_sum_unred.reduce();

    let mut coeffs = if log_block == 0 {
        let mut c = vec![F128::ZERO; block_len];
        for i in 0..n_queries {
            c[queries[i]] += alpha_pows[i];
        }
        c
    } else {
        let ntt = AdditiveNttF128::standard(log_block);
        transpose_forward_ntt_sparse_truncated(&ntt, queries, &alpha_pows, log_block, log_inv_rate)
    };
    coeffs.truncate(n);
    (coeffs, enforced_sum)
}

pub(crate) fn induce_sumcheck_poly_auto(
    log_msg_cols: usize,
    log_inv_rate: usize,
    sks_vks: &[F128],
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    let log_block = log_msg_cols + log_inv_rate;
    let use_ntt =
        log_msg_cols >= 12 && queries.len() > 4 * (1usize << log_inv_rate) * log_block.max(1);
    if use_ntt {
        induce_sumcheck_poly_via_ntt(
            log_msg_cols,
            log_inv_rate,
            opened_rows,
            v_challenges,
            queries,
            alpha,
        )
    } else {
        induce_sumcheck_poly(
            log_msg_cols,
            sks_vks,
            opened_rows,
            v_challenges,
            queries,
            alpha,
        )
    }
}

#[cfg(test)]
fn transpose_forward_ntt_sparse(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
) -> Vec<F128> {
    transpose_forward_ntt_sparse_truncated(ntt, positions, values, log_d, 0)
}

fn transpose_forward_ntt_sparse_truncated(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    min_layer: usize,
) -> Vec<F128> {
    use rayon::prelude::*;
    use std::collections::HashMap;
    let n = 1usize << log_d;
    let dense_layers = log_d.saturating_sub(min_layer);

    let k = if log_d >= 12 {
        8usize.min(dense_layers)
    } else {
        0
    };

    if k == 0 {
        let mut data = vec![F128::ZERO; n];
        for (&p, &v) in positions.iter().zip(values) {
            data[p] += v;
        }
        if log_d > min_layer {
            transpose_forward_ntt_layers_range(ntt, &mut data, log_d, log_d - 1, min_layer);
        }
        if min_layer > 0 {
            let keep_n = 1usize << (log_d - min_layer);
            let num_cosets = 1usize << min_layer;
            let (first, rest) = data.split_at_mut(keep_n);
            for c in 0..(num_cosets - 1) {
                let coset = &rest[c * keep_n..(c + 1) * keep_n];
                crate::field::f128_slice::xor_assign(first, coset);
            }
        }
        return data;
    }

    let wmask = (1usize << k) - 1;

    let mut windows: HashMap<usize, Vec<F128>> = HashMap::new();
    for (&p, &v) in positions.iter().zip(values) {
        let buf = windows
            .entry(p >> k)
            .or_insert_with(|| vec![F128::ZERO; 1 << k]);
        buf[p & wmask] += v;
    }

    let win_vec: Vec<(usize, Vec<F128>)> = windows.into_iter().collect();
    let processed: Vec<(usize, Vec<F128>)> = win_vec
        .into_par_iter()
        .map(|(w, mut buf)| {
            let mut s = 0usize;
            while s + 1 < k {
                let layer_inner = log_d - 1 - s;
                let layer_outer = log_d - 2 - s;
                let q_len = 1usize << s;
                let block_size = q_len << 2;
                let nblocks = (1usize << k) / block_size;
                for jb in 0..nblocks {
                    let blk_outer = (w << (k - s - 2)) + jb;
                    let t_ia = ntt.twiddle(layer_inner, 2 * blk_outer);
                    let t_ib = ntt.twiddle(layer_inner, 2 * blk_outer + 1);
                    let t_o = ntt.twiddle(layer_outer, blk_outer);
                    let base = jb * block_size;
                    for r in 0..q_len {
                        let x0 = buf[base + r];
                        let x1 = buf[base + r + q_len];
                        let x2 = buf[base + r + 2 * q_len];
                        let x3 = buf[base + r + 3 * q_len];
                        let s01 = x0 + x1;
                        let y1 = t_ia * s01 + x1;
                        let s23 = x2 + x3;
                        let y3 = t_ib * s23 + x3;
                        let z0 = s01 + s23;
                        let z1 = y1 + y3;
                        buf[base + r] = z0;
                        buf[base + r + q_len] = z1;
                        buf[base + r + 2 * q_len] = t_o * z0 + s23;
                        buf[base + r + 3 * q_len] = t_o * z1 + y3;
                    }
                }
                s += 2;
            }
            while s < k {
                let layer = log_d - 1 - s;
                let bsh = 1usize << s;
                let block_size = bsh << 1;
                let nblocks = (1usize << k) / block_size;
                for jb in 0..nblocks {
                    let t = ntt.twiddle(layer, (w << (k - s - 1)) + jb);
                    let base = jb * block_size;
                    let (top, bot) = buf[base..base + block_size].split_at_mut(bsh);
                    transpose_butterfly_slice(top, bot, t);
                }
                s += 1;
            }
            (w, buf)
        })
        .collect();

    let mut data = vec![F128::ZERO; n];
    for (w, buf) in processed {
        data[(w << k)..((w + 1) << k)].copy_from_slice(&buf);
    }

    if log_d - k > min_layer {
        transpose_forward_ntt_layers_range(ntt, &mut data, log_d, log_d - 1 - k, min_layer);
    }

    if min_layer > 0 {
        let keep_n = 1usize << (log_d - min_layer);
        let num_cosets = 1usize << min_layer;
        let (first, rest) = data.split_at_mut(keep_n);
        const PAR_XOR_CHUNK: usize = 8192;
        if keep_n >= PAR_XOR_CHUNK {
            first
                .par_chunks_mut(PAR_XOR_CHUNK)
                .enumerate()
                .for_each(|(ci, dst)| {
                    let off = ci * PAR_XOR_CHUNK;
                    let len = dst.len();
                    for c in 0..(num_cosets - 1) {
                        let src = &rest[c * keep_n + off..c * keep_n + off + len];
                        crate::field::f128_slice::xor_assign(dst, src);
                    }
                });
        } else {
            for c in 0..(num_cosets - 1) {
                let coset = &rest[c * keep_n..(c + 1) * keep_n];
                crate::field::f128_slice::xor_assign(first, coset);
            }
        }
    }
    data
}

pub(crate) struct LigeroWitness {
    pub mat: Vec<F128>,
    pub tree: Vec<Hash>,
    pub block_len: usize,
    pub num_interleaved: usize,
}

impl Drop for LigeroWitness {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.mat));
        crate::scratch::give_hash(std::mem::take(&mut self.tree));
    }
}

impl Drop for SumcheckProver {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.f));
        crate::scratch::give_f128(std::mem::take(&mut self.combined_basis));
        crate::scratch::give_f128(std::mem::take(&mut self.scratch_f));
        crate::scratch::give_f128(std::mem::take(&mut self.scratch_b));
    }
}

impl LigeroWitness {
    #[inline]
    pub fn row(&self, pos: usize) -> &[F128] {
        let start = pos * self.num_interleaved;
        &self.mat[start..start + self.num_interleaved]
    }

    #[inline]
    pub fn root(&self) -> Hash {
        self.tree[self.tree.len() - 1]
    }
}

pub(crate) fn ligero_commit(
    poly: &[F128],
    log_msg_cols: usize,
    log_num_interleaved: usize,
    log_inv_rate: usize,
    ntt: &AdditiveNttF128,
    kind: HashKind,
) -> LigeroWitness {
    let msg_cols = 1usize << log_msg_cols;
    let num_interleaved = 1usize << log_num_interleaved;
    let block_len = msg_cols << log_inv_rate;
    let log_block_len = log_msg_cols + log_inv_rate;
    assert_eq!(poly.len(), num_interleaved * msg_cols);
    assert!(log_block_len <= ntt.log_domain_size());

    let codeword_len = block_len * num_interleaved;
    let mut mat = crate::scratch::take_f128(codeword_len);
    if crate::gpu_commit::gpu_commit_enabled() {
        super::commit::replicate_message_fill(&mut mat, poly);

        if kind == HashKind::Blake3
            && let Some(tree) = crate::gpu_commit::try_gpu_commit_ntt_and_blake3_merkle(
                ntt,
                &mut mat,
                num_interleaved,
                log_inv_rate,
                block_len,
            )
        {
            return LigeroWitness {
                mat,
                tree,
                block_len,
                num_interleaved,
            };
        }

        if !crate::gpu_commit::try_gpu_interleaved_ntt_from_layer(
            ntt,
            &mut mat,
            num_interleaved,
            log_inv_rate,
        ) {
            ntt.forward_transform_interleaved_from_layer(&mut mat, num_interleaved, log_inv_rate);
        }
    } else {
        let leaf_size_bytes = num_interleaved * core::mem::size_of::<F128>();
        let mut tree: Vec<merkle::Hash> = crate::scratch::take_hash(2 * block_len - 1);
        let tree_base = tree.as_mut_ptr() as usize;
        let folded_depth = std::sync::atomic::AtomicUsize::new(usize::MAX);

        ntt.forward_transform_interleaved_from_replicated_msg_on_sub_done(
            &mut mat,
            poly,
            num_interleaved,
            log_inv_rate,
            &|range, sub_data| {
                debug_assert!(range.len().is_power_of_two() && range.start % range.len() == 0);
                let sub_bytes: &[u8] = unsafe {
                    core::slice::from_raw_parts(
                        sub_data.as_ptr() as *const u8,
                        std::mem::size_of_val(sub_data),
                    )
                };
                let leaves_out: &mut [merkle::Hash] = unsafe {
                    core::slice::from_raw_parts_mut(
                        (tree_base as *mut merkle::Hash).add(range.start),
                        range.len(),
                    )
                };
                merkle::hash_leaves_serial(sub_bytes, leaf_size_bytes, leaves_out, kind);

                let depth = range.len().trailing_zeros() as usize;
                folded_depth.fetch_min(depth, std::sync::atomic::Ordering::Relaxed);

                let mut prev_start = range.start;
                let mut prev_len = range.len();
                let mut prev_level_start = 0usize;
                let mut level_start = block_len;
                let mut level_len = block_len >> 1;
                for _ in 0..depth {
                    let parent_offset = (prev_start - prev_level_start) >> 1;
                    let parent_start = level_start + parent_offset;
                    let parent_len = prev_len >> 1;
                    let (children, parents) = unsafe {
                        let base = tree_base as *mut merkle::Hash;
                        (
                            core::slice::from_raw_parts(base.add(prev_start), prev_len),
                            core::slice::from_raw_parts_mut(base.add(parent_start), parent_len),
                        )
                    };
                    merkle::hash_pairs_level_serial(children, parents, kind);
                    prev_level_start = level_start;
                    prev_start = parent_start;
                    prev_len = parent_len;
                    level_start += level_len;
                    level_len >>= 1;
                }
            },
        );
        let depth_done = folded_depth.load(std::sync::atomic::Ordering::Relaxed);
        merkle::build_upper_levels(&mut tree, block_len, depth_done, kind);
        return LigeroWitness {
            mat,
            tree,
            block_len,
            num_interleaved,
        };
    }

    let leaf_size_bytes = num_interleaved * core::mem::size_of::<F128>();
    let data_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            mat.as_ptr() as *const u8,
            mat.len() * core::mem::size_of::<F128>(),
        )
    };
    debug_assert_eq!(data_bytes.len(), block_len * leaf_size_bytes);
    let tree = merkle::merkle_tree(data_bytes, block_len, kind);

    LigeroWitness {
        mat,
        tree,
        block_len,
        num_interleaved,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SumcheckMessage {
    pub u_0: F128,
    pub u_2: F128,
}

#[derive(Clone, Copy, Debug)]
struct RoundQuad {
    c: F128,
    b: F128,
    a: F128,
}

impl RoundQuad {
    #[inline]
    fn from_msg(msg: SumcheckMessage, t_r: F128) -> Self {
        Self {
            c: msg.u_0,
            b: t_r + msg.u_2,
            a: msg.u_2,
        }
    }
    #[inline]
    fn eval(&self, r: F128) -> F128 {
        self.c + r * self.b + r * r * self.a
    }
    #[inline]
    fn fold(p1: &Self, p2: &Self, alpha: F128) -> Self {
        Self {
            c: p1.c + alpha * p2.c,
            b: p1.b + alpha * p2.b,
            a: p1.a + alpha * p2.a,
        }
    }
}

fn round_msg_lsb_slice(fc: &[F128], bc: &[F128]) -> (F128, F128) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut u0_ll = _mm_setzero_si128();
        let mut u0_mid = _mm_setzero_si128();
        let mut u0_hh = _mm_setzero_si128();
        let mut u2_ll = _mm_setzero_si128();
        let mut u2_mid = _mm_setzero_si128();
        let mut u2_hh = _mm_setzero_si128();
        let f_ptr = fc.as_ptr() as *const __m128i;
        let b_ptr = bc.as_ptr() as *const __m128i;
        let mut k = 0usize;
        while k + 1 < fc.len() {
            let f0 = _mm_loadu_si128(f_ptr.add(k));
            let f1 = _mm_loadu_si128(f_ptr.add(k + 1));
            let b0 = _mm_loadu_si128(b_ptr.add(k));
            let b1 = _mm_loadu_si128(b_ptr.add(k + 1));
            crate::gf2_kernels::k0_f128_mul_acc_m128i(f0, b0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            crate::gf2_kernels::k0_f128_mul_acc_m128i(
                _mm_xor_si128(f0, f1),
                _mm_xor_si128(b0, b1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            k += 2;
        }
        let r0 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
        let r2 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
        (
            core::mem::transmute::<__m128i, F128>(r0),
            core::mem::transmute::<__m128i, F128>(r2),
        )
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let mut u0_ll = z;
        let mut u0_mid = z;
        let mut u0_hh = z;
        let mut u2_ll = z;
        let mut u2_mid = z;
        let mut u2_hh = z;
        let f_ptr = fc.as_ptr() as *const u64;
        let b_ptr = bc.as_ptr() as *const u64;
        let mut k = 0usize;
        while k + 1 < fc.len() {
            let idx = 2 * k;
            let f0 = vld1q_u64(f_ptr.add(idx));
            let f1 = vld1q_u64(f_ptr.add(idx + 2));
            let b0 = vld1q_u64(b_ptr.add(idx));
            let b1 = vld1q_u64(b_ptr.add(idx + 2));
            crate::gf2_kernels::k0_f128_mul_acc_neon(f0, b0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            crate::gf2_kernels::k0_f128_mul_acc_neon(
                veorq_u64(f0, f1),
                veorq_u64(b0, b1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            k += 2;
        }
        let r0 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u0_ll, u0_mid, u0_hh);
        let r2 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u2_ll, u2_mid, u2_hh);
        return (
            F128 {
                lo: vgetq_lane_u64::<0>(r0),
                hi: vgetq_lane_u64::<1>(r0),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(r2),
                hi: vgetq_lane_u64::<1>(r2),
            },
        );
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes"),
    )))]
    {
        let mut u0 = crate::field::F256Unreduced::ZERO;
        let mut u2 = crate::field::F256Unreduced::ZERO;
        let mut k = 0;
        while k + 1 < fc.len() {
            let f0 = fc[k];
            let f1 = fc[k + 1];
            let b0 = bc[k];
            let b1 = bc[k + 1];
            u0 ^= f0.mul_unreduced(b0);
            u2 ^= (f0 + f1).mul_unreduced(b0 + b1);
            k += 2;
        }
        (u0.reduce(), u2.reduce())
    }
}

fn round_msg_lsb(f: &[F128], b: &[F128]) -> SumcheckMessage {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);

    const PAR_THRESHOLD: usize = 8192;
    let half = n / 2;
    if half < PAR_THRESHOLD {
        let (u_0, u_2) = round_msg_lsb_slice(f, b);
        return SumcheckMessage { u_0, u_2 };
    }

    const CHUNK_PAIRS: usize = 8192;
    let (u_0, u_2) = f
        .par_chunks(2 * CHUNK_PAIRS)
        .zip(b.par_chunks(2 * CHUNK_PAIRS))
        .map(|(fc, bc)| round_msg_lsb_slice(fc, bc))
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    SumcheckMessage { u_0, u_2 }
}

fn round_msg_and_eval_lsb_slice(fc: &[F128], bc: &[F128]) -> (F128, F128, F128) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut u0_ll = _mm_setzero_si128();
        let mut u0_mid = _mm_setzero_si128();
        let mut u0_hh = _mm_setzero_si128();
        let mut u2_ll = _mm_setzero_si128();
        let mut u2_mid = _mm_setzero_si128();
        let mut u2_hh = _mm_setzero_si128();
        let mut y1_ll = _mm_setzero_si128();
        let mut y1_mid = _mm_setzero_si128();
        let mut y1_hh = _mm_setzero_si128();
        let f_ptr = fc.as_ptr() as *const __m128i;
        let b_ptr = bc.as_ptr() as *const __m128i;
        let mut k = 0usize;
        while k + 1 < fc.len() {
            let f0 = _mm_loadu_si128(f_ptr.add(k));
            let f1 = _mm_loadu_si128(f_ptr.add(k + 1));
            let b0 = _mm_loadu_si128(b_ptr.add(k));
            let b1 = _mm_loadu_si128(b_ptr.add(k + 1));
            crate::gf2_kernels::k0_f128_mul_acc_m128i(f0, b0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            crate::gf2_kernels::k0_f128_mul_acc_m128i(
                _mm_xor_si128(f0, f1),
                _mm_xor_si128(b0, b1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            crate::gf2_kernels::k0_f128_mul_acc_m128i(f1, b1, &mut y1_ll, &mut y1_mid, &mut y1_hh);
            k += 2;
        }
        let r0 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
        let r2 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
        let ry = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(y1_ll, y1_mid, y1_hh);
        (
            core::mem::transmute::<__m128i, F128>(r0),
            core::mem::transmute::<__m128i, F128>(r2),
            core::mem::transmute::<__m128i, F128>(ry),
        )
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let mut u0_ll = z;
        let mut u0_mid = z;
        let mut u0_hh = z;
        let mut u2_ll = z;
        let mut u2_mid = z;
        let mut u2_hh = z;
        let mut y1_ll = z;
        let mut y1_mid = z;
        let mut y1_hh = z;
        let f_ptr = fc.as_ptr() as *const u64;
        let b_ptr = bc.as_ptr() as *const u64;
        let mut k = 0usize;
        while k + 1 < fc.len() {
            let idx = 2 * k;
            let f0 = vld1q_u64(f_ptr.add(idx));
            let f1 = vld1q_u64(f_ptr.add(idx + 2));
            let b0 = vld1q_u64(b_ptr.add(idx));
            let b1 = vld1q_u64(b_ptr.add(idx + 2));
            crate::gf2_kernels::k0_f128_mul_acc_neon(f0, b0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            crate::gf2_kernels::k0_f128_mul_acc_neon(
                veorq_u64(f0, f1),
                veorq_u64(b0, b1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            crate::gf2_kernels::k0_f128_mul_acc_neon(f1, b1, &mut y1_ll, &mut y1_mid, &mut y1_hh);
            k += 2;
        }
        let r0 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u0_ll, u0_mid, u0_hh);
        let r2 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u2_ll, u2_mid, u2_hh);
        let ry = crate::gf2_kernels::k0_f128_reduce_3lane_neon(y1_ll, y1_mid, y1_hh);
        return (
            F128 {
                lo: vgetq_lane_u64::<0>(r0),
                hi: vgetq_lane_u64::<1>(r0),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(r2),
                hi: vgetq_lane_u64::<1>(r2),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(ry),
                hi: vgetq_lane_u64::<1>(ry),
            },
        );
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes"),
    )))]
    {
        let mut u0 = crate::field::F256Unreduced::ZERO;
        let mut u2 = crate::field::F256Unreduced::ZERO;
        let mut y1 = crate::field::F256Unreduced::ZERO;
        let mut k = 0;
        while k + 1 < fc.len() {
            let f0 = fc[k];
            let f1 = fc[k + 1];
            let b0 = bc[k];
            let b1 = bc[k + 1];
            u0 ^= f0.mul_unreduced(b0);
            u2 ^= (f0 + f1).mul_unreduced(b0 + b1);
            y1 ^= f1.mul_unreduced(b1);
            k += 2;
        }
        (u0.reduce(), u2.reduce(), y1.reduce())
    }
}

fn round_msg_and_eval_lsb(f: &[F128], b: &[F128]) -> (SumcheckMessage, F128) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);

    const PAR_THRESHOLD: usize = 8192;
    let half = n / 2;
    if half < PAR_THRESHOLD {
        let (u_0, u_2, y_1) = round_msg_and_eval_lsb_slice(f, b);
        return (SumcheckMessage { u_0, u_2 }, u_0 + y_1);
    }

    const CHUNK_PAIRS: usize = 8192;
    let (u_0, u_2, y_1) = f
        .par_chunks(2 * CHUNK_PAIRS)
        .zip(b.par_chunks(2 * CHUNK_PAIRS))
        .map(|(fc, bc)| round_msg_and_eval_lsb_slice(fc, bc))
        .reduce(
            || (F128::ZERO, F128::ZERO, F128::ZERO),
            |(a0, a2, ay), (b0, b2, by)| (a0 + b0, a2 + b2, ay + by),
        );
    (SumcheckMessage { u_0, u_2 }, u_0 + y_1)
}

#[cfg(test)]
fn partial_eval_lsb_one(evals: &mut Vec<F128>, r: F128) {
    use rayon::prelude::*;
    let n = evals.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;

    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        for j in 0..half {
            let v0 = evals[2 * j];
            let v1 = evals[2 * j + 1];
            evals[j] = v0 + (v0 + v1) * r;
        }
        evals.truncate(half);
        return;
    }

    let folded: Vec<F128> = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(|j| {
            let v0 = evals[2 * j];
            let v1 = evals[2 * j + 1];
            v0 + (v0 + v1) * r
        })
        .collect();
    *evals = folded;
}

fn fold_and_msg_lsb_into(
    f: &[F128],
    b: &[F128],
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
) -> SumcheckMessage {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    debug_assert_eq!(nf.len(), half);
    debug_assert_eq!(nb.len(), half);

    const PAR_THRESHOLD: usize = 8192;
    if half < PAR_THRESHOLD {
        let (u_0, u_2) = crate::field::f128_slice::fold_pairs_and_msg(f, b, 0, nf, nb, r);
        return SumcheckMessage { u_0, u_2 };
    }

    const CHUNK: usize = 16384;
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        let stream_f = half >= (1 << 20);
        let stream_b = half >= (1 << 19);
        let (u_0, u_2) = nf
            .par_chunks_mut(CHUNK)
            .zip(nb.par_chunks_mut(CHUNK))
            .enumerate()
            .map(|(ci, (fc, bc))| unsafe {
                use core::arch::x86_64::*;
                let base = ci * CHUNK;
                let f_ptr = f.as_ptr() as *const u64;
                let b_ptr = b.as_ptr() as *const u64;
                let fc_ptr = fc.as_mut_ptr() as *mut u64;
                let bc_ptr = bc.as_mut_ptr() as *mut u64;
                let len = fc.len();
                let vr = _mm_set_epi64x(r.hi as i64, r.lo as i64);
                let (vr0, vr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vr);
                let lanes = len & !3;
                let (mut u0_ll, mut u0_mid, mut u0_hh) = (
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                );
                let (mut u2_ll, mut u2_mid, mut u2_hh) = (
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                );
                let mut t = 0usize;
                macro_rules! fold_msg_step4_x86 {
                    ($store_f:ident, $store_b:ident) => {{
                        while t < lanes {
                            let s0 = 4 * (base + t);
                            let d0 = 2 * t;
                            let fe0 = _mm_loadu_si128(f_ptr.add(s0) as *const __m128i);
                            let fo0 = _mm_loadu_si128(f_ptr.add(s0 + 2) as *const __m128i);
                            let fe1 = _mm_loadu_si128(f_ptr.add(s0 + 4) as *const __m128i);
                            let fo1 = _mm_loadu_si128(f_ptr.add(s0 + 6) as *const __m128i);
                            let (fp0, fp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                _mm_xor_si128(fe0, fo0),
                                _mm_xor_si128(fe1, fo1),
                                vr0,
                                vr1,
                                vr0,
                                vr1,
                            );
                            let nf0 = _mm_xor_si128(fe0, fp0);
                            let nf1 = _mm_xor_si128(fe1, fp1);
                            $store_f(fc_ptr.add(d0) as *mut __m128i, nf0);
                            $store_f(fc_ptr.add(d0 + 2) as *mut __m128i, nf1);
                            let be0 = _mm_loadu_si128(b_ptr.add(s0) as *const __m128i);
                            let bo0 = _mm_loadu_si128(b_ptr.add(s0 + 2) as *const __m128i);
                            let be1 = _mm_loadu_si128(b_ptr.add(s0 + 4) as *const __m128i);
                            let bo1 = _mm_loadu_si128(b_ptr.add(s0 + 6) as *const __m128i);
                            let (bp0, bp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                _mm_xor_si128(be0, bo0),
                                _mm_xor_si128(be1, bo1),
                                vr0,
                                vr1,
                                vr0,
                                vr1,
                            );
                            let nb0 = _mm_xor_si128(be0, bp0);
                            let nb1 = _mm_xor_si128(be1, bp1);
                            $store_b(bc_ptr.add(d0) as *mut __m128i, nb0);
                            $store_b(bc_ptr.add(d0 + 2) as *mut __m128i, nb1);
                            crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                nf0,
                                nb0,
                                &mut u0_ll,
                                &mut u0_mid,
                                &mut u0_hh,
                            );
                            crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                _mm_xor_si128(nf0, nf1),
                                _mm_xor_si128(nb0, nb1),
                                &mut u2_ll,
                                &mut u2_mid,
                                &mut u2_hh,
                            );

                            let fe2 = _mm_loadu_si128(f_ptr.add(s0 + 8) as *const __m128i);
                            let fo2 = _mm_loadu_si128(f_ptr.add(s0 + 10) as *const __m128i);
                            let fe3 = _mm_loadu_si128(f_ptr.add(s0 + 12) as *const __m128i);
                            let fo3 = _mm_loadu_si128(f_ptr.add(s0 + 14) as *const __m128i);
                            let (fp2, fp3) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                _mm_xor_si128(fe2, fo2),
                                _mm_xor_si128(fe3, fo3),
                                vr0,
                                vr1,
                                vr0,
                                vr1,
                            );
                            let nf2 = _mm_xor_si128(fe2, fp2);
                            let nf3 = _mm_xor_si128(fe3, fp3);
                            $store_f(fc_ptr.add(d0 + 4) as *mut __m128i, nf2);
                            $store_f(fc_ptr.add(d0 + 6) as *mut __m128i, nf3);
                            let be2 = _mm_loadu_si128(b_ptr.add(s0 + 8) as *const __m128i);
                            let bo2 = _mm_loadu_si128(b_ptr.add(s0 + 10) as *const __m128i);
                            let be3 = _mm_loadu_si128(b_ptr.add(s0 + 12) as *const __m128i);
                            let bo3 = _mm_loadu_si128(b_ptr.add(s0 + 14) as *const __m128i);
                            let (bp2, bp3) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                _mm_xor_si128(be2, bo2),
                                _mm_xor_si128(be3, bo3),
                                vr0,
                                vr1,
                                vr0,
                                vr1,
                            );
                            let nb2 = _mm_xor_si128(be2, bp2);
                            let nb3 = _mm_xor_si128(be3, bp3);
                            $store_b(bc_ptr.add(d0 + 4) as *mut __m128i, nb2);
                            $store_b(bc_ptr.add(d0 + 6) as *mut __m128i, nb3);
                            crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                nf2,
                                nb2,
                                &mut u0_ll,
                                &mut u0_mid,
                                &mut u0_hh,
                            );
                            crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                _mm_xor_si128(nf2, nf3),
                                _mm_xor_si128(nb2, nb3),
                                &mut u2_ll,
                                &mut u2_mid,
                                &mut u2_hh,
                            );
                            t += 4;
                        }
                        while t < (len & !1) {
                            let s = 4 * (base + t);
                            let fe0 = _mm_loadu_si128(f_ptr.add(s) as *const __m128i);
                            let fo0 = _mm_loadu_si128(f_ptr.add(s + 2) as *const __m128i);
                            let fe1 = _mm_loadu_si128(f_ptr.add(s + 4) as *const __m128i);
                            let fo1 = _mm_loadu_si128(f_ptr.add(s + 6) as *const __m128i);
                            let (fp0, fp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                _mm_xor_si128(fe0, fo0),
                                _mm_xor_si128(fe1, fo1),
                                vr0,
                                vr1,
                                vr0,
                                vr1,
                            );
                            let nf0 = _mm_xor_si128(fe0, fp0);
                            let nf1 = _mm_xor_si128(fe1, fp1);
                            let d = 2 * t;
                            $store_f(fc_ptr.add(d) as *mut __m128i, nf0);
                            $store_f(fc_ptr.add(d + 2) as *mut __m128i, nf1);
                            let be0 = _mm_loadu_si128(b_ptr.add(s) as *const __m128i);
                            let bo0 = _mm_loadu_si128(b_ptr.add(s + 2) as *const __m128i);
                            let be1 = _mm_loadu_si128(b_ptr.add(s + 4) as *const __m128i);
                            let bo1 = _mm_loadu_si128(b_ptr.add(s + 6) as *const __m128i);
                            let (bp0, bp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                _mm_xor_si128(be0, bo0),
                                _mm_xor_si128(be1, bo1),
                                vr0,
                                vr1,
                                vr0,
                                vr1,
                            );
                            let nb0 = _mm_xor_si128(be0, bp0);
                            let nb1 = _mm_xor_si128(be1, bp1);
                            $store_b(bc_ptr.add(d) as *mut __m128i, nb0);
                            $store_b(bc_ptr.add(d + 2) as *mut __m128i, nb1);
                            crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                nf0,
                                nb0,
                                &mut u0_ll,
                                &mut u0_mid,
                                &mut u0_hh,
                            );
                            crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                _mm_xor_si128(nf0, nf1),
                                _mm_xor_si128(nb0, nb1),
                                &mut u2_ll,
                                &mut u2_mid,
                                &mut u2_hh,
                            );
                            t += 2;
                        }
                    }};
                }
                if stream_f {
                    fold_msg_step4_x86!(_mm_stream_si128, _mm_stream_si128);
                    _mm_sfence();
                } else if stream_b {
                    fold_msg_step4_x86!(_mm_storeu_si128, _mm_stream_si128);
                    _mm_sfence();
                } else {
                    fold_msg_step4_x86!(_mm_storeu_si128, _mm_storeu_si128);
                }
                let u0 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
                let u2 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
                (
                    core::mem::transmute::<__m128i, F128>(u0),
                    core::mem::transmute::<__m128i, F128>(u2),
                )
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
            );
        return SumcheckMessage { u_0, u_2 };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        let (u_0, u_2) = nf
            .par_chunks_mut(CHUNK)
            .zip(nb.par_chunks_mut(CHUNK))
            .enumerate()
            .map(|(ci, (fc, bc))| unsafe {
                use core::arch::aarch64::*;
                let base = ci * CHUNK;
                let f_ptr = f.as_ptr() as *const u64;
                let b_ptr = b.as_ptr() as *const u64;
                let fc_ptr = fc.as_mut_ptr() as *mut u64;
                let bc_ptr = bc.as_mut_ptr() as *mut u64;
                let len = fc.len();
                let z = vdupq_n_u64(0);
                let vr = vld1q_u64(&r as *const F128 as *const u64);
                let (vr0, vr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vr);
                let lanes4 = len & !3;
                let lanes2 = len & !1;
                let (mut u0_ll, mut u0_mid, mut u0_hh) = (z, z, z);
                let (mut u2_ll, mut u2_mid, mut u2_hh) = (z, z, z);
                let mut t = 0usize;
                while t < lanes4 {
                    let s = 4 * (base + t);
                    let fe0 = vld1q_u64(f_ptr.add(s));
                    let fo0 = vld1q_u64(f_ptr.add(s + 2));
                    let fe1 = vld1q_u64(f_ptr.add(s + 4));
                    let fo1 = vld1q_u64(f_ptr.add(s + 6));
                    let fe2 = vld1q_u64(f_ptr.add(s + 8));
                    let fo2 = vld1q_u64(f_ptr.add(s + 10));
                    let fe3 = vld1q_u64(f_ptr.add(s + 12));
                    let fo3 = vld1q_u64(f_ptr.add(s + 14));
                    let (fp0, fp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        veorq_u64(fe0, fo0),
                        veorq_u64(fe1, fo1),
                        vr0,
                        vr1,
                        vr0,
                        vr1,
                    );
                    let (fp2, fp3) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        veorq_u64(fe2, fo2),
                        veorq_u64(fe3, fo3),
                        vr0,
                        vr1,
                        vr0,
                        vr1,
                    );
                    let nf0 = veorq_u64(fe0, fp0);
                    let nf1 = veorq_u64(fe1, fp1);
                    let nf2 = veorq_u64(fe2, fp2);
                    let nf3 = veorq_u64(fe3, fp3);
                    let d = 2 * t;
                    vst1q_u64(fc_ptr.add(d), nf0);
                    vst1q_u64(fc_ptr.add(d + 2), nf1);
                    vst1q_u64(fc_ptr.add(d + 4), nf2);
                    vst1q_u64(fc_ptr.add(d + 6), nf3);
                    let be0 = vld1q_u64(b_ptr.add(s));
                    let bo0 = vld1q_u64(b_ptr.add(s + 2));
                    let be1 = vld1q_u64(b_ptr.add(s + 4));
                    let bo1 = vld1q_u64(b_ptr.add(s + 6));
                    let be2 = vld1q_u64(b_ptr.add(s + 8));
                    let bo2 = vld1q_u64(b_ptr.add(s + 10));
                    let be3 = vld1q_u64(b_ptr.add(s + 12));
                    let bo3 = vld1q_u64(b_ptr.add(s + 14));
                    let (bp0, bp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        veorq_u64(be0, bo0),
                        veorq_u64(be1, bo1),
                        vr0,
                        vr1,
                        vr0,
                        vr1,
                    );
                    let (bp2, bp3) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        veorq_u64(be2, bo2),
                        veorq_u64(be3, bo3),
                        vr0,
                        vr1,
                        vr0,
                        vr1,
                    );
                    let nb0 = veorq_u64(be0, bp0);
                    let nb1 = veorq_u64(be1, bp1);
                    let nb2 = veorq_u64(be2, bp2);
                    let nb3 = veorq_u64(be3, bp3);
                    vst1q_u64(bc_ptr.add(d), nb0);
                    vst1q_u64(bc_ptr.add(d + 2), nb1);
                    vst1q_u64(bc_ptr.add(d + 4), nb2);
                    vst1q_u64(bc_ptr.add(d + 6), nb3);
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        nf0,
                        nb0,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        nf2,
                        nb2,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        veorq_u64(nf0, nf1),
                        veorq_u64(nb0, nb1),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        veorq_u64(nf2, nf3),
                        veorq_u64(nb2, nb3),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                    t += 4;
                }
                while t < lanes2 {
                    let s = 4 * (base + t);
                    let fe0 = vld1q_u64(f_ptr.add(s));
                    let fo0 = vld1q_u64(f_ptr.add(s + 2));
                    let fe1 = vld1q_u64(f_ptr.add(s + 4));
                    let fo1 = vld1q_u64(f_ptr.add(s + 6));
                    let (fp0, fp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        veorq_u64(fe0, fo0),
                        veorq_u64(fe1, fo1),
                        vr0,
                        vr1,
                        vr0,
                        vr1,
                    );
                    let nf0 = veorq_u64(fe0, fp0);
                    let nf1 = veorq_u64(fe1, fp1);
                    let d = 2 * t;
                    vst1q_u64(fc_ptr.add(d), nf0);
                    vst1q_u64(fc_ptr.add(d + 2), nf1);
                    let be0 = vld1q_u64(b_ptr.add(s));
                    let bo0 = vld1q_u64(b_ptr.add(s + 2));
                    let be1 = vld1q_u64(b_ptr.add(s + 4));
                    let bo1 = vld1q_u64(b_ptr.add(s + 6));
                    let (bp0, bp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        veorq_u64(be0, bo0),
                        veorq_u64(be1, bo1),
                        vr0,
                        vr1,
                        vr0,
                        vr1,
                    );
                    let nb0 = veorq_u64(be0, bp0);
                    let nb1 = veorq_u64(be1, bp1);
                    vst1q_u64(bc_ptr.add(d), nb0);
                    vst1q_u64(bc_ptr.add(d + 2), nb1);
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        nf0,
                        nb0,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        veorq_u64(nf0, nf1),
                        veorq_u64(nb0, nb1),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                    t += 2;
                }
                let u0 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u0_ll, u0_mid, u0_hh);
                let u2 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u2_ll, u2_mid, u2_hh);
                (
                    core::mem::transmute::<uint64x2_t, F128>(u0),
                    core::mem::transmute::<uint64x2_t, F128>(u2),
                )
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
            );
        return SumcheckMessage { u_0, u_2 };
    }

    #[allow(unreachable_code)]
    let (u_0, u_2) = nf
        .par_chunks_mut(CHUNK)
        .zip(nb.par_chunks_mut(CHUNK))
        .enumerate()
        .map(|(ci, (fc, bc))| {
            let base = ci * CHUNK;
            crate::field::f128_slice::fold_pairs_and_msg(f, b, base, fc, bc, r)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
        );
    #[allow(unreachable_code)]
    SumcheckMessage { u_0, u_2 }
}

pub struct SumcheckProver {
    f: Vec<F128>,

    combined_basis: Vec<F128>,
    scratch_f: Vec<F128>,
    scratch_b: Vec<F128>,
    t_r: F128,
    transcript: Vec<SumcheckMessage>,
    pending_glue: Option<(Vec<F128>, F128)>,
}

impl SumcheckProver {
    pub fn new(f: Vec<F128>, b1: Vec<F128>, h1: F128) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let mut inst = Self {
            f,
            combined_basis: b1,
            scratch_f: Vec::new(),
            scratch_b: Vec::new(),
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
        };
        let msg = round_msg_lsb(&inst.f, &inst.combined_basis);
        inst.transcript.push(msg);
        (inst, msg)
    }

    pub fn new_with_first_msg(
        f: Vec<F128>,
        b1: Vec<F128>,
        h1: F128,
        first_msg: SumcheckMessage,
    ) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let mut inst = Self {
            f,
            combined_basis: b1,
            scratch_f: Vec::new(),
            scratch_b: Vec::new(),
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
        };
        inst.transcript.push(first_msg);
        (inst, first_msg)
    }

    pub(crate) fn new_after_s0(
        f: Vec<F128>,
        b1: Vec<F128>,
        h1: F128,
        prefix_transcript: Vec<SumcheckMessage>,
    ) -> Self {
        assert_eq!(f.len(), b1.len());
        Self {
            f,
            combined_basis: b1,
            scratch_f: Vec::new(),
            scratch_b: Vec::new(),
            t_r: h1,
            transcript: prefix_transcript,
            pending_glue: None,
        }
    }

    pub fn fold(&mut self, r: F128) -> SumcheckMessage {
        let half = self.f.len() / 2;
        if self.scratch_f.capacity() < half {
            crate::scratch::give_f128(std::mem::replace(
                &mut self.scratch_f,
                crate::scratch::take_f128(half),
            ));
        } else {
            unsafe {
                self.scratch_f.set_len(half);
            }
        }
        if self.scratch_b.capacity() < half {
            crate::scratch::give_f128(std::mem::replace(
                &mut self.scratch_b,
                crate::scratch::take_f128(half),
            ));
        } else {
            unsafe {
                self.scratch_b.set_len(half);
            }
        }
        let msg = fold_and_msg_lsb_into(
            &self.f,
            &self.combined_basis,
            &mut self.scratch_f[..half],
            &mut self.scratch_b[..half],
            r,
        );
        std::mem::swap(&mut self.f, &mut self.scratch_f);
        std::mem::swap(&mut self.combined_basis, &mut self.scratch_b);
        self.f.truncate(half);
        self.combined_basis.truncate(half);
        self.transcript.push(msg);
        msg
    }

    pub fn introduce_new(&mut self, b_new: Vec<F128>, h_new: F128) -> SumcheckMessage {
        assert_eq!(b_new.len(), self.f.len());
        let msg = round_msg_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        msg
    }

    pub fn introduce_new_with_eval(&mut self, b_new: Vec<F128>) -> (SumcheckMessage, F128) {
        assert_eq!(b_new.len(), self.f.len());
        let (msg, h_new) = round_msg_and_eval_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        (msg, h_new)
    }

    pub fn glue(&mut self, alpha: F128) {
        use rayon::prelude::*;
        let (b_new, h_new) = self
            .pending_glue
            .take()
            .expect("glue without introduce_new");
        assert_eq!(b_new.len(), self.combined_basis.len());
        const PAR_THRESHOLD: usize = 8192;
        if self.combined_basis.len() < PAR_THRESHOLD {
            crate::field::f128_slice::scalar_mul_add_assign(
                &mut self.combined_basis,
                &b_new,
                alpha,
            );
        } else {
            const CHUNK: usize = 8192;
            self.combined_basis
                .par_chunks_mut(CHUNK)
                .zip(b_new.par_chunks(CHUNK))
                .for_each(|(dst, src)| {
                    crate::field::f128_slice::scalar_mul_add_assign(dst, src, alpha);
                });
        }
        self.t_r += alpha * h_new;
    }

    pub fn f(&self) -> &[F128] {
        &self.f
    }

    pub fn transcript(&self) -> &[SumcheckMessage] {
        &self.transcript
    }
}

fn sample_distinct_queries<Ch: Challenger>(
    challenger: &mut Ch,
    block_len: usize,
    count: usize,
) -> Vec<usize> {
    assert!(
        count <= block_len,
        "sample_distinct_queries: count ({count}) > block_len ({block_len}) — config is too thin for this query count"
    );
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let v = challenger.sample_f128();
        let q = (v.lo as usize) % block_len;
        if seen.insert(q) {
            out.push(q);
        }
    }
    out.sort_unstable();
    out
}

fn merkle_multi_proof_for(tree: &[Hash], block_len: usize, queries: &[usize]) -> Vec<Hash> {
    merkle::merkle_multi_proof(tree, block_len, queries)
}

pub fn recursive_prover<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> LigeritoProof {
    let trace = std::env::var("LIGERITO_TRACE").is_ok();
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }
    let t_total = std::time::Instant::now();
    let mut t_commits = std::time::Duration::ZERO;
    let t_induce = std::time::Duration::ZERO;
    let t_sumcheck = std::time::Duration::ZERO;
    let t_opens = std::time::Duration::ZERO;
    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;

    assert_eq!(poly.len(), 1usize << log_n);
    assert_eq!(eval_point.len(), log_n);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(
        config.log_inv_rates.len(),
        r + 1,
        "log_inv_rates must have R+1 entries"
    );
    assert!(r >= 1, "recursive_steps must be ≥ 1");

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate_0);
    let t = std::time::Instant::now();
    let wtns_0 = ligero_commit(
        poly,
        log_msg_cols_0,
        initial_k,
        log_inv_rate_0,
        &ntt_0,
        config.merkle_hash,
    );
    let t_l0 = t.elapsed();
    t_commits += t_l0;
    tlog!("  [ligerito]   L0 commit: {:.2?}", t_l0);
    recursive_prover_inner(
        config,
        poly,
        wtns_0,
        eval_point,
        claimed_value,
        challenger,
        t_total,
        t_commits,
        t_induce,
        t_sumcheck,
        t_opens,
        trace,
    )
}

pub fn recursive_prover_with_l0<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    l0_codeword: Vec<F128>,
    l0_tree: Vec<Hash>,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> LigeritoProof {
    let trace = std::env::var("LIGERITO_TRACE").is_ok();
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }
    let t_total = std::time::Instant::now();
    let t_commits = std::time::Duration::ZERO;
    let t_induce = std::time::Duration::ZERO;
    let t_sumcheck = std::time::Duration::ZERO;
    let t_opens = std::time::Duration::ZERO;

    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;

    assert_eq!(poly.len(), 1usize << log_n);
    assert_eq!(eval_point.len(), log_n);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(r >= 1, "recursive_steps must be ≥ 1");

    let block_len = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved = 1usize << initial_k;
    let _ = r;
    assert_eq!(
        l0_codeword.len(),
        block_len * num_interleaved,
        "external L0 codeword wrong size"
    );
    assert_eq!(
        l0_tree.len(),
        2 * block_len - 1,
        "external L0 tree wrong size"
    );

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    let wtns_0 = LigeroWitness {
        mat: l0_codeword,
        tree: l0_tree,
        block_len,
        num_interleaved,
    };
    tlog!("  [ligerito]   L0 commit: REUSED (skipped)");

    recursive_prover_inner(
        config,
        poly,
        wtns_0,
        eval_point,
        claimed_value,
        challenger,
        t_total,
        t_commits,
        t_induce,
        t_sumcheck,
        t_opens,
        trace,
    )
}

pub fn recursive_prover_with_basis<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        None,
        None,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn recursive_prover_with_basis_precomputed_round0<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        None,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_prover_with_basis_s0<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    s0_claims: Vec<super::ring_switch::S0Factors>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        Some(s0_claims),
        challenger,
    )
}

fn fold_s0_factors_and_message(
    claims: &mut [super::ring_switch::S0Factors],
    round: usize,
    r: F128,
) -> SumcheckMessage {
    let active_banks = 1usize << (5 - round);
    let mut u0 = crate::field::F256Unreduced::ZERO;
    let mut u2 = crate::field::F256Unreduced::ZERO;

    for claim in claims.iter_mut() {
        for b in 0..active_banks {
            let src0 = (2 * b) * 128;
            let src1 = (2 * b + 1) * 128;
            let dst = b * 128;
            for t in 0..128 {
                let a0 = claim.a_state[src0 + t];
                let a1 = claim.a_state[src1 + t];
                claim.a_state[dst + t] = a0 + (a0 + a1) * r;

                let w0 = claim.w_state[src0 + t];
                let w1 = claim.w_state[src1 + t];
                claim.w_state[dst + t] = w0 + (w0 + w1) * r;
            }
        }

        for pair in 0..(active_banks / 2) {
            let a0 = &claim.a_state[(2 * pair) * 128..(2 * pair + 1) * 128];
            let a1 = &claim.a_state[(2 * pair + 1) * 128..(2 * pair + 2) * 128];
            let w0 = &claim.w_state[(2 * pair) * 128..(2 * pair + 1) * 128];
            let w1 = &claim.w_state[(2 * pair + 1) * 128..(2 * pair + 2) * 128];
            for t in 0..128 {
                u0 ^= a0[t].mul_unreduced(w0[t]);
                u2 ^= (a0[t] + a1[t]).mul_unreduced(w0[t] + w1[t]);
            }
        }
    }

    SumcheckMessage {
        u_0: u0.reduce(),
        u_2: u2.reduce(),
    }
}

fn s0_final_generators(
    claim: &super::ring_switch::S0Factors,
    r_lane_fold: &[F128; 6],
) -> [F128; 128] {
    let r5 = r_lane_fold[5];
    let mut generators = [F128::ZERO; 128];
    for t in 0..128 {
        let w0 = claim.w_state[t];
        let w1 = claim.w_state[128 + t];
        generators[t] = w0 + (w0 + w1) * r5;
    }
    generators
}

fn materialize_s0(
    packed_witness: &[F128],
    ordinary_b: &[F128],
    claims: &[super::ring_switch::S0Factors],
    r_lane_fold: &[F128; 6],
) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    assert!(packed_witness.len() >= 128 && packed_witness.len().is_power_of_two());
    assert!(ordinary_b.is_empty() || ordinary_b.len() == packed_witness.len());
    let has_ordinary = !ordinary_b.is_empty();
    let out_len = packed_witness.len() >> 6;
    assert!(out_len >= 2);

    let w_hi4: [F128; 16] = crate::zerocheck::univariate_skip::build_eq(&r_lane_fold[2..])
        .try_into()
        .expect("4 fold challenges -> 16 weights");
    let r0 = r_lane_fold[0];
    let r1 = r_lane_fold[1];

    if claims.is_empty() {
        let mut folded_f = crate::scratch::take_f128(out_len);
        let mut folded_b = crate::scratch::take_f128(out_len);
        crate::field::f128_slice::fold64_banked_nested(
            packed_witness,
            &mut folded_f,
            &w_hi4,
            r0,
            r1,
        );
        if has_ordinary {
            crate::field::f128_slice::fold64_banked_nested(
                ordinary_b,
                &mut folded_b,
                &w_hi4,
                r0,
                r1,
            );
        } else {
            folded_b.fill(F128::ZERO);
        }
        let msg = round_msg_lsb(&folded_f, &folded_b);
        return (folded_f, folded_b, msg);
    }

    let block_len = claims[0].eq_lo.len();
    let num_blocks = claims[0].eq_hi.len();
    assert_eq!(block_len * num_blocks, out_len);
    assert!(block_len >= 2 && block_len.is_multiple_of(2));
    for claim in &claims[1..] {
        assert_eq!(claim.eq_lo.len(), block_len);
        assert_eq!(claim.eq_hi.len(), num_blocks);
    }

    let direct_tables: Vec<Vec<F128>> = claims
        .iter()
        .map(|claim| {
            let gens = s0_final_generators(claim, r_lane_fold);
            super::ring_switch::build_s0_table_from_generators(&gens)
        })
        .collect();

    let pair_factored =
        if !has_ordinary && claims.len() == 2 && block_len >= 256 && block_len.is_multiple_of(4) {
            super::ring_switch::prepare_pair_factored_eq_lo(&claims[0].eq_lo).zip(
                super::ring_switch::prepare_pair_factored_eq_lo(&claims[1].eq_lo),
            )
        } else {
            None
        };

    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);

    let (u_0, u_2) = folded_f
        .par_chunks_mut(block_len)
        .zip(folded_b.par_chunks_mut(block_len))
        .enumerate()
        .map(|(i_hi, (f_out, b_out))| {
            let w_base = (i_hi * block_len) << 6;
            crate::field::f128_slice::fold64_banked_nested(
                &packed_witness[w_base..w_base + (block_len << 6)],
                f_out,
                &w_hi4,
                r0,
                r1,
            );
            if let Some(((ref u0_half, r0_0), (ref u1_half, r0_1))) = pair_factored {
                return super::ring_switch::combine_two_deferred_block_and_round0_pair_factored(
                    b_out,
                    f_out,
                    u0_half,
                    r0_0,
                    claims[0].eq_hi[i_hi],
                    &direct_tables[0],
                    u1_half,
                    r0_1,
                    claims[1].eq_hi[i_hi],
                    &direct_tables[1],
                );
            }
            if !has_ordinary && claims.len() == 2 && block_len.is_multiple_of(4) {
                return super::ring_switch::combine_two_deferred_block_and_round0(
                    b_out,
                    f_out,
                    &claims[0].eq_lo,
                    claims[0].eq_hi[i_hi],
                    &direct_tables[0],
                    &claims[1].eq_lo,
                    claims[1].eq_hi[i_hi],
                    &direct_tables[1],
                );
            }
            if has_ordinary {
                crate::field::f128_slice::fold64_banked_nested(
                    &ordinary_b[w_base..w_base + (block_len << 6)],
                    b_out,
                    &w_hi4,
                    r0,
                    r1,
                );
            } else {
                b_out.fill(F128::ZERO);
            }

            let mut u0 = crate::field::F256Unreduced::ZERO;
            let mut u2 = crate::field::F256Unreduced::ZERO;
            for t in 0..(block_len / 2) {
                let f0 = f_out[2 * t];
                let f1 = f_out[2 * t + 1];
                let mut b0 = b_out[2 * t];
                let mut b1 = b_out[2 * t + 1];
                for (claim, table) in claims.iter().zip(direct_tables.iter()) {
                    let e_hi = claim.eq_hi[i_hi];
                    b0 += super::ring_switch::fold_one_slot(claim.eq_lo[2 * t] * e_hi, table);
                    b1 += super::ring_switch::fold_one_slot(claim.eq_lo[2 * t + 1] * e_hi, table);
                }
                b_out[2 * t] = b0;
                b_out[2 * t + 1] = b1;
                u0 ^= f0.mul_unreduced(b0);
                u2 ^= (f0 + f1).mul_unreduced(b0 + b1);
            }
            (u0.reduce(), u2.reduce())
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
        );

    (folded_f, folded_b, SumcheckMessage { u_0, u_2 })
}

#[allow(clippy::too_many_arguments)]
fn recursive_prover_with_basis_impl<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    first_msg: Option<SumcheckMessage>,
    s0_claims: Option<Vec<super::ring_switch::S0Factors>>,
    challenger: &mut Ch,
) -> LigeritoProof {
    let log_n = packed_witness.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    let has_s0 = s0_claims.as_ref().is_some_and(|claims| !claims.is_empty());

    assert_eq!(packed_witness.len(), 1usize << log_n);
    assert!(
        b_initial.len() == (1usize << log_n) || (has_s0 && b_initial.is_empty()),
        "b_initial must have length 2^log_n or be empty when s0 is active"
    );
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(r >= 1);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    assert_eq!(l0_codeword.len(), block_len_0 * num_interleaved_0);
    assert_eq!(l0_tree.len(), 2 * block_len_0 - 1);

    let trace = std::env::var("LIG_PROVE_TRACE").is_ok();
    let mut t_init_sumcheck = std::time::Duration::ZERO;
    let mut t_commits = std::time::Duration::ZERO;
    let mut t_opens = std::time::Duration::ZERO;
    let mut t_induce = std::time::Duration::ZERO;
    let mut t_sumcheck_folds = std::time::Duration::ZERO;
    let mut t_intro_glue = std::time::Duration::ZERO;
    let mut t_ood = std::time::Duration::ZERO;

    let t_total = std::time::Instant::now();

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);

    let initial_root: Hash = l0_tree[l0_tree.len() - 1];
    let l0_block_len = block_len_0;
    let l0_num_interleaved = num_interleaved_0;
    let l0_row = |q: usize| -> &[F128] {
        let start = q * l0_num_interleaved;
        &l0_codeword[start..start + l0_num_interleaved]
    };
    challenger.observe_bytes(&initial_root);

    assert_eq!(
        config.ood_samples.first().copied().unwrap_or(0),
        0,
        "L0 must not take explicit OOD samples"
    );
    let mut ood_values: Vec<F128> = Vec::new();
    let mut fold_grinding_nonces: Vec<u64> = Vec::new();
    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };

    let _t = std::time::Instant::now();
    let mut r_lane_fold = Vec::with_capacity(initial_k);
    let mut sc_prover = if let Some(mut direct_claims) = s0_claims {
        assert_eq!(initial_k, 6, "s0 requires initial_k == 6");
        let start_msg = first_msg.expect("s0 requires precomputed round-0 message");
        let mut prefix_transcript = Vec::with_capacity(initial_k + 1);
        prefix_transcript.push(start_msg);
        let mut t_r = target;
        challenger.observe_f128(start_msg.u_0);
        challenger.observe_f128(start_msg.u_2);

        for j in 0..(initial_k - 1) {
            let bits = fold_bits(0).saturating_sub(j as u32);
            if bits > 0 {
                fold_grinding_nonces.push(challenger.grind_pow(bits));
            }
            let r = challenger.sample_f128();
            t_r = RoundQuad::from_msg(prefix_transcript[j], t_r).eval(r);
            let msg = fold_s0_factors_and_message(&mut direct_claims, j, r);
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            prefix_transcript.push(msg);
            r_lane_fold.push(r);
        }

        let j_last = initial_k - 1;
        let bits = fold_bits(0).saturating_sub(j_last as u32);
        if bits > 0 {
            fold_grinding_nonces.push(challenger.grind_pow(bits));
        }
        let r_last = challenger.sample_f128();
        t_r = RoundQuad::from_msg(prefix_transcript[j_last], t_r).eval(r_last);
        r_lane_fold.push(r_last);
        let r_arr: [F128; 6] = r_lane_fold
            .as_slice()
            .try_into()
            .expect("initial_k == 6 fold challenges");

        let (folded_f, folded_b, msg_after_fold) =
            materialize_s0(&packed_witness, &b_initial, &direct_claims, &r_arr);
        crate::scratch::give_f128(packed_witness);
        crate::scratch::give_f128(b_initial);
        challenger.observe_f128(msg_after_fold.u_0);
        challenger.observe_f128(msg_after_fold.u_2);
        prefix_transcript.push(msg_after_fold);

        SumcheckProver::new_after_s0(folded_f, folded_b, t_r, prefix_transcript)
    } else {
        let (mut sc_prover, start_msg) = match first_msg {
            Some(msg) => SumcheckProver::new_with_first_msg(packed_witness, b_initial, target, msg),
            None => SumcheckProver::new(packed_witness, b_initial, target),
        };
        challenger.observe_f128(start_msg.u_0);
        challenger.observe_f128(start_msg.u_2);

        for j in 0..initial_k {
            let bits = fold_bits(0).saturating_sub(j as u32);
            if bits > 0 {
                fold_grinding_nonces.push(challenger.grind_pow(bits));
            }
            let r = challenger.sample_f128();
            let msg = sc_prover.fold(r);
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            r_lane_fold.push(r);
        }
        sc_prover
    };
    if trace {
        t_init_sumcheck += _t.elapsed();
    }

    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1);
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    let _t = std::time::Instant::now();
    let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
    let wtns_1 = ligero_commit(
        sc_prover.f(),
        log_msg_cols_1,
        log_num_interleaved_1,
        log_inv_rate_1,
        &ntt_1,
        config.merkle_hash,
    );
    if trace {
        t_commits += _t.elapsed();
    }
    challenger.observe_bytes(&wtns_1.root());

    {
        let _t = std::time::Instant::now();
        for _ in 0..ood_count(1) {
            let z = challenger.sample_f128_vec(n1);

            let eq_z = super::ring_switch::build_eq_parallel(&z);
            let (intro, y) = sc_prover.introduce_new_with_eval(eq_z);
            challenger.observe_f128(y);
            ood_values.push(y);
            challenger.observe_f128(intro.u_0);
            challenger.observe_f128(intro.u_2);
            let beta = challenger.sample_f128();
            sc_prover.glue(beta);
        }
        if trace {
            t_ood += _t.elapsed();
        }
    }

    let pow_nonce_0 = challenger.grind_pow(config.grinding_bits[0] as u32);
    let mut grinding_nonces: Vec<u64> = vec![pow_nonce_0];

    let num_queries_0 = config.queries[0];
    let queries_0 = sample_distinct_queries(challenger, l0_block_len, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let _t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = {
        use rayon::prelude::*;
        queries_0.par_iter().map(|&q| l0_row(q).to_vec()).collect()
    };
    let merkle_proof_0 = merkle_multi_proof_for(l0_tree, l0_block_len, &queries_0);
    if trace {
        t_opens += _t.elapsed();
    }

    let sks_vks_n1 = eval_sk_at_vks(n1);
    let _t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &opened_rows_0,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );
    if trace {
        t_induce += _t.elapsed();
    }
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0,
        merkle_proof: merkle_proof_0,
    };

    let _t = std::time::Instant::now();
    let intro_msg_0 = sc_prover.introduce_new(basis_0_induced, enforced_sum_0);
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    sc_prover.glue(beta_0);
    if trace {
        t_intro_glue += _t.elapsed();
    }

    let mut wtns_prev = wtns_1;
    let mut recursive_roots: Vec<Hash> = vec![wtns_prev.root()];
    let mut recursive_proofs: Vec<RecursiveProof> = Vec::new();

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        let mut level_rs = Vec::with_capacity(k_i);
        let _t = std::time::Instant::now();
        for j in 0..k_i {
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                fold_grinding_nonces.push(challenger.grind_pow(bits));
            }
            let ri = challenger.sample_f128();
            let msg = sc_prover.fold(ri);
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            level_rs.push(ri);
        }
        if trace {
            t_sumcheck_folds += _t.elapsed();
        }

        if i == r - 1 {
            let yr = sc_prover.f().to_vec();
            for v in &yr {
                challenger.observe_f128(*v);
            }

            let nonce_last = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
            grinding_nonces.push(nonce_last);
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            let _t = std::time::Instant::now();
            let opened_rows_last: Vec<Vec<F128>> = queries_last
                .iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect();
            let merkle_proof_last =
                merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_last);
            if trace {
                t_opens += _t.elapsed();
            }
            if trace {
                let total = t_total.elapsed();
                eprintln!("[lig-prove] total = {:.2} ms", total.as_secs_f64() * 1e3);
                eprintln!(
                    "  initial sumcheck (initial_k folds + SC build): {:.2} ms",
                    t_init_sumcheck.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  recursive commits (NTT + merkle):              {:.2} ms",
                    t_commits.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  opens (rows + multi-proof):                    {:.2} ms",
                    t_opens.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  induce_sumcheck_poly:                          {:.2} ms",
                    t_induce.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  sumcheck recursive folds:                      {:.2} ms",
                    t_sumcheck_folds.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  introduce_new + glue:                          {:.2} ms",
                    t_intro_glue.as_secs_f64() * 1e3
                );
                if !ood_values.is_empty() {
                    eprintln!(
                        "  OOD samples ({}): MLE evals + glue:            {:.2} ms",
                        ood_values.len(),
                        t_ood.as_secs_f64() * 1e3
                    );
                }
            }
            return LigeritoProof {
                initial_root,
                initial_proof,
                recursive_roots,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: opened_rows_last,
                    merkle_proof: merkle_proof_last,
                },
                sumcheck_transcript: sc_prover.transcript().to_vec(),
                grinding_nonces,
                ood_values,
                fold_grinding_nonces,
            };
        }

        let n_next = sc_prover.f().len().trailing_zeros() as usize;
        let log_num_interleaved_next = config.recursive_ks[i + 1];
        assert!(n_next >= log_num_interleaved_next);
        let log_msg_cols_next = n_next - log_num_interleaved_next;
        let log_inv_rate_next = config.log_inv_rates[i + 2];
        let _t = std::time::Instant::now();
        let ntt_next = AdditiveNttF128::standard(log_msg_cols_next + log_inv_rate_next);
        let wtns_next = ligero_commit(
            sc_prover.f(),
            log_msg_cols_next,
            log_num_interleaved_next,
            log_inv_rate_next,
            &ntt_next,
            config.merkle_hash,
        );
        if trace {
            t_commits += _t.elapsed();
        }
        let root_next = wtns_next.root();
        challenger.observe_bytes(&root_next);
        recursive_roots.push(root_next);

        {
            let _t = std::time::Instant::now();
            for _ in 0..ood_count(i + 2) {
                let z = challenger.sample_f128_vec(n_next);
                let eq_z = super::ring_switch::build_eq_parallel(&z);
                let (intro, y) = sc_prover.introduce_new_with_eval(eq_z);
                challenger.observe_f128(y);
                ood_values.push(y);
                challenger.observe_f128(intro.u_0);
                challenger.observe_f128(intro.u_2);
                let beta = challenger.sample_f128();
                sc_prover.glue(beta);
            }
            if trace {
                t_ood += _t.elapsed();
            }
        }

        let nonce_i = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
        grinding_nonces.push(nonce_i);
        let num_queries_i = config.queries[i + 1];
        let queries_i = sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let _t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = queries_i
            .iter()
            .map(|&q| wtns_prev.row(q).to_vec())
            .collect();
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        if trace {
            t_opens += _t.elapsed();
        }

        let sks_vks_i = eval_sk_at_vks(n_next);
        let _t = std::time::Instant::now();
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_next,
            &sks_vks_i,
            &opened_rows_i,
            &level_rs,
            &queries_i,
            &alpha_i,
        );
        if trace {
            t_induce += _t.elapsed();
        }
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows_i,
            merkle_proof: merkle_proof_i,
        });

        let _t = std::time::Instant::now();
        let intro_msg_i = sc_prover.introduce_new(basis_i_induced, enforced_sum_i);
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        sc_prover.glue(beta_i);
        if trace {
            t_intro_glue += _t.elapsed();
        }

        wtns_prev = wtns_next;
    }

    unreachable!()
}

#[allow(clippy::too_many_arguments)]
pub fn recursive_verifier_with_basis_succinct<Ch, F>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    log_n: usize,
    target: F128,
    expected_initial_root: &Hash,
    eval_b_residual: F,
    challenger: &mut Ch,
) -> bool
where
    Ch: Challenger,
    F: Fn(&[F128], usize) -> Vec<F128>,
{
    let trace = std::env::var("LIG_VERIFY_TRACE").is_ok();
    let mut t_merkle = std::time::Duration::ZERO;
    let mut t_sample_q = std::time::Duration::ZERO;
    let mut t_enforced = std::time::Duration::ZERO;
    let mut t_residual = std::time::Duration::ZERO;
    let mut t_evalb = std::time::Duration::ZERO;
    let t_start = std::time::Instant::now();

    let initial_k = config.initial_k;
    let r = config.recursive_steps;
    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }
    if &proof.initial_root != expected_initial_root {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);
    challenger.observe_bytes(&proof.initial_root);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;

    let mut t_r = target;
    let mut tx_idx = 0usize;
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };
    if config.ood_samples.first().copied().unwrap_or(0) != 0 {
        return false;
    }
    let mut fold_nonce_idx = 0usize;
    let mut ood_idx = 0usize;

    struct OodCtx {
        z: Vec<F128>,
        ris_start: usize,
        beta: F128,
    }
    let mut ood_ctxs: Vec<OodCtx> = Vec::new();

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                return false;
            }
            fold_nonce_idx += 1;
        }
        let ri = challenger.sample_f128();
        r_lane_fold.push(ri);
        t_r = running_quad.eval(ri);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        running_quad = RoundQuad::from_msg(msg, t_r);
    }

    if proof.recursive_roots.is_empty() {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    for _ in 0..ood_count(1) {
        let z = challenger.sample_f128_vec(log_n - initial_k);
        if ood_idx >= proof.ood_values.len() {
            return false;
        }
        let y = proof.ood_values[ood_idx];
        ood_idx += 1;
        challenger.observe_f128(y);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg.u_0);
        challenger.observe_f128(intro_msg.u_2);
        let intro_quad = RoundQuad::from_msg(intro_msg, y);
        let beta = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
        t_r += beta * y;
        ood_ctxs.push(OodCtx {
            z,
            ris_start: initial_k,
            beta,
        });
    }

    let mut nonce_idx = 0usize;
    if nonce_idx >= proof.grinding_nonces.len() {
        return false;
    }
    if !challenger.verify_pow(
        proof.grinding_nonces[nonce_idx],
        config.grinding_bits[0] as u32,
    ) {
        return false;
    }
    nonce_idx += 1;

    let num_queries_0 = config.queries[0];
    let _t = std::time::Instant::now();
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    if trace {
        t_sample_q += _t.elapsed();
    }
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let _t = std::time::Instant::now();
    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
    ) {
        return false;
    }
    if trace {
        t_merkle += _t.elapsed();
    }

    let n1 = log_n - initial_k;
    let _t = std::time::Instant::now();
    let enforced_sum_0 = induce_sumcheck_enforced_sum(
        &proof.initial_proof.opened_rows,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );
    if trace {
        t_enforced += _t.elapsed();
    }

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;

    struct LevelCtx {
        log_msg_cols: usize,
        queries: Vec<usize>,
        alpha: Vec<F128>,
        ris_start: usize,
        beta: F128,
    }
    let mut level_ctxs: Vec<LevelCtx> = vec![LevelCtx {
        log_msg_cols: n1,
        queries: queries_0.clone(),
        alpha: alpha_0,
        ris_start: initial_k,
        beta: beta_0,
    }];
    let mut ris: Vec<F128> = r_lane_fold.clone();

    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1];
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for j in 0..k_i {
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                    return false;
                }
                if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                    return false;
                }
                fold_nonce_idx += 1;
            }
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            if ood_idx != proof.ood_values.len()
                || fold_nonce_idx != proof.fold_grinding_nonces.len()
            {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }

            if nonce_idx >= proof.grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(
                proof.grinding_nonces[nonce_idx],
                config.grinding_bits[i + 1] as u32,
            ) {
                return false;
            }

            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = config.queries[i + 1];
            let _t = std::time::Instant::now();
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);

            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if trace {
                t_sample_q += _t.elapsed();
            }
            let _t = std::time::Instant::now();
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
            ) {
                return false;
            }
            if trace {
                t_merkle += _t.elapsed();
            }

            let enforced_sum_last = induce_sumcheck_enforced_sum(
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            level_ctxs.push(LevelCtx {
                log_msg_cols: n_current,
                queries: queries_last.clone(),
                alpha: alpha_last,
                ris_start: ris.len(),
                beta: beta_last,
            });

            let yr_len = yr.len();
            let yr_log_n = n_current;

            let _t = std::time::Instant::now();
            let induced_residuals: Vec<Vec<F128>> = level_ctxs
                .iter()
                .map(|ctx| {
                    let sks_vks = eval_sk_at_vks(ctx.log_msg_cols);
                    let ris_for_basis =
                        &ris[ctx.ris_start..ctx.ris_start + ctx.log_msg_cols - yr_log_n];
                    induce_sumcheck_evaluate_at_residual(
                        ctx.log_msg_cols,
                        &sks_vks,
                        &ctx.queries,
                        &ctx.alpha,
                        ris_for_basis,
                        yr_log_n,
                    )
                })
                .collect();
            if trace {
                t_residual += _t.elapsed();
            }
            for resid in &induced_residuals {
                if resid.len() != yr_len {
                    return false;
                }
            }

            let mut ood_residuals: Vec<Vec<F128>> = Vec::with_capacity(ood_ctxs.len());
            for ctx in &ood_ctxs {
                if ctx.z.len() < yr_log_n || ctx.ris_start + (ctx.z.len() - yr_log_n) > ris.len() {
                    return false;
                }
                let folded = ctx.z.len() - yr_log_n;
                let mut scalar = ctx.beta;
                for b in 0..folded {
                    scalar *= F128::ONE + ctx.z[b] + ris[ctx.ris_start + b];
                }
                let mut tail = build_eq_table(&ctx.z[folded..]);
                for v in tail.iter_mut() {
                    *v *= scalar;
                }
                ood_residuals.push(tail);
            }

            let _te = std::time::Instant::now();
            let evb_vec = eval_b_residual(&ris, yr_log_n);
            if trace {
                t_evalb += _te.elapsed();
            }
            if evb_vec.len() != yr_len {
                return false;
            }
            let mut inner = F128::ZERO;
            let _t = std::time::Instant::now();
            for y in 0..yr_len {
                let mut combined_y = evb_vec[y];
                for (k, residual) in induced_residuals.iter().enumerate() {
                    combined_y += level_ctxs[k].beta * residual[y];
                }
                for resid in &ood_residuals {
                    combined_y += resid[y];
                }
                inner += yr[y] * combined_y;
            }
            if trace {
                t_residual += _t.elapsed();
            }
            if trace {
                let total = t_start.elapsed();
                eprintln!("[lig-verify] total = {:.2} ms", total.as_secs_f64() * 1e3);
                eprintln!(
                    "  merkle multi-proofs:       {:.2} ms",
                    t_merkle.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  sample_distinct_queries:   {:.2} ms",
                    t_sample_q.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  enforced_sum (eq+dot):     {:.2} ms",
                    t_enforced.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  residual basis eval:       {:.2} ms",
                    t_residual.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  eval_b (yr_len positions): {:.2} ms",
                    t_evalb.as_secs_f64() * 1e3
                );
            }
            return inner == t_r;
        }

        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        for _ in 0..ood_count(i + 2) {
            let z = challenger.sample_f128_vec(n_current);
            if ood_idx >= proof.ood_values.len() {
                return false;
            }
            let y = proof.ood_values[ood_idx];
            ood_idx += 1;
            challenger.observe_f128(y);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let intro_msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(intro_msg.u_0);
            challenger.observe_f128(intro_msg.u_2);
            let intro_quad = RoundQuad::from_msg(intro_msg, y);
            let beta = challenger.sample_f128();
            running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
            t_r += beta * y;
            ood_ctxs.push(OodCtx {
                z,
                ris_start: ris.len(),
                beta,
            });
        }

        if nonce_idx >= proof.grinding_nonces.len() {
            return false;
        }
        if !challenger.verify_pow(
            proof.grinding_nonces[nonce_idx],
            config.grinding_bits[i + 1] as u32,
        ) {
            return false;
        }
        nonce_idx += 1;

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = config.queries[i + 1];
        let _t = std::time::Instant::now();
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        if trace {
            t_sample_q += _t.elapsed();
        }
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        let _t = std::time::Instant::now();
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
            config.merkle_hash,
        ) {
            return false;
        }
        if trace {
            t_merkle += _t.elapsed();
        }

        let _t = std::time::Instant::now();
        let enforced_sum_i =
            induce_sumcheck_enforced_sum(&rp.opened_rows, &level_rs, &queries_i, &alpha_i);
        if trace {
            t_enforced += _t.elapsed();
        }

        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        level_ctxs.push(LevelCtx {
            log_msg_cols: n_current,
            queries: queries_i.clone(),
            alpha: alpha_i,
            ris_start: ris.len(),
            beta: beta_i,
        });

        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!()
}

#[allow(clippy::too_many_arguments)]
pub fn recursive_verifier_with_basis<Ch: Challenger>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    b_initial: &[F128],
    target: F128,
    expected_initial_root: &Hash,
    challenger: &mut Ch,
) -> bool {
    let log_n = b_initial.len().trailing_zeros() as usize;
    let initial_k = config.initial_k;
    let r = config.recursive_steps;

    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }
    if b_initial.len() != 1usize << log_n {
        return false;
    }
    if &proof.initial_root != expected_initial_root {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);
    challenger.observe_bytes(&proof.initial_root);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;

    let mut t_r = target;
    let mut tx_idx = 0usize;
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };
    if config.ood_samples.first().copied().unwrap_or(0) != 0 {
        return false;
    }
    let mut fold_nonce_idx = 0usize;
    let mut ood_idx = 0usize;

    let mut ood_bases: Vec<(Vec<F128>, usize, F128)> = Vec::new();

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                return false;
            }
            fold_nonce_idx += 1;
        }
        let ri = challenger.sample_f128();
        r_lane_fold.push(ri);
        t_r = running_quad.eval(ri);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        running_quad = RoundQuad::from_msg(msg, t_r);
    }

    if proof.recursive_roots.is_empty() {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    for _ in 0..ood_count(1) {
        let z = challenger.sample_f128_vec(log_n - initial_k);
        if ood_idx >= proof.ood_values.len() {
            return false;
        }
        let y = proof.ood_values[ood_idx];
        ood_idx += 1;
        challenger.observe_f128(y);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg.u_0);
        challenger.observe_f128(intro_msg.u_2);
        let intro_quad = RoundQuad::from_msg(intro_msg, y);
        let beta = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
        t_r += beta * y;
        ood_bases.push((build_eq_table(&z), initial_k, beta));
    }

    let mut nonce_idx = 0usize;
    if nonce_idx >= proof.grinding_nonces.len() {
        return false;
    }
    if !challenger.verify_pow(
        proof.grinding_nonces[nonce_idx],
        config.grinding_bits[0] as u32,
    ) {
        return false;
    }
    nonce_idx += 1;

    let num_queries_0 = config.queries[0];
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
    ) {
        return false;
    }

    let n1 = log_n - initial_k;
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &proof.initial_proof.opened_rows,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;

    let mut basis_polys: Vec<Vec<F128>> = vec![b_initial.to_vec(), basis_0_induced];
    let mut basis_ris_starts: Vec<usize> = vec![0, initial_k];
    let mut basis_separations: Vec<F128> = vec![beta_0];
    let mut ris: Vec<F128> = r_lane_fold.clone();

    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1];
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for j in 0..k_i {
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                    return false;
                }
                if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                    return false;
                }
                fold_nonce_idx += 1;
            }
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            if ood_idx != proof.ood_values.len()
                || fold_nonce_idx != proof.fold_grinding_nonces.len()
            {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }

            if nonce_idx >= proof.grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(
                proof.grinding_nonces[nonce_idx],
                config.grinding_bits[i + 1] as u32,
            ) {
                return false;
            }

            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);

            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
            ) {
                return false;
            }

            let sks_vks_last = eval_sk_at_vks(n_current);
            let (basis_last_induced, enforced_sum_last) = induce_sumcheck_poly(
                n_current,
                &sks_vks_last,
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            basis_polys.push(basis_last_induced);
            basis_ris_starts.push(ris.len());
            basis_separations.push(beta_last);

            let yr_len = yr.len();
            let mut combined = vec![F128::ZERO; yr_len];
            for (k, basis) in basis_polys.iter().enumerate() {
                let start = basis_ris_starts[k];
                let residual = partial_eval_lsb(basis, &ris[start..]);
                if residual.len() != yr_len {
                    return false;
                }
                let sep = if k == 0 {
                    F128::ONE
                } else {
                    basis_separations[k - 1]
                };
                for (c, &rr) in combined.iter_mut().zip(residual.iter()) {
                    *c += sep * rr;
                }
            }

            for (basis, start, beta) in &ood_bases {
                let residual = partial_eval_lsb(basis, &ris[*start..]);
                if residual.len() != yr_len {
                    return false;
                }
                for (c, &rr) in combined.iter_mut().zip(residual.iter()) {
                    *c += *beta * rr;
                }
            }
            let inner: F128 = yr
                .iter()
                .zip(combined.iter())
                .map(|(&y, &c)| y * c)
                .fold(F128::ZERO, |a, v| a + v);
            return inner == t_r;
        }

        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        for _ in 0..ood_count(i + 2) {
            let z = challenger.sample_f128_vec(n_current);
            if ood_idx >= proof.ood_values.len() {
                return false;
            }
            let y = proof.ood_values[ood_idx];
            ood_idx += 1;
            challenger.observe_f128(y);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let intro_msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(intro_msg.u_0);
            challenger.observe_f128(intro_msg.u_2);
            let intro_quad = RoundQuad::from_msg(intro_msg, y);
            let beta = challenger.sample_f128();
            running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
            t_r += beta * y;
            ood_bases.push((build_eq_table(&z), ris.len(), beta));
        }

        if nonce_idx >= proof.grinding_nonces.len() {
            return false;
        }
        if !challenger.verify_pow(
            proof.grinding_nonces[nonce_idx],
            config.grinding_bits[i + 1] as u32,
        ) {
            return false;
        }
        nonce_idx += 1;

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = config.queries[i + 1];
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
            config.merkle_hash,
        ) {
            return false;
        }

        let sks_vks_i = eval_sk_at_vks(n_current);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_current,
            &sks_vks_i,
            &rp.opened_rows,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        basis_polys.push(basis_i_induced);
        basis_ris_starts.push(ris.len());
        basis_separations.push(beta_i);

        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!()
}

#[allow(clippy::too_many_arguments)]
fn recursive_prover_inner<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    wtns_0: LigeroWitness,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
    t_total: std::time::Instant,
    mut t_commits: std::time::Duration,
    mut t_induce: std::time::Duration,
    mut t_sumcheck: std::time::Duration,
    mut t_opens: std::time::Duration,
    trace: bool,
) -> LigeritoProof {
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }

    assert!(
        config.ood_samples.iter().all(|&s| s == 0)
            && config.fold_grinding_bits.iter().all(|&b| b == 0),
        "OOD samples / fold grinding require the with_basis prover path"
    );
    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    let log_inv_rate_0 = config.log_inv_rates[0];

    let initial_root = wtns_0.root();
    challenger.observe_bytes(&initial_root);

    let v_challenges_0 = eval_point[..initial_k].to_vec();
    let f1 = partial_eval_lsb(poly, &v_challenges_0);
    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1, "n1 < k_0");
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
    let t = std::time::Instant::now();
    let wtns_1 = ligero_commit(
        &f1,
        log_msg_cols_1,
        log_num_interleaved_1,
        log_inv_rate_1,
        &ntt_1,
        config.merkle_hash,
    );
    let t_l1 = t.elapsed();
    t_commits += t_l1;
    tlog!("  [ligerito]   L1 commit: {:.2?}", t_l1);
    challenger.observe_bytes(&wtns_1.root());

    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_distinct_queries(challenger, wtns_0.block_len, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| wtns_0.row(q).to_vec()).collect();
    let merkle_proof_0 = merkle_multi_proof_for(&wtns_0.tree, wtns_0.block_len, &queries_0);
    t_opens += t.elapsed();
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0.clone(),
        merkle_proof: merkle_proof_0,
    };

    let sks_vks_n1 = eval_sk_at_vks(n1);
    let t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &opened_rows_0,
        &v_challenges_0,
        &queries_0,
        &alpha_0,
    );
    t_induce += t.elapsed();

    let eq_z_residual = build_eq_table(&eval_point[initial_k..]);
    let t = std::time::Instant::now();
    let (mut sc_prover, start_msg) = SumcheckProver::new(f1, eq_z_residual, claimed_value);
    t_sumcheck += t.elapsed();
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    let intro_msg_0 = sc_prover.introduce_new(basis_0_induced, enforced_sum_0);
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    sc_prover.glue(beta_0);

    let mut wtns_prev = wtns_1;
    let mut recursive_roots: Vec<Hash> = vec![wtns_prev.root()];
    let mut recursive_proofs: Vec<RecursiveProof> = Vec::new();

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        let mut level_rs = Vec::with_capacity(k_i);
        let t = std::time::Instant::now();
        for _ in 0..k_i {
            let ri = challenger.sample_f128();
            let msg = sc_prover.fold(ri);
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            level_rs.push(ri);
        }
        t_sumcheck += t.elapsed();

        if i == r - 1 {
            tlog!(
                "  [ligerito] commits: {:.2?}  induce: {:.2?}  sumcheck: {:.2?}  opens: {:.2?}  TOTAL: {:.2?}",
                t_commits,
                t_induce,
                t_sumcheck,
                t_opens,
                t_total.elapsed()
            );

            let yr = sc_prover.f().to_vec();
            for v in &yr {
                challenger.observe_f128(*v);
            }

            let num_queries_last = udr_queries(config.log_inv_rates[i + 1]);
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            let opened_rows_last: Vec<Vec<F128>> = queries_last
                .iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect();
            let merkle_proof_last =
                merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_last);
            return LigeritoProof {
                initial_root,
                initial_proof,
                recursive_roots,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: opened_rows_last,
                    merkle_proof: merkle_proof_last,
                },
                sumcheck_transcript: sc_prover.transcript().to_vec(),
                grinding_nonces: Vec::new(),
                ood_values: Vec::new(),
                fold_grinding_nonces: Vec::new(),
            };
        }

        let n_next = sc_prover.f().len().trailing_zeros() as usize;
        let log_num_interleaved_next = config.recursive_ks[i + 1];
        assert!(
            n_next >= log_num_interleaved_next,
            "f.n ({n_next}) < k_{} ({log_num_interleaved_next})",
            i + 1
        );
        let log_msg_cols_next = n_next - log_num_interleaved_next;
        let log_inv_rate_next = config.log_inv_rates[i + 2];
        let ntt_next = AdditiveNttF128::standard(log_msg_cols_next + log_inv_rate_next);
        let f_evals = sc_prover.f().to_vec();
        let t = std::time::Instant::now();
        let wtns_next = ligero_commit(
            &f_evals,
            log_msg_cols_next,
            log_num_interleaved_next,
            log_inv_rate_next,
            &ntt_next,
            config.merkle_hash,
        );
        let t_li = t.elapsed();
        t_commits += t_li;
        tlog!("  [ligerito]   L{} commit: {:.2?}", i + 2, t_li);
        let root_next = wtns_next.root();
        challenger.observe_bytes(&root_next);
        recursive_roots.push(root_next);

        let num_queries_i = udr_queries(config.log_inv_rates[i + 1]);
        let queries_i = sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = queries_i
            .iter()
            .map(|&q| wtns_prev.row(q).to_vec())
            .collect();
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        t_opens += t.elapsed();
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows_i.clone(),
            merkle_proof: merkle_proof_i,
        });

        let sks_vks_i = eval_sk_at_vks(n_next);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_next,
            &sks_vks_i,
            &opened_rows_i,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        let intro_msg_i = sc_prover.introduce_new(basis_i_induced, enforced_sum_i);
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        sc_prover.glue(beta_i);

        wtns_prev = wtns_next;
    }

    unreachable!("recursive loop should return on last iter")
}

fn verify_level_opens(
    root: &Hash,
    block_len: usize,
    queries: &[usize],
    opened_rows: &[Vec<F128>],
    expected_num_interleaved: usize,
    multi_proof: &[Hash],
    kind: HashKind,
) -> bool {
    if queries.len() != opened_rows.len() {
        return false;
    }
    let mut leaf_hashes: Vec<Hash> = Vec::with_capacity(opened_rows.len());
    for row in opened_rows {
        if row.len() != expected_num_interleaved {
            return false;
        }
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                row.as_ptr() as *const u8,
                row.len() * core::mem::size_of::<F128>(),
            )
        };
        leaf_hashes.push(merkle::hash_leaf(bytes, kind));
    }
    merkle::verify_merkle_multi_proof(root, block_len, queries, &leaf_hashes, multi_proof, kind)
}

pub fn recursive_verifier<Ch: Challenger>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> bool {
    let log_n = eval_point.len();
    let initial_k = config.initial_k;
    let r = config.recursive_steps;

    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }

    if config.ood_samples.iter().any(|&s| s != 0)
        || config.fold_grinding_bits.iter().any(|&b| b != 0)
    {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    challenger.observe_bytes(&proof.initial_root);
    if proof.recursive_roots.len() != r {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));

    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
    ) {
        return false;
    }

    let n1 = log_n - initial_k;
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &proof.initial_proof.opened_rows,
        &eval_point[..initial_k],
        &queries_0,
        &alpha_0,
    );

    let eq_z_residual = build_eq_table(&eval_point[initial_k..]);

    let mut basis_polys: Vec<Vec<F128>> = vec![eq_z_residual];
    let mut basis_ris_starts: Vec<usize> = vec![0];
    let mut basis_separations: Vec<F128> = Vec::new();
    let mut ris: Vec<F128> = Vec::new();
    let mut t_r = claimed_value;
    let mut tx_idx = 0usize;

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;
    basis_polys.push(basis_0_induced);
    basis_ris_starts.push(0);
    basis_separations.push(beta_0);

    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1];
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for _ in 0..k_i {
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }
            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = udr_queries(prev_log_inv_rate);
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);

            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
            ) {
                return false;
            }

            let sks_vks_last = eval_sk_at_vks(n_current);
            let (basis_last_induced, enforced_sum_last) = induce_sumcheck_poly(
                n_current,
                &sks_vks_last,
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            basis_polys.push(basis_last_induced);
            basis_ris_starts.push(ris.len());
            basis_separations.push(beta_last);

            let yr_len = yr.len();
            let mut combined = vec![F128::ZERO; yr_len];
            for (k, basis) in basis_polys.iter().enumerate() {
                let start = basis_ris_starts[k];
                let residual = partial_eval_lsb(basis, &ris[start..]);
                if residual.len() != yr_len {
                    return false;
                }
                let sep = if k == 0 {
                    F128::ONE
                } else {
                    basis_separations[k - 1]
                };
                for (c, &r) in combined.iter_mut().zip(residual.iter()) {
                    *c += sep * r;
                }
            }
            let inner: F128 = yr
                .iter()
                .zip(combined.iter())
                .map(|(&y, &c)| y * c)
                .fold(F128::ZERO, |a, v| a + v);
            return inner == t_r;
        }

        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = udr_queries(prev_log_inv_rate);
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));

        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
            config.merkle_hash,
        ) {
            return false;
        }

        let sks_vks_i = eval_sk_at_vks(n_current);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_current,
            &sks_vks_i,
            &rp.opened_rows,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        basis_polys.push(basis_i_induced);
        basis_ris_starts.push(ris.len());
        basis_separations.push(beta_i);

        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!("loop should return at i = r - 1")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blake3_m29_udr_example() -> LigeritoSecurityConfig {
        LigeritoSecurityConfig::derive_paper_compatible(29, 1, 100).expect("derive m29 fast")
    }

    #[test]
    fn ligerito_security_config_m29_toml_loads() {
        let toml_str = include_str!("../../configs/ligerito/m29_fast.toml");
        let cfg = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_fast.toml must parse and validate");
        assert_eq!(cfg.m, 29);
        assert_eq!(cfg.log_n, 22);
        assert_eq!(cfg.initial_k, 6);
        assert_eq!(cfg.hash, "sha256");
        assert_eq!(cfg.levels.len(), 5);

        assert_eq!(cfg.levels[0].regime, SoundnessRegime::JohnsonOod);
        assert_eq!(cfg.levels[0].queries, 218);
        assert_eq!(cfg.levels[0].grinding_bits, 0);
        assert!(cfg.levels[0].fold_grinding_bits > 0);
        assert_eq!(cfg.levels[0].ood_samples, 0);
        assert!(cfg.levels[1].ood_samples >= 1);
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 6, 1).unwrap();
        assert_eq!(pv.log_inv_rates, default.log_inv_rates);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.queries[0], 218);

        let toml_str = include_str!("../../configs/ligerito/m29_slim.toml");
        let cfg_slim = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_slim.toml must parse and validate");
        assert_eq!(cfg_slim.levels[0].log_inv_rate, 2);

        assert_eq!(cfg_slim.levels[0].queries, 90);
        assert_eq!(cfg_slim.levels[0].grinding_bits, 16);
        let (pv_slim, _vc_slim) = cfg_slim.to_prover_verifier_configs().unwrap();
        let default_slim = default_config(22, 6, 2).unwrap();
        assert_eq!(pv_slim.log_inv_rates, default_slim.log_inv_rates);
        assert_eq!(pv_slim.recursive_ks, default_slim.recursive_ks);
    }

    #[test]
    #[ignore]
    fn regen_embedded_tomls() {
        for m in [22usize, 29, 32] {
            for profile in [
                LigeritoProfile::Fast,
                LigeritoProfile::Slim,
                LigeritoProfile::Secure,
            ] {
                let cfg = LigeritoSecurityConfig::derive_profile(m, profile)
                    .unwrap_or_else(|e| panic!("derive m{m}_{}: {e}", profile.as_str()));
                let toml = cfg.to_toml_string().expect("serialize");
                eprintln!(
                    "\n# ====== configs/ligerito/m{m}_{}.toml ======",
                    profile.as_str()
                );
                eprintln!("{toml}");
            }
        }
    }

    #[test]
    fn ligerito_security_config_rejects_paper_inconsistent_eps_pg() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_pg_bits = 50.0;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("doesn't match") && err.contains("prediction"),
            "expected paper-mismatch error, got: {err}"
        );
    }

    #[test]
    fn ligerito_security_config_rejects_paper_inconsistent_eps_query() {
        let mut cfg = blake3_m29_udr_example();

        cfg.levels[0].expected_eps_query_bits += 5.0;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("doesn't match") && err.contains("prediction"),
            "expected paper-mismatch error, got: {err}"
        );
    }

    #[test]
    fn ligerito_all_embedded_configs_validate() {
        for &(key, toml) in EMBEDDED_CONFIGS {
            LigeritoSecurityConfig::from_toml_str(toml).unwrap_or_else(|e| {
                panic!(
                    "embedded config m={} profile={} invalid: {e}",
                    key.0,
                    key.1.as_str()
                )
            });
        }
    }

    #[test]
    fn ligerito_derive_paper_compatible_for_all_embedded() {
        let pairs: &[(usize, usize)] = &[(22, 1), (28, 1), (29, 1), (29, 2), (30, 1), (30, 2)];
        for &(m, r) in pairs {
            let cfg = LigeritoSecurityConfig::derive_paper_compatible(m, r, 100)
                .unwrap_or_else(|e| panic!("derive m={m} r={r}: {e}"));
            cfg.validate()
                .unwrap_or_else(|e| panic!("derived m={m} r={r} fails validate: {e}"));
        }
        for m in 22..=35usize {
            for profile in [
                LigeritoProfile::Fast,
                LigeritoProfile::Slim,
                LigeritoProfile::Secure,
            ] {
                let cfg = LigeritoSecurityConfig::derive_profile(m, profile)
                    .unwrap_or_else(|e| panic!("derive m={m} {}: {e}", profile.as_str()));
                cfg.validate().unwrap_or_else(|e| {
                    panic!("derived m={m} {} fails validate: {e}", profile.as_str())
                });
            }
        }
    }

    #[test]
    fn ligerito_prover_config_for_lookup() {
        let pv = prover_config_for(22, 6, LigeritoProfile::Fast).expect("m29 fast must load");
        assert_eq!(pv.queries[0], 218);
        assert_eq!(pv.fold_grinding_bits[0], 16);

        let pv = prover_config_for(22, 6, LigeritoProfile::Slim).expect("m29 slim must load");
        assert_eq!(pv.queries[0], 90);
        assert_eq!(pv.grinding_bits[0], 16);

        let pv = prover_config_for(22, 6, LigeritoProfile::Secure).expect("m29 secure must load");
        assert!(pv.queries[0] > 280);
        assert_eq!(pv.ood_samples.iter().sum::<usize>(), 0);

        let err = prover_config_for(29, 6, LigeritoProfile::Fast).unwrap_err();
        assert!(
            err.contains("no security config registered"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ligerito_security_config_toml_roundtrip() {
        let cfg = blake3_m29_udr_example();
        let s = cfg.to_toml_string().expect("serialize");
        let back = LigeritoSecurityConfig::from_toml_str(&s).expect("deserialize");
        assert_eq!(back.levels.len(), cfg.levels.len());
        assert_eq!(back.levels[0].queries, cfg.levels[0].queries);
        assert_eq!(back.levels[0].grinding_bits, cfg.levels[0].grinding_bits);
        assert_eq!(back.final_block.yr_log_n, cfg.final_block.yr_log_n);
    }

    #[test]
    fn ligerito_security_config_validates() {
        let cfg = blake3_m29_udr_example();
        cfg.validate()
            .unwrap_or_else(|e| panic!("validate failed: {e}"));
    }

    #[test]
    fn ligerito_security_config_hash_field_selects_merkle_hash() {
        let mut cfg = blake3_m29_udr_example();
        assert_eq!(cfg.hash, "sha256", "example config baseline");
        let (p, v) = cfg.to_prover_verifier_configs().expect("sha256 configs");
        assert_eq!(p.merkle_hash, HashKind::Sha256);
        assert_eq!(v.merkle_hash, HashKind::Sha256);

        cfg.hash = "blake3".into();
        let (p, v) = cfg.to_prover_verifier_configs().expect("blake3 configs");
        assert_eq!(p.merkle_hash, HashKind::Blake3);
        assert_eq!(v.merkle_hash, HashKind::Blake3);

        cfg.validate().expect("blake3 config validates");
        let back = LigeritoSecurityConfig::from_toml_str(&cfg.to_toml_string().unwrap())
            .expect("toml roundtrip");
        assert_eq!(back.merkle_hash().unwrap(), HashKind::Blake3);
    }

    #[test]
    fn ligerito_security_config_rejects_unknown_hash() {
        let mut cfg = blake3_m29_udr_example();
        cfg.hash = "keccak256".into();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("hash") && err.contains("keccak256"),
            "err = {err}"
        );
        assert!(cfg.to_prover_verifier_configs().is_err());
    }

    #[test]
    fn embedded_configs_all_declare_a_supported_hash() {
        for &((m, profile), toml) in EMBEDDED_CONFIGS {
            let cfg = LigeritoSecurityConfig::from_toml_str(toml)
                .unwrap_or_else(|e| panic!("m{m} {profile:?}: {e}"));
            cfg.merkle_hash()
                .unwrap_or_else(|e| panic!("m{m} {profile:?}: {e}"));
        }
    }

    #[test]
    fn ligerito_security_config_rejects_insufficient_queries() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_query_bits = 50.0;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("expected_eps_query_bits"), "err = {err}");
    }

    #[test]
    fn ligerito_security_config_rejects_udr_with_eta() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].eta = Some(0.02);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("udr") && err.contains("eta"), "err = {err}");
    }

    #[test]
    fn ligerito_security_config_rejects_udr_without_proximity_loss() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].proximity_loss = None;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("udr") && err.contains("proximity_loss"),
            "err = {err}"
        );
    }

    #[test]
    fn ligerito_security_config_rejects_johnson_with_proximity_loss() {
        let mut cfg = blake3_m29_udr_example();

        cfg.levels[0].regime = SoundnessRegime::JohnsonOod;
        cfg.levels[0].eta = Some(0.02);
        cfg.levels[0].proximity_loss = Some(0.01);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("proximity_loss") && err.contains("udr"),
            "err = {err}"
        );
    }

    #[test]
    fn ligerito_security_config_udr_regime_validates() {
        let mut cfg = blake3_m29_udr_example();

        let eps_star = 0.0f64;
        let rho = 0.5f64;
        let delta = 1.0 - rho;
        let n = ((cfg.levels[0].log_msg_cols + cfg.levels[0].log_inv_rate) as f64).exp2();
        let gamma = delta / 2.0 - 3.0 / (delta * n) - eps_star;
        let per_q = (1.0 / (1.0 - gamma)).log2();
        let queries = (100.0 / per_q).ceil() as usize;

        let log_a_base = (gamma * n + 1.0).log2();
        let eps_pg = 128.0 - log_a_base;
        cfg.levels[0].regime = SoundnessRegime::Udr;
        cfg.levels[0].eta = None;
        cfg.levels[0].proximity_loss = Some(eps_star);
        cfg.levels[0].queries = queries;
        cfg.levels[0].grinding_bits = 0;
        cfg.levels[0].fold_grinding_bits = (100.0 - eps_pg).ceil().max(0.0) as usize;
        cfg.levels[0].expected_eps_pg_bits = (eps_pg * 10.0).round() / 10.0;
        cfg.levels[0].expected_eps_query_bits = ((queries as f64 * per_q) * 10.0).round() / 10.0;
        cfg.validate()
            .unwrap_or_else(|e| panic!("UDR config failed to validate: {e}"));
    }

    #[test]
    fn ligerito_security_config_serde_roundtrip() {
        let cfg = blake3_m29_udr_example();
        let json = serde_json::to_string_pretty(&cfg).expect("serialize");
        let back: LigeritoSecurityConfig = serde_json::from_str(&json).expect("deserialize");
        back.validate().expect("roundtripped config validates");
        assert_eq!(back.levels.len(), cfg.levels.len());

        assert_eq!(back.levels[0].queries, 241);
        assert_eq!(back.levels[0].grinding_bits, 0);
    }

    #[test]
    fn ligerito_security_config_drives_roundtrip_with_grinding() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0x6817_D146);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];

        let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let grinding_bits = vec![6usize, 0];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );
        assert_eq!(proof.grinding_nonces.len(), 2, "one nonce per level");

        let v_cfg = VerifierConfig {
            log_inv_rates,
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(
            ok,
            "verifier should accept proof with valid grinding nonces"
        );

        let mut bad_proof = proof.clone();
        bad_proof.grinding_nonces[0] = bad_proof.grinding_nonces[0].wrapping_add(1);
        let mut v_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &bad_proof, &b, target, &initial_root, &mut v_ch);
        assert!(
            !ok,
            "verifier must reject proof with tampered grinding nonce"
        );
    }

    #[test]
    fn ligerito_security_config_matches_default_config() {
        let cfg = blake3_m29_udr_example();
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 6, 1).unwrap();
        assert_eq!(pv.log_inv_rates, default.log_inv_rates);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.initial_k, default.initial_k);
    }

    #[test]
    fn partial_eval_then_eval_equals_full_eval() {
        let n = 6;
        let len = 1usize << n;
        let evals: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0xDEAD_BEEF_CAFE_BABE),
                    0xA5A5 ^ i as u64,
                )
            })
            .collect();
        let point: Vec<F128> = (0..n)
            .map(|i| F128::new(0x1111 * (i as u64 + 1), 0x2222 * (i as u64 + 1)))
            .collect();

        let full = eval_mle_lsb(&evals, &point);

        let k = 3;
        let (lo, hi) = point.split_at(k);
        let residual = partial_eval_lsb(&evals, lo);
        assert_eq!(residual.len(), 1usize << (n - k));
        let after = eval_mle_lsb(&residual, hi);
        assert_eq!(full, after);

        let eq = build_eq_table(&point);
        let dot = evals
            .iter()
            .zip(eq.iter())
            .map(|(&e, &q)| e * q)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(dot, full);
    }

    #[test]
    fn stateful_sumcheck_single_basis_roundtrip() {
        use crate::challenger::Challenger;
        let n = 5;
        let len = 1usize << n;
        let f: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0x1234_5678_9ABC_DEF0),
                    0x55AA ^ i as u64,
                )
            })
            .collect();
        let b: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0xFEDC_BA98_7654_3210),
                    0xAA55 ^ i as u64,
                )
            })
            .collect();
        let h: F128 = f
            .iter()
            .zip(b.iter())
            .map(|(&fi, &bi)| fi * bi)
            .fold(F128::ZERO, |a, v| a + v);

        let (mut prover, _first) = SumcheckProver::new(f.clone(), b.clone(), h);
        let mut ch = crate::challenger::RandomChallenger::new(0xC0FFEE);
        let mut ris: Vec<F128> = Vec::new();
        for _ in 0..(n - 1) {
            let r = ch.sample_f128();
            ris.push(r);
            prover.fold(r);
        }
        assert_eq!(prover.f().len(), 2);
        assert_eq!(prover.combined_basis.len(), 2);

        let msgs = prover.transcript().to_vec();
        assert_eq!(msgs.len(), n);
        let r_last = ch.sample_f128();
        let mut t_r = h;
        for (i, msg) in msgs.iter().enumerate() {
            let quad = RoundQuad::from_msg(*msg, t_r);
            assert_eq!(
                quad.eval(F128::ZERO) + quad.eval(F128::ONE),
                t_r,
                "round {i}: q(0)+q(1) != T_r"
            );
            let r_i = if i < n - 1 { ris[i] } else { r_last };
            t_r = quad.eval(r_i);
        }
        let one_plus_r = F128::ONE + r_last;
        let f_resid = prover.f()[0] * one_plus_r + prover.f()[1] * r_last;
        let b_resid = prover.combined_basis[0] * one_plus_r + prover.combined_basis[1] * r_last;
        assert_eq!(f_resid * b_resid, t_r, "residual inner product != t_r");
    }

    #[test]
    fn stateful_sumcheck_introduce_glue() {
        use crate::challenger::Challenger;
        let n = 5;
        let len = 1usize << n;
        let mk = |seed: u64| -> Vec<F128> {
            (0..len)
                .map(|i| F128::new(seed.wrapping_mul(i as u64 + 1), seed ^ (i as u64) << 7))
                .collect()
        };
        let f = mk(0xC1);
        let b1 = mk(0xB1);
        let b2 = mk(0xB2);
        let h1: F128 = f
            .iter()
            .zip(b1.iter())
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |a, v| a + v);

        let (mut prover, _first) = SumcheckProver::new(f.clone(), b1.clone(), h1);
        let mut ch = crate::challenger::RandomChallenger::new(0xBEEF);

        let r0 = ch.sample_f128();
        prover.fold(r0);

        let mut b2_folded = b2.clone();
        partial_eval_lsb_one(&mut b2_folded, r0);

        let h2_folded: F128 = b2_folded
            .iter()
            .zip(prover.f().iter())
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |a, v| a + v);
        prover.introduce_new(b2_folded.clone(), h2_folded);
        let alpha = ch.sample_f128();
        prover.glue(alpha);

        let mut ris = vec![r0];
        for _ in 0..(n - 2) {
            let r = ch.sample_f128();
            ris.push(r);
            prover.fold(r);
        }
        let r_last = ch.sample_f128();
        ris.push(r_last);
        assert_eq!(prover.f().len(), 2);

        let msgs = prover.transcript().to_vec();

        assert_eq!(msgs.len(), 1 + 1 + 1 + (n - 2));

        let mut t_r = h1;

        let q0 = RoundQuad::from_msg(msgs[0], t_r);
        assert_eq!(q0.eval(F128::ZERO) + q0.eval(F128::ONE), t_r);
        t_r = q0.eval(r0);

        let q1 = RoundQuad::from_msg(msgs[1], t_r);
        assert_eq!(q1.eval(F128::ZERO) + q1.eval(F128::ONE), t_r);

        let q_intro = RoundQuad::from_msg(msgs[2], h2_folded);
        assert_eq!(
            q_intro.eval(F128::ZERO) + q_intro.eval(F128::ONE),
            h2_folded
        );

        let combined = RoundQuad::fold(&q1, &q_intro, alpha);
        t_r += alpha * h2_folded;

        assert_eq!(combined.eval(F128::ZERO) + combined.eval(F128::ONE), t_r);

        let mut running = combined;

        assert_eq!(msgs.len(), n + 1);
        for (k, &r) in ris.iter().enumerate().skip(1).take(n - 2) {
            t_r = running.eval(r);
            let msg = msgs[2 + k];
            running = RoundQuad::from_msg(msg, t_r);
            assert_eq!(
                running.eval(F128::ZERO) + running.eval(F128::ONE),
                t_r,
                "post-glue round k={k}"
            );
        }

        t_r = running.eval(r_last);

        let one_plus_r = F128::ONE + r_last;
        let f_resid = prover.f()[0] * one_plus_r + prover.f()[1] * r_last;

        let combined_resid =
            prover.combined_basis[0] * one_plus_r + prover.combined_basis[1] * r_last;
        assert_eq!(
            f_resid * combined_resid,
            t_r,
            "residual inner product != t_r"
        );
    }

    #[test]
    fn induce_sumcheck_poly_consistent_with_codeword() {
        use crate::challenger::Challenger;
        let log_msg = 4;
        let log_inv_rate = 1;
        let msg_cols = 1usize << log_msg;
        let block_len = msg_cols << log_inv_rate;

        let mut ch = crate::challenger::RandomChallenger::new(0xF00DCAFE);
        let msg: Vec<F128> = (0..msg_cols).map(|_| ch.sample_f128()).collect();

        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let mut codeword = vec![F128::ZERO; block_len];
        codeword[..msg_cols].copy_from_slice(&msg);
        ntt.forward_transform(&mut codeword);

        let num_queries = 6;
        let mut queries: Vec<usize> = Vec::new();
        while queries.len() < num_queries {
            let q = (ch.sample_f128().lo as usize) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&q| vec![codeword[q]]).collect();
        let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
        let sks_vks = eval_sk_at_vks(log_msg);

        let (basis_poly, enforced_sum) =
            induce_sumcheck_poly(log_msg, &sks_vks, &opened_rows, &[], &queries, &alpha);
        assert_eq!(basis_poly.len(), msg_cols);

        let alpha_weights: Vec<F128> = crate::lincheck::build_eq_table(&alpha)
            .into_iter()
            .take(queries.len())
            .collect();
        let expected: F128 = queries
            .iter()
            .zip(alpha_weights.iter())
            .map(|(&q, &w)| w * codeword[q])
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(enforced_sum, expected, "enforced_sum != eq(α)-batched c[q]");

        let inner: F128 = msg
            .iter()
            .zip(basis_poly.iter())
            .map(|(&m, &b)| m * b)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(inner, enforced_sum, "msg · basis_poly != enforced_sum");
    }

    #[test]
    fn induce_sumcheck_poly_via_ntt_matches_dense() {
        use crate::challenger::Challenger;
        let shapes = [
            (4usize, 1usize, 0usize, 6usize),
            (3, 1, 2, 5),
            (6, 2, 3, 30),
            (10, 1, 6, 218),
            (8, 3, 3, 71),
            (5, 5, 3, 43),
            (0, 2, 1, 3),
        ];
        for (si, &(log_msg, log_inv_rate, log_int, n_queries)) in shapes.iter().enumerate() {
            let block_len = 1usize << (log_msg + log_inv_rate);
            let num_interleaved = 1usize << log_int;
            let mut ch = crate::challenger::RandomChallenger::new(0xA11CE ^ si as u64);
            let mut queries: Vec<usize> = Vec::new();
            while queries.len() < n_queries.min(block_len) {
                let q = (ch.sample_f128().lo as usize) % block_len;
                if !queries.contains(&q) {
                    queries.push(q);
                }
            }
            let nq = queries.len();
            let opened_rows: Vec<Vec<F128>> = (0..nq)
                .map(|_| ch.sample_f128_vec(num_interleaved))
                .collect();
            let v_challenges = ch.sample_f128_vec(log_int);
            let alpha = ch.sample_f128_vec(ceil_log2(nq.max(1)));
            let sks_vks = eval_sk_at_vks(log_msg);

            let dense = induce_sumcheck_poly(
                log_msg,
                &sks_vks,
                &opened_rows,
                &v_challenges,
                &queries,
                &alpha,
            );
            let ntt = induce_sumcheck_poly_via_ntt(
                log_msg,
                log_inv_rate,
                &opened_rows,
                &v_challenges,
                &queries,
                &alpha,
            );
            assert_eq!(ntt.1, dense.1, "shape {si}: enforced_sum");
            assert_eq!(ntt.0, dense.0, "shape {si}: basis_poly");
        }
    }

    #[test]
    fn transpose_sparse_matches_dense() {
        use crate::challenger::Challenger;
        for &log_d in &[6usize, 11, 12, 14, 16, 18] {
            for &nq in &[1usize, 5, 43, 218] {
                let n = 1usize << log_d;
                let nq = nq.min(n);
                let mut ch =
                    crate::challenger::RandomChallenger::new(0xC0DE ^ (log_d * 131 + nq) as u64);
                let ntt = AdditiveNttF128::standard(log_d);
                let mut positions: Vec<usize> = Vec::new();
                let mut values: Vec<F128> = Vec::new();
                while positions.len() < nq {
                    let p = (ch.sample_f128().lo as usize) % n;
                    if !positions.contains(&p) {
                        positions.push(p);
                        values.push(ch.sample_f128());
                    }
                }

                let mut dense = vec![F128::ZERO; n];
                for (&p, &v) in positions.iter().zip(&values) {
                    dense[p] += v;
                }
                transpose_forward_ntt(&ntt, &mut dense, log_d);
                let sparse = transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d);
                assert_eq!(sparse, dense, "log_d={log_d}, nq={nq}");
            }
        }
    }

    #[test]
    fn induce_sumcheck_poly_with_interleaving_and_v_challenges() {
        use crate::challenger::Challenger;
        let log_msg = 3;
        let log_interleaved = 2;
        let log_inv_rate = 1;
        let msg_cols = 1usize << log_msg;
        let num_interleaved = 1usize << log_interleaved;
        let block_len = msg_cols << log_inv_rate;
        let poly_len = msg_cols * num_interleaved;

        let mut ch = crate::challenger::RandomChallenger::new(0xDEAD_BEEF);

        let poly: Vec<F128> = (0..poly_len).map(|_| ch.sample_f128()).collect();

        let v_challenges: Vec<F128> = (0..log_interleaved).map(|_| ch.sample_f128()).collect();
        let f_folded = partial_eval_lsb(&poly, &v_challenges);
        assert_eq!(f_folded.len(), msg_cols);

        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let w = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
        assert_eq!(w.block_len, block_len);

        let num_queries = 5;
        let mut queries: Vec<usize> = Vec::new();
        while queries.len() < num_queries {
            let q = (ch.sample_f128().lo as usize) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&q| w.row(q).to_vec()).collect();

        let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
        let sks_vks = eval_sk_at_vks(log_msg);
        let (basis_poly, enforced_sum) = induce_sumcheck_poly(
            log_msg,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            &alpha,
        );

        let inner: F128 = f_folded
            .iter()
            .zip(basis_poly.iter())
            .map(|(&m, &b)| m * b)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(
            inner, enforced_sum,
            "folded-msg · basis_poly != enforced_sum (interleaved + v_challenges path)"
        );
    }

    #[test]
    fn ligerito_r1_roundtrip_accepts() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let num_queries = 0;

        let mut rng = crate::challenger::RandomChallenger::new(0xCAFE_F00D);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();

        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let grinding_bits = vec![0; log_inv_rates.len()];
        let prover_cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let verifier_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let _ = num_queries;

        let mut p_ch = crate::challenger::FsChallenger::new(b"test");
        let proof = recursive_prover(&prover_cfg, &poly, &z, v, &mut p_ch);

        let mut v_ch = crate::challenger::FsChallenger::new(b"test");
        let ok = recursive_verifier(&verifier_cfg, &proof, &z, v, &mut v_ch);
        assert!(ok, "verifier rejected a valid proof");
    }

    fn size_breakdown_at(
        log_n: usize,
        initial_k: usize,
        recursive_ks: Vec<usize>,
        log_inv_rates: Vec<usize>,
    ) -> usize {
        use crate::challenger::Challenger;
        use std::time::Instant;
        assert_eq!(log_inv_rates.len(), recursive_ks.len() + 1);

        let r = recursive_ks.len();
        let mut recursive_log_msg_cols = Vec::with_capacity(r);
        let mut n_running = log_n - initial_k;
        for &k in &recursive_ks {
            assert!(n_running >= k);
            recursive_log_msg_cols.push(n_running - k);
            n_running -= k;
        }

        let mut rng = crate::challenger::RandomChallenger::new(0xBEEFCAFE);
        let queries_per_level: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        eprintln!(
            "log_n={log_n}  initial_k={initial_k}  ks={:?}  log_inv_rates={:?}  queries={:?}",
            recursive_ks, log_inv_rates, queries_per_level
        );
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);
        drop(eq);

        let grinding_bits = vec![0; log_inv_rates.len()];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: recursive_ks.clone(),
            queries: queries_per_level.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; r + 1],
            ood_samples: vec![0; r + 1],
            merkle_hash: Default::default(),
        };
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols,
            recursive_ks: recursive_ks.clone(),
            queries: queries_per_level,
            grinding_bits,
            fold_grinding_bits: vec![0; r + 1],
            ood_samples: vec![0; r + 1],
            merkle_hash: Default::default(),
        };

        let mut best = std::time::Duration::from_secs(3600);
        let mut proof = {
            let mut p_ch = crate::challenger::FsChallenger::new(b"size-test");
            recursive_prover(&cfg, &poly, &z, v, &mut p_ch)
        };
        for _ in 0..3 {
            let mut p_ch = crate::challenger::FsChallenger::new(b"size-test");
            let t = Instant::now();
            proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);
            let el = t.elapsed();
            if el < best {
                best = el;
            }
        }
        eprintln!(
            "--- Ligerito proof: prover {:.2?} (best of 3), size: ---",
            best
        );
        proof.print_size_breakdown();

        let mut v_ch = crate::challenger::FsChallenger::new(b"size-test");
        assert!(recursive_verifier(&v_cfg, &proof, &z, v, &mut v_ch));
        proof.size_bytes()
    }

    #[test]
    fn ligerito_size_breakdown_m20_uniform_rate() {
        size_breakdown_at(20, 4, vec![4, 3], vec![1, 1, 1]);
    }

    #[test]
    fn ligerito_size_breakdown_m20_decreasing_rate() {
        size_breakdown_at(20, 4, vec![4, 3], vec![1, 2, 4]);
    }

    #[test]
    fn ligerito_size_breakdown_m20_decreasing_rate_thin() {
        size_breakdown_at(20, 4, vec![3, 3, 3], vec![1, 2, 3, 4]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_aggressive() {
        size_breakdown_at(24, 3, vec![3, 3, 3, 3, 3], vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_uniform_rate() {
        size_breakdown_at(24, 5, vec![5, 4, 3], vec![1, 1, 1, 1]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_decreasing_rate() {
        size_breakdown_at(24, 4, vec![4, 4, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m22() {
        size_breakdown_at(22, 4, vec![4, 4, 3], vec![1, 2, 3, 4]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m22_initial_k6() {
        size_breakdown_at(22, 6, vec![3, 3, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m23() {
        size_breakdown_at(23, 4, vec![4, 4, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    fn multi_proof_num_siblings(positions: &[usize], num_leaves: usize) -> usize {
        let mut active: Vec<usize> = positions.to_vec();
        active.sort_unstable();
        active.dedup();
        let mut sib_count = 0usize;
        let mut level_len = num_leaves;
        while level_len > 1 {
            let mut next = Vec::with_capacity(active.len());
            let mut i = 0;
            while i < active.len() {
                let p = active[i];
                let sib_active = i + 1 < active.len() && active[i + 1] == (p ^ 1);
                if sib_active {
                    i += 2;
                } else {
                    sib_count += 1;
                    i += 1;
                }
                next.push(p >> 1);
            }
            active = next;
            level_len >>= 1;
        }
        sib_count
    }

    fn estimate_size_at(
        log_n: usize,
        initial_k: usize,
        recursive_ks: Vec<usize>,
        log_inv_rates: Vec<usize>,
    ) -> usize {
        const ELEM: usize = core::mem::size_of::<F128>();
        assert_eq!(log_inv_rates.len(), recursive_ks.len() + 1);
        let r = recursive_ks.len();
        let kb = |b: usize| {
            if b >= 1024 * 1024 {
                format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
            } else if b >= 1024 {
                format!("{:.1} KB", b as f64 / 1024.0)
            } else {
                format!("{} B", b)
            }
        };

        let mut log_num_interleaved: Vec<usize> = vec![initial_k];
        log_num_interleaved.extend_from_slice(&recursive_ks);
        let mut log_msg_cols: Vec<usize> = Vec::with_capacity(r + 1);
        let mut n_running = log_n;
        for i in 0..=r {
            assert!(
                n_running >= log_num_interleaved[i],
                "config infeasible at commit {i}: dim {n_running} < lanes {}",
                log_num_interleaved[i]
            );
            log_msg_cols.push(n_running - log_num_interleaved[i]);
            n_running -= log_num_interleaved[i];
        }
        let yr_log_n = n_running;
        let queries_per_level: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let log_block_len: Vec<usize> = log_msg_cols
            .iter()
            .zip(log_inv_rates.iter())
            .map(|(&m, &r)| m + r)
            .collect();

        eprintln!(
            "m={log_n}  initial_k={initial_k}  ks={:?}  rates={:?}  queries={:?}  yr_log={yr_log_n}",
            recursive_ks, log_inv_rates, queries_per_level
        );

        let mut ch = crate::challenger::FsChallenger::new(b"estimate");
        let mut total_opened = 0usize;
        let mut total_merkle = 0usize;
        for i in 0..=r {
            let bl = 1usize << log_block_len[i];
            let qn = queries_per_level[i];
            if qn > bl {
                eprintln!(
                    "  INFEASIBLE at commit {i}: queries ({qn}) > block_len ({bl}). Pick a higher rate (smaller bl) or smaller queries."
                );
                return usize::MAX;
            }
            let qs = sample_distinct_queries(&mut ch, bl, qn);
            let sib = multi_proof_num_siblings(&qs, bl);
            let opened = qn * (1usize << log_num_interleaved[i]) * ELEM;
            let merkle = sib * 32;
            let label = if i == 0 {
                "L0 (initial)"
            } else if i == r {
                "L{} (final)"
            } else {
                "L{} (recursive)"
            };
            eprintln!(
                "  {label} [bl=2^{}, lanes=2^{}, q={qn}]: opened={}  merkle={} ({} sibs)",
                log_block_len[i],
                log_num_interleaved[i],
                kb(opened),
                kb(merkle),
                sib,
            );
            total_opened += opened;
            total_merkle += merkle;
        }
        let yr_b = (1usize << yr_log_n) * ELEM;
        let roots_b = (r + 1) * 32;

        let sumcheck_msgs = 1 + r + recursive_ks.iter().sum::<usize>();
        let tx_b = sumcheck_msgs * 2 * ELEM;
        let total = total_opened + total_merkle + yr_b + roots_b + tx_b;
        eprintln!(
            "  TOTALS: opened={}  merkle={}  yr={}  roots={}  transcript={}  → GRAND={}",
            kb(total_opened),
            kb(total_merkle),
            kb(yr_b),
            kb(roots_b),
            kb(tx_b),
            kb(total),
        );
        total
    }

    #[test]
    fn estimator_matches_actual_m20() {
        let estimated = estimate_size_at(20, 4, vec![4, 3], vec![1, 2, 4]);

        let actual = size_breakdown_at(20, 4, vec![4, 3], vec![1, 2, 4]);
        let diff = estimated.abs_diff(actual);
        eprintln!("estimator={estimated}  actual={actual}  diff={diff}");

        assert!(
            diff < actual / 20,
            "estimator drift too large: {diff} bytes"
        );
    }

    #[test]
    fn estimate_ligerito_m29() {
        eprintln!("\n=== Ligerito m=29 — decreasing rate (the real Ligerito design) ===");

        estimate_size_at(29, 4, vec![4, 4, 4, 4, 3], vec![1, 2, 3, 4, 5, 6]);

        eprintln!(
            "\n=== Ligerito m=29 — uniform rate 1/2 (basefold-style baseline, infeasible at deepest level) ==="
        );

        estimate_size_at(29, 4, vec![4, 4, 4, 4, 3], vec![1, 1, 1, 1, 1, 1]);

        eprintln!("\n=== Ligerito m=29 — uniform rate, shallower (R=2) ===");

        estimate_size_at(29, 4, vec![10, 10], vec![1, 1, 1]);

        eprintln!("\n=== Ligerito m=29 — thinner lanes ===");
        estimate_size_at(
            29,
            3,
            vec![3, 3, 3, 3, 3, 3, 3],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        );
    }

    #[test]
    fn estimate_ligerito_m30() {
        eprintln!("\n=== Ligerito m=30 — decreasing rate ===");
        estimate_size_at(30, 4, vec![4, 4, 4, 4, 4, 3], vec![1, 2, 3, 4, 5, 6, 7]);

        eprintln!("\n=== Ligerito m=30 — thinner lanes ===");
        estimate_size_at(
            30,
            3,
            vec![3, 3, 3, 3, 3, 3, 3, 3],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
    }

    #[test]
    fn estimate_ligerito_m29_initial_k6() {
        eprintln!(
            "\n=== Ligerito m=29 — initial_k=6 (matches basefold's 64-lane initial leaves) ==="
        );

        eprintln!("\n  Config A: thin recursive lanes, aggressive rate decrease");
        estimate_size_at(29, 6, vec![3, 3, 3, 3, 3, 2], vec![1, 2, 3, 4, 5, 6, 7]);

        eprintln!("\n  Config B: medium recursive lanes, fewer levels");
        estimate_size_at(29, 6, vec![4, 4, 4, 3, 3], vec![1, 2, 3, 4, 5, 6]);

        eprintln!("\n  Config C: 2x6-bit recursive lanes (= basefold's epoch leaves)");
        estimate_size_at(29, 6, vec![6, 6, 4, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn estimate_ligerito_m30_initial_k6() {
        eprintln!("\n=== Ligerito m=30 — initial_k=6 ===");
        eprintln!("\n  Config A: thin recursive lanes");
        estimate_size_at(
            30,
            6,
            vec![3, 3, 3, 3, 3, 3, 2],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        );

        eprintln!("\n  Config B: medium");
        estimate_size_at(30, 6, vec![4, 4, 4, 4, 3, 3], vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn ligerito_r2_roundtrip_accepts() {
        use crate::challenger::Challenger;
        let log_n = 18;
        let initial_k = 3;
        let k_0 = 3;
        let k_1 = 2;
        let log_inv_rate = 1;
        let num_queries = 0;

        let mut rng = crate::challenger::RandomChallenger::new(0xABCD_1234);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate; 3];
        let _ = num_queries;
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 2,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0, log_n - initial_k - k_0 - k_1],
            recursive_ks: vec![k_0, k_1],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 3],
            ood_samples: vec![0; 3],
            merkle_hash: Default::default(),
        };
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 2,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0, log_n - initial_k - k_0 - k_1],
            recursive_ks: vec![k_0, k_1],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 3],
            ood_samples: vec![0; 3],
            merkle_hash: Default::default(),
        };

        let mut p_ch = crate::challenger::FsChallenger::new(b"test-r2");
        let proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);
        assert_eq!(proof.recursive_roots.len(), 2);
        assert_eq!(proof.recursive_proofs.len(), 1);

        let mut v_ch = crate::challenger::FsChallenger::new(b"test-r2");
        let ok = recursive_verifier(&v_cfg, &proof, &z, v, &mut v_ch);
        assert!(ok, "R=2 verifier rejected valid proof");
    }

    #[test]
    fn ligerito_proof_bincode_roundtrip() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let mut rng = crate::challenger::RandomChallenger::new(0xDEED_F00D);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut p_ch = crate::challenger::FsChallenger::new(b"serde");
        let proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        let bytes = bincode::serialize(&proof).expect("serialize");
        let proof2: LigeritoProof = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(proof, proof2);
        eprintln!("LigeritoProof bincode size: {} bytes", bytes.len());
    }

    #[test]
    fn recursive_prover_with_basis_roundtrip_single_claim() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xBA51_CAFE);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"basis-test");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"basis-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(ok, "basis-based verifier rejected valid proof");
    }

    #[test]
    fn induce_sumcheck_evaluate_at_residual_matches_dense() {
        use crate::challenger::Challenger;
        let log_msg_cols = 6;
        let yr_log_n = 2;
        let prefix_len = log_msg_cols - yr_log_n;
        let num_interleaved = 4;
        let log_num_interleaved = 2;
        let num_queries = 5;

        let mut rng = crate::challenger::RandomChallenger::new(0x2017_5052);
        let queries: Vec<usize> = (0..num_queries).map(|i| (i * 7 + 3) % (1 << 8)).collect();
        let opened_rows: Vec<Vec<F128>> = (0..num_queries)
            .map(|_| (0..num_interleaved).map(|_| rng.sample_f128()).collect())
            .collect();
        let v_challenges: Vec<F128> = (0..log_num_interleaved)
            .map(|_| rng.sample_f128())
            .collect();
        let alpha: Vec<F128> = (0..ceil_log2(num_queries))
            .map(|_| rng.sample_f128())
            .collect();
        let ris_for_basis: Vec<F128> = (0..prefix_len).map(|_| rng.sample_f128()).collect();
        let sks_vks = eval_sk_at_vks(log_msg_cols);

        let (basis_dense, dense_enforced_sum) = induce_sumcheck_poly(
            log_msg_cols,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            &alpha,
        );
        let dense_residual = partial_eval_lsb(&basis_dense, &ris_for_basis);

        let succinct_enforced_sum =
            induce_sumcheck_enforced_sum(&opened_rows, &v_challenges, &queries, &alpha);
        let succinct_residual = induce_sumcheck_evaluate_at_residual(
            log_msg_cols,
            &sks_vks,
            &queries,
            &alpha,
            &ris_for_basis,
            yr_log_n,
        );

        assert_eq!(
            succinct_enforced_sum, dense_enforced_sum,
            "enforced_sum mismatch"
        );
        assert_eq!(
            succinct_residual.len(),
            dense_residual.len(),
            "residual length mismatch"
        );
        for (i, (s, d)) in succinct_residual
            .iter()
            .zip(dense_residual.iter())
            .enumerate()
        {
            assert_eq!(s, d, "residual mismatch at y={i}");
        }
    }

    #[test]
    fn final_level_binding_pins_yr_to_committed_codeword() {
        use crate::challenger::Challenger;
        let log_msg_cols = 5;
        let log_inv_rate = 1;
        let num_queries = 20;
        let msg_cols = 1usize << log_msg_cols;
        let block_len = msg_cols << log_inv_rate;

        let mut rng = crate::challenger::RandomChallenger::new(0xB19D_1235);

        let yr: Vec<F128> = (0..msg_cols).map(|_| rng.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_msg_cols + log_inv_rate);
        let wtns = ligero_commit(&yr, log_msg_cols, 0, log_inv_rate, &ntt, HashKind::Sha256);

        let mut queries: Vec<usize> = Vec::new();
        let mut q = 1usize;
        while queries.len() < num_queries {
            q = (q * 73 + 41) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&p| wtns.row(p).to_vec()).collect();

        let level_rs: Vec<F128> = Vec::new();
        let alpha: Vec<F128> = (0..ceil_log2(num_queries))
            .map(|_| rng.sample_f128())
            .collect();

        let enforced_sum = induce_sumcheck_enforced_sum(&opened_rows, &level_rs, &queries, &alpha);
        let sks_vks = eval_sk_at_vks(log_msg_cols);
        let induced_basis = induce_sumcheck_evaluate_at_residual(
            log_msg_cols,
            &sks_vks,
            &queries,
            &alpha,
            &[],
            log_msg_cols,
        );
        let inner = |v: &[F128]| -> F128 {
            v.iter()
                .zip(induced_basis.iter())
                .map(|(&a, &b)| a * b)
                .fold(F128::ZERO, |s, x| s + x)
        };

        assert_eq!(
            inner(&yr),
            enforced_sum,
            "honest yr must satisfy ⟨yr, induced_basis⟩ == enforced_sum"
        );

        let jnz = induced_basis
            .iter()
            .position(|b| !b.is_zero())
            .expect("induced basis must not be identically zero");
        let mut yr_bad = yr.clone();
        yr_bad[jnz] += F128::ONE;
        assert_ne!(
            inner(&yr_bad),
            enforced_sum,
            "a forged yr must break the final-level proximity tie"
        );
    }

    #[test]
    fn recursive_verifier_with_basis_succinct_matches_dense() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0x52CC_2017);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"succ-cmp");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let mut v_ch = crate::challenger::FsChallenger::new(b"succ-cmp");
        let dense_ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(dense_ok, "dense verifier must accept");

        let mut v_ch2 = crate::challenger::FsChallenger::new(b"succ-cmp");
        let eval_b_residual = |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
            let yr_len = 1usize << yr_log_n;
            let mut point = ris.to_vec();
            point.resize(ris.len() + yr_log_n, F128::ZERO);
            (0..yr_len)
                .map(|y| {
                    for j in 0..yr_log_n {
                        point[ris.len() + j] = if (y >> j) & 1 == 1 {
                            F128::ONE
                        } else {
                            F128::ZERO
                        };
                    }
                    crate::zerocheck::multilinear::eq_eval(&z, &point)
                })
                .collect()
        };
        let succ_ok = recursive_verifier_with_basis_succinct(
            &v_cfg,
            &proof,
            log_n,
            target,
            &initial_root,
            eval_b_residual,
            &mut v_ch2,
        );
        assert!(succ_ok, "succinct verifier must accept");
    }

    fn ood_test_configs(
        log_n: usize,
        initial_k: usize,
        ks: &[usize],
        ood_samples: Vec<usize>,
        fold_grinding_bits: Vec<usize>,
    ) -> (ProverConfig, VerifierConfig) {
        let r = ks.len();
        let log_inv_rates: Vec<usize> = (0..=r).map(|i| 1 + i).collect();
        let mut recursive_log_msg_cols = Vec::new();
        let mut dim = log_n - initial_k;
        for &k in ks {
            recursive_log_msg_cols.push(dim - k);
            dim -= k;
        }
        let queries = vec![20usize; r + 1];
        let grinding_bits = vec![0usize; r + 1];
        let p = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: ks.to_vec(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: fold_grinding_bits.clone(),
            ood_samples: ood_samples.clone(),
            merkle_hash: Default::default(),
        };
        let v = VerifierConfig {
            log_inv_rates,
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols,
            recursive_ks: ks.to_vec(),
            queries,
            grinding_bits,
            fold_grinding_bits,
            ood_samples,
            merkle_hash: Default::default(),
        };
        (p, v)
    }

    #[test]
    fn ligerito_ood_and_fold_grinding_roundtrip_and_tamper() {
        use crate::challenger::Challenger;
        let log_n = 12;
        let initial_k = 2;
        let ks = [2usize, 2];

        let (p_cfg, v_cfg) = ood_test_configs(log_n, initial_k, &ks, vec![0, 2, 2], vec![3, 3, 3]);

        let mut rng = crate::challenger::RandomChallenger::new(0x00D_7E57);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"ood-test");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        assert_eq!(proof.ood_values.len(), 4, "2 OOD samples each at L1 and L2");

        assert_eq!(proof.fold_grinding_nonces.len(), initial_k + ks[0] + ks[1]);

        let dense = |proof: &LigeritoProof| {
            let mut ch = crate::challenger::FsChallenger::new(b"ood-test");
            recursive_verifier_with_basis(&v_cfg, proof, &b, target, &initial_root, &mut ch)
        };
        let eval_b_residual = {
            let z = z.clone();
            move |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
                let yr_len = 1usize << yr_log_n;
                let mut point = ris.to_vec();
                point.resize(ris.len() + yr_log_n, F128::ZERO);
                (0..yr_len)
                    .map(|y| {
                        for j in 0..yr_log_n {
                            point[ris.len() + j] = if (y >> j) & 1 == 1 {
                                F128::ONE
                            } else {
                                F128::ZERO
                            };
                        }
                        crate::zerocheck::multilinear::eq_eval(&z, &point)
                    })
                    .collect()
            }
        };
        let succinct = |proof: &LigeritoProof| {
            let mut ch = crate::challenger::FsChallenger::new(b"ood-test");
            recursive_verifier_with_basis_succinct(
                &v_cfg,
                proof,
                log_n,
                target,
                &initial_root,
                &eval_b_residual,
                &mut ch,
            )
        };

        assert!(dense(&proof), "dense verifier must accept OOD proof");
        assert!(succinct(&proof), "succinct verifier must accept OOD proof");

        let mut bad_ood = proof.clone();
        bad_ood.ood_values[0] += F128::ONE;
        assert!(!dense(&bad_ood), "dense must reject tampered OOD value");
        assert!(
            !succinct(&bad_ood),
            "succinct must reject tampered OOD value"
        );

        let mut bad_nonce = proof.clone();
        bad_nonce.fold_grinding_nonces[0] ^= 0xDEAD_BEEF;
        assert!(!dense(&bad_nonce), "dense must reject tampered fold nonce");
        assert!(
            !succinct(&bad_nonce),
            "succinct must reject tampered fold nonce"
        );
    }

    #[test]
    fn ligerito_fast_profile_m22_roundtrip() {
        use crate::challenger::Challenger;
        let m = 22usize;
        let log_n = m - crate::pcs::LOG_PACKING;
        let initial_k = 6;
        let p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast prover config");
        let v_cfg = verifier_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast verifier config");

        assert!(p_cfg.ood_samples.iter().skip(1).any(|&s| s > 0));
        assert!(p_cfg.fold_grinding_bits.iter().any(|&g| g > 0));

        let mut rng = crate::challenger::RandomChallenger::new(0xFA57_0022);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"m22-fast");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly,
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let mut v_ch = crate::challenger::FsChallenger::new(b"m22-fast");
        assert!(
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch),
            "m22 fast profile proof must verify"
        );
    }

    #[test]
    fn ligerito_m22_roundtrip_under_blake3() {
        use crate::challenger::Challenger;
        let m = 22usize;
        let log_n = m - crate::pcs::LOG_PACKING;
        let initial_k = 6;
        let mut p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast prover config");
        let mut v_cfg = verifier_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast verifier config");

        assert_eq!(p_cfg.merkle_hash, HashKind::Sha256);
        p_cfg.merkle_hash = HashKind::Blake3;
        v_cfg.merkle_hash = HashKind::Blake3;

        let mut rng = crate::challenger::RandomChallenger::new(0xB1A5_E300);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            HashKind::Blake3,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly,
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let mut v_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        assert!(
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch),
            "blake3 Merkle proof must verify"
        );

        let mut wrong_cfg = v_cfg.clone();
        wrong_cfg.merkle_hash = HashKind::Sha256;
        let mut w_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        assert!(
            !recursive_verifier_with_basis(
                &wrong_cfg,
                &proof,
                &b,
                target,
                &initial_root,
                &mut w_ch
            ),
            "a sha256-configured verifier must reject a blake3 proof"
        );
    }

    #[test]
    fn ligerito_m22_roundtrip_over_hash_matrix() {
        use crate::challenger::Challenger;
        const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];
        let log_n = 22usize - crate::pcs::LOG_PACKING;
        let initial_k = 6;

        for merkle_hash in KINDS {
            for fs_hash in KINDS {
                let mut p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast).unwrap();
                let mut v_cfg =
                    verifier_config_for(log_n, initial_k, LigeritoProfile::Fast).unwrap();
                p_cfg.merkle_hash = merkle_hash;
                v_cfg.merkle_hash = merkle_hash;

                let mut rng = crate::challenger::RandomChallenger::new(0x4A11_0000);
                let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
                let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
                let b = build_eq_table(&z);
                let target: F128 = poly
                    .iter()
                    .zip(b.iter())
                    .map(|(&a, &c)| a * c)
                    .fold(F128::ZERO, |a, x| a + x);

                let log_msg_cols_0 = log_n - initial_k;
                let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
                let wtns_0 =
                    ligero_commit(&poly, log_msg_cols_0, initial_k, 1, &ntt_0, merkle_hash);
                let initial_root = wtns_0.root();

                let mut p_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", fs_hash);
                let proof = recursive_prover_with_basis(
                    &p_cfg,
                    poly,
                    b.clone(),
                    target,
                    &wtns_0.mat,
                    &wtns_0.tree,
                    &mut p_ch,
                );

                let mut v_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", fs_hash);
                assert!(
                    recursive_verifier_with_basis(
                        &v_cfg,
                        &proof,
                        &b,
                        target,
                        &initial_root,
                        &mut v_ch
                    ),
                    "merkle={merkle_hash} fs={fs_hash} must verify"
                );

                let other_fs = match fs_hash {
                    HashKind::Sha256 => HashKind::Blake3,
                    HashKind::Blake3 => HashKind::Sha256,
                };
                let mut w_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", other_fs);
                assert!(
                    !recursive_verifier_with_basis(
                        &v_cfg,
                        &proof,
                        &b,
                        target,
                        &initial_root,
                        &mut w_ch
                    ),
                    "merkle={merkle_hash}: an {other_fs} transcript must reject an {fs_hash} proof"
                );
            }
        }
    }

    #[test]
    fn recursive_prover_with_basis_roundtrip_batched_claims() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xBA51_BA51);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z1: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let z2: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let g1 = rng.sample_f128();
        let g2 = rng.sample_f128();
        let b1 = build_eq_table(&z1);
        let b2 = build_eq_table(&z2);
        let b: Vec<F128> = b1
            .iter()
            .zip(b2.iter())
            .map(|(&a, &c)| g1 * a + g2 * c)
            .collect();
        let v1: F128 = poly
            .iter()
            .zip(b1.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);
        let v2: F128 = poly
            .iter()
            .zip(b2.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);
        let target = g1 * v1 + g2 * v2;

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"batched");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"batched");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(ok, "batched-basis verifier rejected valid proof");
    }

    #[test]
    fn recursive_prover_with_l0_matches_full() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xACED_BEEF);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let mut p_ch = crate::challenger::FsChallenger::new(b"l0-test");
        let proof_a = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let mut wtns_0_external = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let mut p_ch_b = crate::challenger::FsChallenger::new(b"l0-test");
        let proof_b = recursive_prover_with_l0(
            &cfg,
            &poly,
            std::mem::take(&mut wtns_0_external.mat),
            std::mem::take(&mut wtns_0_external.tree),
            &z,
            v,
            &mut p_ch_b,
        );

        assert_eq!(proof_a.initial_root, proof_b.initial_root);
        assert_eq!(proof_a.recursive_roots, proof_b.recursive_roots);
        assert_eq!(proof_a.final_proof.yr, proof_b.final_proof.yr);
        assert_eq!(
            proof_a.sumcheck_transcript.len(),
            proof_b.sumcheck_transcript.len()
        );
        for (ma, mb) in proof_a
            .sumcheck_transcript
            .iter()
            .zip(proof_b.sumcheck_transcript.iter())
        {
            assert_eq!(ma.u_0, mb.u_0);
            assert_eq!(ma.u_2, mb.u_2);
        }

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"l0-test");
        assert!(recursive_verifier(&v_cfg, &proof_b, &z, v, &mut v_ch));
    }

    #[test]
    fn ligerito_r1_rejects_mutated_yr() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let num_queries = 0;

        let mut rng = crate::challenger::RandomChallenger::new(0xDEAD_BEEF);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let _ = num_queries;
        let prover_cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let verifier_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let mut p_ch = crate::challenger::FsChallenger::new(b"test-mut");
        let mut proof = recursive_prover(&prover_cfg, &poly, &z, v, &mut p_ch);

        proof.final_proof.yr[0] += F128::ONE;

        let mut v_ch = crate::challenger::FsChallenger::new(b"test-mut");
        let ok = recursive_verifier(&verifier_cfg, &proof, &z, v, &mut v_ch);
        assert!(!ok, "verifier accepted a proof with mutated yr");
    }

    #[test]
    fn ligero_commit_encoding_roundtrips_via_inv_ntt() {
        let log_msg = 4;
        let log_interleaved = 3;
        let log_inv_rate = 1;
        let msg_cols = 1 << log_msg;
        let num_interleaved = 1 << log_interleaved;
        let block_len = msg_cols << log_inv_rate;

        let poly: Vec<F128> = (0..num_interleaved * msg_cols)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0x9E3779B97F4A7C15),
                    0x1234 ^ i as u64,
                )
            })
            .collect();

        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let w = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
        assert_eq!(w.block_len, block_len);
        assert_eq!(w.num_interleaved, num_interleaved);
        assert_eq!(w.mat.len(), block_len * num_interleaved);

        for lane in 0..num_interleaved {
            let mut col: Vec<F128> = (0..block_len)
                .map(|pos| w.mat[pos * num_interleaved + lane])
                .collect();
            ntt.inverse_transform(&mut col);
            for col_idx in 0..msg_cols {
                assert_eq!(
                    col[col_idx],
                    poly[col_idx * num_interleaved + lane],
                    "lane {lane} col_idx {col_idx} mismatch",
                );
            }
            for col_idx in msg_cols..block_len {
                assert_eq!(
                    col[col_idx],
                    F128::ZERO,
                    "lane {lane} pad position {col_idx} not zero",
                );
            }
        }

        let w2 = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
        assert_eq!(w.root(), w2.root());
    }
}
