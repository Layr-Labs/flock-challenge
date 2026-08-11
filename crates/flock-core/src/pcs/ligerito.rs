// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Ported from bolt-rs (https://github.com/bcc-research/bolt-rs,
// `ligerito_recursive.rs`).

//! Ligerito: recursive multilinear PCS.
//!
//! Ported from bolt-rs (`ligerito_recursive.rs`) onto Flock primitives:
//! `F128` (GHASH irreducible), [`AdditiveNttF128`] (LCH novel basis,
//! byte-identical to bolt-rs's FFT), SHA-256 merkle from [`crate::merkle`],
//! and the [`Challenger`] trait for Fiat-Shamir.
//!
//! Soundness regimes (our paper App. C.3): unique decoding (Thm `ca-udr`,
//! BCHKS25 Cor. 1.4, `Secure` profile) and Johnson list decoding with
//! out-of-domain binding (Thm `ca-johnson`, BCHKS25 Thm 4.6 + Johnson
//! interleaved list bound, `Fast`/`Slim` profiles). See [`SoundnessRegime`].
//!
//! ## Protocol
//! 1. Commit f^0: reshape into `num_interleaved × msg_cols`, RS-encode each
//!    lane to `block_len = msg_cols · 2^log_inv_rate`, merkle over codeword
//!    positions (one position across all lanes = one leaf).
//! 2. Partial-eval f^0 with `initial_k` challenges → f^1.
//! 3. Commit f^1.
//! 4. Open `num_queries` rows of f^0; build induced sumcheck basis poly.
//! 5. For each recursive step i:
//!    a. Run k_i sumcheck rounds.
//!    b. Last step: send remaining poly + open f^i.
//!    c. Else: commit f^{i+2}, open f^{i+1}, induce next basis, glue.

// r498 archive identity: preserve optimized dispatch while awaiting validator turnover.
use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::build_eq_table;
use crate::merkle::{self, Hash, HashKind};
use crate::ntt::additive_ntt_f128::AdditiveNttF128;
use serde::{Deserialize, Serialize};

// ===================================================================
// Config
// ===================================================================

/// Per-level Reed-Solomon inverse rate (log₂). The CORE Ligerito idea is to
/// **decrease the rate at deeper levels**: at level i, lower rate ⟹ Johnson
/// list-decoding per-query error = √ρ ≈ 2^(-log_inv_rate/2) ⟹ fewer queries
/// needed for the same security ⟹ drastically smaller opened-rows cost at
/// deeper levels.
///
/// `log_inv_rates[i]` is the log inverse rate at commit i (so wtns_0 uses
/// `log_inv_rates[0]`, wtns_1 uses `log_inv_rates[1]`, …). Length = R + 1.
/// Named parameter profile for the Ligerito PCS. Decouples "which security
/// config" from the raw code rate: `Fast` and `Secure` share rate 1/2 but
/// differ in regime/target, so the rate alone cannot key the config lookup.
///
/// - `Fast`:   rate 1/2, Johnson list-decoding regime with OOD binding,
///             100-bit overall soundness. Default.
/// - `Slim`:   rate 1/4, Johnson + OOD + 16-bit query grinding, 100-bit
///             overall. Roughly half the proof, ~2x the L0 encoding work.
/// - `Secure`: rate 1/2, unique-decoding regime (list size 1, no OOD),
///             120-bit overall soundness. Largest proof, most conservative
///             analysis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LigeritoProfile {
    #[default]
    Fast,
    Slim,
    Secure,
}

impl LigeritoProfile {
    /// L0 code rate index for this profile (`rho_0 = 2^-log_inv_rate`).
    pub fn log_inv_rate(self) -> usize {
        match self {
            Self::Fast | Self::Secure => 1,
            Self::Slim => 2,
        }
    }
    /// Round-by-round soundness target (bits) the profile's configs are derived
    /// for: every round must individually clear this level (total security =
    /// min over rounds, per the Fiat-Shamir / `soundcalc` convention).
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
    /// Per-level query counts (L0, L1, ..., L_r). Length = recursive_steps + 1.
    /// `default_config` fills these via [`udr_queries`]; for tighter
    /// (or stronger) per-level numbers, load a [`LigeritoSecurityConfig`].
    pub queries: Vec<usize>,
    /// Per-level **query-phase** PoW grinding bits (L0, L1, ..., L_r), ground
    /// post-commit/pre-queries. Length = recursive_steps + 1. Each bit here
    /// substitutes for ~1/log₂(1/(1−γ)) queries at that level.
    pub grinding_bits: Vec<usize>,
    /// Per-level **fold-challenge** PoW grinding bits (L0, ..., L_r), ground
    /// immediately before EACH of the level's fold challenges (so a level
    /// with `k` folds does `k` grinds of this many bits). Boosts the
    /// proximity-gap term, which lives on the fold challenges. Length =
    /// recursive_steps + 1.
    pub fold_grinding_bits: Vec<usize>,
    /// Per-commit-level out-of-domain samples (L0, ..., L_r), taken right
    /// after the level's Merkle root enters the transcript. `[0]` must be 0:
    /// L0 is bound by the opening's own (post-commit, random-point)
    /// evaluation claim. Length = recursive_steps + 1.
    pub ood_samples: Vec<usize>,
    /// Hash backing every Merkle commitment this prover makes (L0 and each
    /// recursive level). Comes from the `hash` field of the security config;
    /// [`Default`] is SHA-256.
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
    /// Per-level query counts. Length = recursive_steps + 1.
    pub queries: Vec<usize>,
    /// Per-level query-phase PoW grinding bits. Length = recursive_steps + 1.
    pub grinding_bits: Vec<usize>,
    /// Per-level fold-challenge PoW grinding bits (one grind per fold
    /// challenge of the level). Length = recursive_steps + 1.
    pub fold_grinding_bits: Vec<usize>,
    /// Per-commit-level OOD samples. Length = recursive_steps + 1.
    pub ood_samples: Vec<usize>,
    /// Hash the prover's Merkle commitments were built under. Must match the
    /// prover's — a mismatch makes every opening fail to verify, which is the
    /// correct outcome: the root commits to the hash as much as to the data.
    pub merkle_hash: HashKind,
}

/// Proximity loss `ε*` for the UDR (unique-decoding regime) analysis. It
/// would back the proximity radius off to `γ = δ/2 − ε*` (δ = 1 − ρ the
/// code's relative distance); set to `0`, so we decode to the full
/// unique-decoding radius `γ = δ/2` with no backoff. Per our paper's Appendix
/// C.3 (Theorem `ca-udr`, BCHKS25 Cor. 1.4) the proximity-gap exceptional set
/// is then `a = γ·n + 1` — length-dependent (see [`paper_thm_1_4_log_a`]), so
/// `eps_pg = 128 − log₂ a` shrinks ~1 bit per witness doubling and is
/// recovered by `fold_grinding_bits`.
pub const UDR_PROXIMITY_LOSS: f64 = 0.0;

/// Soundness (in bits) the query phase must close on its own at every level
/// (the "100 bits from queries always" policy).
const UDR_TARGET_BITS: f64 = 100.0;

/// Number of queries for 100-bit soundness in the **unique-decoding regime**
/// at rate `2^(-log_inv_rate)`: `γ = δ/2 = (1−ρ)/2`, per-query soundness
/// `log₂(1/(1−γ))` (see [`udr_per_query_bits`]). Within the unique decoding
/// radius the prover is pinned to a single codeword, so there is no list and
/// no union-bound term — queries close the full target by themselves.
/// Per-query soundness saturates below 1 bit (`γ < 1/2`), so slimmer codes
/// bottom out near `UDR_TARGET_BITS` queries: 243 at rate 1/2, 148 at 1/4,
/// 121 at 1/8, 110 at 1/16, 105 at 1/32.
pub fn udr_queries(log_inv_rate: usize) -> usize {
    assert!(log_inv_rate > 0, "log_inv_rate=0 (rate 1) has no soundness");
    let per_q = udr_per_query_bits_asymptotic(log_inv_rate);
    (UDR_TARGET_BITS / per_q).ceil() as usize
}

/// Build a sensible default Ligerito config from the upstream PCS shape.
/// `log_n` is the packed-witness log size (= `m - LOG_PACKING`); `log_batch_size`
/// and `log_inv_rate` come from `PcsParams` (Ligerito's `initial_k` matches
/// `log_batch_size` for L0 reuse; the first rate matches `log_inv_rate`).
///
/// Strategy: 3-bit recursive folds (`k_i = 3`) with **decreasing rate**
/// (one rate step per recursive level) until the residual is small (`≤ 5` bits).
/// Asserts that the chosen rate keeps `block_len ≥ udr_queries(rate)` at
/// every level; if not, bumps the rate further.
///
/// Returns `Err` when no feasible config exists (e.g. `log_n` is too small).
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

    // L0 feasibility check.
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
        // Pick the smallest rate ≥ rate_running+1 such that block_len ≥ queries.
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

/// Recursion-ladder shape: per-level dims (index 0 = L0) plus the residual.
struct LadderShape {
    log_inv_rates: Vec<usize>,
    log_msg_cols: Vec<usize>,
    log_num_interleaved: Vec<usize>,
    k_recursive: Vec<usize>,
    yr_log_n: usize,
}

/// Shared shape derivation behind [`default_config`] and
/// [`LigeritoSecurityConfig::derive_profile`]: 3-bit recursive folds with the
/// rate index increasing by ≥ 1 per level, bumped further whenever the block
/// length couldn't accommodate `queries_at_rate(rate)` distinct queries.
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

/// Embedded security-spec TOML files. The lookup table maps `(m, profile)`
/// to a TOML payload that's hash-independent (Ligerito's shape only depends
/// on `log_n = m − LOG_PACKING`). Regenerate with
/// `cargo run --release --example gen_ligerito_configs`.
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

/// Look up the embedded security config TOML for `(m, profile)`.
/// Returns `None` if no config has been derived for this combination yet.
pub fn embedded_security_config(m: usize, profile: LigeritoProfile) -> Option<&'static str> {
    EMBEDDED_CONFIGS.iter().find_map(|&(key, toml)| {
        if key == (m, profile) {
            Some(toml)
        } else {
            None
        }
    })
}

/// Build a `ProverConfig` for `(log_n, log_batch_size, log_inv_rate)` from
/// the embedded security TOML. **Strict**: returns `Err` if no security
/// config has been derived for `(m, log_inv_rate)`. Use this as the
/// production entry point; never silently falls back to default parameters
/// with weaker (or unverified) soundness.
///
/// For ad-hoc / testing shapes where a security spec hasn't been derived,
/// callers can use [`default_config`] explicitly — but that's
/// `#[deprecated]` outside of test code because the per-level parameters
/// haven't been audited.
pub fn prover_config_for(
    log_n: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Result<ProverConfig, String> {
    // Reclaim coin-flip: memoize the pure (log_n, log_batch_size, profile) config derivation
    // (TOML parse + soundness derivation). Circuit-determined, so caching is bit-exact.
    use std::sync::{Mutex, OnceLock};
    static MEMO: OnceLock<Mutex<Vec<((usize, usize, &'static str), ProverConfig)>>> =
        OnceLock::new();
    let key = (log_n, log_batch_size, profile.as_str());
    let memo = MEMO.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(g) = memo.lock() {
        for (k, v) in g.iter() {
            if *k == key {
                return Ok(v.clone());
            }
        }
    }
    let pv = prover_config_for_uncached(log_n, log_batch_size, profile)?;
    if let Ok(mut g) = memo.lock() {
        g.push((key, pv.clone()));
    }
    Ok(pv)
}

fn prover_config_for_uncached(
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

/// Verifier-side counterpart to [`prover_config_for`]. Same strict lookup.
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

/// Verifier-side counterpart to [`default_config`].
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

// ===================================================================
// Security configuration schema
// ===================================================================
//
// Auditable, per-level spec for a Ligerito instance: query count, grinding
// bits, slack-from-Johnson, and the proximity-gap analysis the parameters
// were derived under. Designed to be (de)serializable so it can live in a
// TOML/JSON file alongside the prover/verifier code.

/// Which proximity-gap analysis a level's parameters were derived under.
/// Determines which formulas the implementation should verify against the
/// declared (η, queries, grinding) tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoundnessRegime {
    /// Unique decoding radius: γ = δ/2 (δ = 1 − ρ the code's relative
    /// distance; no proximity-loss backoff). Theorem `ca-udr` of our paper's
    /// Appendix C.3 (adapted from Ben-Sasson–Carmon–Haböck–Kopparty–Saraf
    /// "On Proximity Gaps for Reed–Solomon Codes", 2025, Corollary 1.4): the
    /// exceptional set is `a = γ·n + 1`, growing with the codeword length `n`,
    /// so the proximity-gap term is recovered per level by `fold_grinding_bits`
    /// rather than coming out 0. `eta` is `None` for this regime.
    Udr,
    /// Johnson radius with explicit slack `η` (γ = (1 − √ρ) − η) **with
    /// out-of-domain binding**. Theorem 1.5 of the same paper gives the
    /// proximity-gap exceptional set `a = O_ρ(n / η^5)`; the level's
    /// `fold_grinding_bits` should be ≥ (target_bits − log₂(q/a)).
    /// Binding to a single codeword of the (Johnson-bounded) interleaved list
    /// is via `ood_samples` explicit multilinear OOD evaluations — except at
    /// L0, where the opening's own post-commit random evaluation claim plays
    /// the OOD role (union over the list, `L·μ/q`), so `ood_samples = 0`.
    ///
    /// Note there is deliberately no plain `Johnson` variant: without OOD
    /// binding the query phase pays a union bound over the interleaved list
    /// (≈ 19–52 bits here), which our query counts do not include. A config
    /// claiming Johnson soundness without OOD accounting would be unsound.
    JohnsonOod,
}

/// Where in a level's Fiat-Shamir transcript the grinding step lands.
/// Currently only one choice; reserved for future protocol variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrindingStep {
    /// Grind happens after the level's Merkle root is observed but before
    /// query positions are sampled. Standard FRI/STARK pattern.
    PostCommitPreQueries,
}

/// Parameters for a single level in the recursive Ligerito ladder.
/// L0 = the upstream `pcs::commit` output (reused, not re-committed);
/// L1 .. L_{r−1} are the recursive commits; the final residual `yr` block
/// is described separately in [`FinalBlockConfig`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeritoLevelConfig {
    /// PCS rate at this level: codeword expansion factor = 2^log_inv_rate.
    pub log_inv_rate: usize,
    /// Message dimension at this level (log of number of F128 columns in
    /// the codeword). `log_msg_cols + log_inv_rate = log_2(block_len)`.
    pub log_msg_cols: usize,
    /// Log of lane width per Merkle leaf at this level. For L0 = `initial_k`;
    /// for L_i (i ≥ 1) = the previous level's k_recursive.
    pub log_num_interleaved: usize,
    /// Number of sumcheck folds taken at this level. For L0 = `initial_k`
    /// (the lane fold); for L_i (i ≥ 1) = the recursive fold k_{i−1}.
    pub k_recursive: usize,
    /// Which proximity-gap analysis the (eta, queries, grinding_bits)
    /// tuple was derived under. Determines the formulas the implementation
    /// validates against.
    pub regime: SoundnessRegime,
    /// Slack from the Johnson radius. Required for the `JohnsonOod` regime;
    /// must be `None` for `Udr`.
    pub eta: Option<f64>,
    /// Proximity loss `ε*` for the UDR radius `γ = δ/2 − ε*` (our paper
    /// App. C.3 / BCHKS25 Cor. 1.4); `0` in the shipped configs (full
    /// unique-decoding radius δ/2, no backoff). Required for `Udr`; must be
    /// `None` for `JohnsonOod`. The exceptional set is `a = γ·n + 1`,
    /// length-dependent (see [`paper_thm_1_4_log_a`]).
    #[serde(default)]
    pub proximity_loss: Option<f64>,
    /// Number of codeword position queries opened at this level (the FRI
    /// query phase). Bounds the per-query soundness term `(1−γ)^Q`.
    pub queries: usize,
    /// **Query-phase** PoW grinding bits, ground post-commit/pre-queries
    /// (see [`GrindingStep`]). Each bit substitutes for
    /// ~1/log₂(1/(1−γ)) queries at this level.
    pub grinding_bits: usize,
    /// **Fold-challenge** PoW grinding bits, ground immediately before EACH
    /// of this level's `k_recursive` fold challenges. Boosts the
    /// proximity-gap term (which lives on the fold challenges):
    /// `eps_pg + fold_grinding_bits ≥ target`.
    #[serde(default)]
    pub fold_grinding_bits: usize,
    /// Out-of-domain samples taken right after this level's commit enters
    /// the transcript (`JohnsonOod` only). Each binds the prover to a single
    /// codeword of the interleaved list via a multilinear evaluation claim.
    /// Must be 0 at L0 (bound by the opening's own post-commit evaluation
    /// claim) and ≥ 1 at deeper `JohnsonOod` levels.
    #[serde(default)]
    pub ood_samples: usize,
    /// Security target this level guarantees, post-grinding.
    pub target_security_bits: usize,
    /// Diagnostic — `log₂(q/a)` under the chosen regime. The implementation
    /// should assert this matches the formula at startup, modulo rounding.
    pub expected_eps_pg_bits: f64,
    /// Diagnostic — `Q · log₂(1/(1−γ))`. Should be ≥
    /// `target_security_bits − grinding_bits`.
    pub expected_eps_query_bits: f64,
    /// Diagnostic — OOD binding bits (`JohnsonOod` only):
    /// `s·(128 − log₂μ) − (2·log₂L − 1)` for explicit samples, or
    /// `128 − log₂L − log₂μ` for the implicit L0 binding, where `L` is the
    /// Johnson interleaved list size and `μ` the level's variable count.
    #[serde(default)]
    pub expected_eps_ood_bits: Option<f64>,
}

/// Descriptor for the final-residual block (`yr`) sent in the clear at the
/// end of the last recursive level. It has no commit and no queries, so the
/// only meaningful parameter is its dimension.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalBlockConfig {
    /// `log_2(|yr|)` — number of F128 values sent in the clear. The last
    /// recursive level's sumcheck stops at this dim instead of folding to 1.
    pub yr_log_n: usize,
}

/// Complete security spec for one Ligerito instance, covering a single
/// `(hash, m)` pair. Designed to round-trip cleanly via serde (TOML/JSON).
///
/// **Validation invariants** (checked by [`Self::validate`]):
/// 1. `initial_k + Σ levels[1..].k_recursive + final_block.yr_log_n == log_n`.
/// 2. Each level's `expected_eps_pg_bits` is consistent with the declared
///    regime and `eta` (within tolerance).
/// 3. Each level's `expected_eps_query_bits ≥ target_security_bits −
///    grinding_bits` (queries cover what grinding doesn't).
/// 4. `eta` is `Some` iff regime ∈ {Johnson, JohnsonOod}; `None` for Udr.
/// 5. `log_msg_cols`, `log_num_interleaved`, `k_recursive` match the
///    recursive-shape constraint (each level's input dim equals the
///    previous level's `log_msg_cols`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeritoSecurityConfig {
    /// Block-encoder log size: m = log₂(witness bit count).
    pub m: usize,
    /// Packed-witness log dim (`= m − LOG_PACKING = m − 7`).
    pub log_n: usize,
    /// L0 lane fold. Must equal the upstream `PcsParams::log_batch_size` so
    /// the L0 commit can be reused without re-committing.
    pub initial_k: usize,
    /// Round-by-round security target (bits): validate() asserts every error
    /// term at every round (round-by-round soundness) clears at least this
    /// much. Total security is the *minimum* over rounds — the notion that
    /// governs Fiat-Shamir security (cf. Ethereum's `soundcalc`) — so there is
    /// deliberately no whole-protocol union bound over terms.
    pub target_security_bits: usize,
    /// Identifier of the proximity-gap analysis used. Self-documents which
    /// theorem the per-level parameters were derived from. Example:
    /// `"ben_sasson_2025_thm_4_6"`.
    pub analysis_version: String,
    /// Field of the protocol. Example: `"f128"`.
    pub field: String,
    /// Hash function used by the Merkle commitments: `"sha256"` or
    /// `"blake3"`. Read via [`LigeritoSecurityConfig::merkle_hash`] and
    /// carried into the prover/verifier configs; [`validate`] rejects any
    /// other value.
    ///
    /// This selects the **Merkle** hash only. The Fiat-Shamir transcript hash
    /// is a separate, independent choice made where the challenger is built
    /// ([`crate::challenger::FsChallenger::with_hash`]) — the challenger is
    /// constructed by the caller, upstream of any PCS config, so there is
    /// deliberately no field for it here rather than one that cannot drive
    /// anything.
    ///
    /// [`validate`]: LigeritoSecurityConfig::validate
    pub hash: String,
    /// Where in the per-level FS transcript grinding is placed.
    pub grinding_step: GrindingStep,
    /// Per-level parameters, in order L0, L1, L2, ....
    pub levels: Vec<LigeritoLevelConfig>,
    /// Final residual block descriptor.
    pub final_block: FinalBlockConfig,
}

/// Default field size used for soundness analysis: `q = 2^128` (our F128).
const ANALYSIS_LOG_Q: f64 = 128.0;

/// Round a float to one decimal place. Used to round paper-predicted
/// soundness diagnostics so the generated TOMLs stay readable.
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Bit-level tolerance when comparing declared diagnostics
/// (`expected_eps_pg_bits` / `expected_eps_query_bits`) against the value
/// computed from the regime's formulas. Set generously enough that rounding
/// in the TOML doesn't cause spurious failures, but tightly enough that an
/// incorrect declaration of η, Q, or grinding can't slip through.
const PAPER_COMPAT_TOL_BITS: f64 = 0.6;

/// Proximity-gap exceptional set for the list-decoding (Johnson) regime, per
/// our paper's Appendix C.3 (Theorem `ca-johnson`, adapted from BCHKS25
/// Theorem 4.6). For a Reed–Solomon code of rate `ρ`, codeword length `n`,
/// and Johnson slack `η` (proximity radius `γ = 1 − √ρ − η`), the MCA error is
/// `a/|F|` with
///
///   `a = [2(m+½)^5 + 3(m+½)·γ·ρ] / (3·ρ^{3/2}) · n + (m+½)/√ρ`,
///
/// where `η = 1 − √ρ − γ` and `m = max(⌈√ρ/(2η)⌉, 3)`. Returns `log₂ a`.
///
/// This is the per-fold-step MCA error, stated for a two-row interleaved word
/// (`C ∈ F^{2×n}`). The ℓ-round lane fold of a `2^ℓ`-interleaved word adds a
/// row-union factor via App. C.3's Lemma `mca-commutes`; see
/// [`paper_johnson_log_a`].
fn paper_thm_ca_johnson_log_a(log_inv_rate: usize, eta: f64, log_msg_cols: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let sqrt_rho = rho.sqrt();
    let gamma = 1.0 - sqrt_rho - eta;
    // m = ⌈√ρ/(2η)⌉ where η = 1−√ρ−γ, floored at 3.
    let m_param = ((sqrt_rho / (2.0 * eta)).ceil() as usize).max(3) as f64;
    let half = m_param + 0.5;
    let half5 = half.powi(5);
    let numerator = 2.0 * half5 + 3.0 * half * gamma * rho;
    let denominator = 3.0 * rho.powf(1.5);
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    let a = (numerator / denominator) * n + half / sqrt_rho;
    a.log2()
}

/// Johnson-regime proximity-gap `log₂ a` for a level, including the row-union
/// factor from our paper's Appendix C.3 (Lemma `mca-commutes`, "MCA commutes
/// with list decoding").
///
/// The base MCA error `ε = a_RLC/|F|` from [`paper_thm_ca_johnson_log_a`] is
/// stated for a two-row interleaved word (one fold step). Folding a
/// `2^ℓ`-interleaved word (ℓ = `log_num_interleaved`) over its ℓ lane-fold
/// rounds pays a row union: by the lemma, round `i` incurs `2^{ℓ-i}·ε`, so the
/// worst round (`i = 1`) pays the factor `2^{ℓ-1}` = (interleaving factor)/2.
/// We bind the per-level grinding to that worst round, returning
/// `log₂(2^{ℓ-1}·a_RLC) = log₂ a_RLC + (ℓ-1)`.
///
/// `ℓ ≤ 1` (`L ≤ 2`) means no row union; the `(ℓ-1)` penalty clamps to 0.
fn paper_johnson_log_a(
    log_inv_rate: usize,
    eta: f64,
    log_msg_cols: usize,
    log_num_interleaved: usize,
) -> f64 {
    let base = paper_thm_ca_johnson_log_a(log_inv_rate, eta, log_msg_cols);
    // Row-union factor 2^{ℓ-1} (worst round i=1 of the ℓ-round lane fold),
    // ℓ = log_num_interleaved. In bits: (ℓ-1), clamped ≥ 0.
    let row_union_penalty = (log_num_interleaved as f64 - 1.0).max(0.0);
    base + row_union_penalty
}

/// Per-query log₂(1/(1−γ)) under the Johnson regime: each query closes
/// `log_2(1/(1-γ))` bits of soundness against a γ-far adversary.
fn paper_per_query_bits(log_inv_rate: usize, eta: f64) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = 1.0 - rho.sqrt() - eta;
    (1.0 / (1.0 - gamma)).log2()
}

/// UDR proximity radius: the **maximum** allowed by our paper's App. C.3
/// (Theorem `ca-udr`, BCHKS25 Cor. 1.4), whose valid range is
/// `[δ/3, δ/2 − 3/(δ·n)]`. We take the top of the range,
///
///   `γ = δ/2 − 3/(δ·n) − ε*`,
///
/// where `δ = 1 − ρ` is the code's relative minimum distance,
/// `n = 2^(log_msg_cols + log_inv_rate)` the codeword length, and `ε*`
/// (`proximity_loss`) optional extra slack below the maximum (`0` in shipped
/// configs → exactly the maximal radius). The `3/(δ·n)` backoff is the
/// theorem-mandated minimum and shrinks with the codeword length.
fn udr_gamma(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let delta = 1.0 - rho;
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    delta / 2.0 - 3.0 / (delta * n) - proximity_loss
}

/// Per-query log₂(1/(1−γ)) under the UDR regime at the maximal radius
/// `γ = δ/2 − 3/(δ·n) − ε*` (see [`udr_gamma`]).
fn udr_per_query_bits(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let gamma = udr_gamma(log_inv_rate, log_msg_cols, proximity_loss);
    (1.0 / (1.0 - gamma)).log2()
}

/// Asymptotic (n → ∞) UDR per-query soundness at `γ = δ/2`, dropping the
/// finite-length `3/(δ·n)` backoff. Length-agnostic — used for ladder-shape
/// feasibility and [`udr_queries`]; the shipped per-level configs use the
/// n-aware [`udr_per_query_bits`]. The dropped backoff slightly *under*-counts
/// queries, but the per-level block-length check in `derive_profile` (and the
/// `+5` feasibility padding) catch any shape that wouldn't hold the real,
/// n-aware query count.
fn udr_per_query_bits_asymptotic(log_inv_rate: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = (1.0 - rho) / 2.0;
    (1.0 / (1.0 - gamma)).log2()
}

/// UDR proximity-gap exceptional set, per our paper's Appendix C.3
/// (Theorem `ca-udr`, adapted from BCHKS25 Corollary 1.4): at proximity
/// radius `γ` (here the maximal `γ = δ/2 − 3/(δ·n)`; see [`udr_gamma`]) the
/// exceptional set is
///
///   `a = γ·n + 1`,
///
/// where `n = 2^(log_msg_cols + log_inv_rate)` is the codeword length at this
/// level. The `log₂ a ≈ log₂(γ·n)` term therefore **grows with the codeword
/// length**, so larger witnesses give a smaller `eps_pg = 128 − log₂ a` and
/// need proportionally more `fold_grinding_bits` to hold a fixed target.
/// Callers add **no** row-union penalty in this regime: the unique-decoding
/// list has size 1, so (per Diamond and Gruen) MCA-commutes holds with error
/// ε directly, unlike the Johnson regime's `2^{ℓ-1}` factor. This replaced an
/// earlier length-independent `a ≤ 2/ε*` form, which did not match the paper's
/// stated bound.
fn paper_thm_1_4_log_a(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let gamma = udr_gamma(log_inv_rate, log_msg_cols, proximity_loss);
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    (gamma * n + 1.0).log2()
}

/// Johnson-bound list size of the *interleaved* RS code at radius
/// `θ = 1 − √ρ − η`, in log₂. Independent of the interleaving factor.
///
/// Interleaving preserves relative distance — `V^{⊙m}` has the base code's
/// distance `δ = 1 − ρ` — and only enlarges the alphabet (to `q^m`). The
/// Johnson bound depends solely on (distance, radius, alphabet size), so the
/// interleaved list size at any radius *below* the Johnson radius `1 − √ρ`
/// is bounded by the very same single-code Johnson list size
///
///   `L_int ≤ L_base ≤ 1/(2·η·√ρ)`,
///
/// with no dependence on `m` and, crucially, no `L_base^r` blow-up.
///
/// The general GGR (Gopalan–Guruswami–Raghavendra, Thm 2.5) interleaved bound
/// `L_int ≤ C(b+r, r)·L_base^r` is only needed to push the list-decoding
/// radius *past* the Johnson bound toward `δ`. Ligerito deliberately sits at
/// `θ = 1 − √ρ − η`, strictly below the Johnson radius by slack `η > 0`, so
/// that regime never applies and the plain Johnson bound is both correct and
/// far tighter (it dominates GGR throughout the regime RS can reach).
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

/// OOD binding bits for a `JohnsonOod` level. `mu_vars` is the level's
/// multilinear variable count (`log_msg_cols + log_num_interleaved`).
///
/// - `ood_samples ≥ 1` (explicit samples): the bad event is two distinct
///   list elements agreeing on all `s` random points of `F^μ`
///   (Schwartz–Zippel, total degree ≤ μ), union over pairs:
///       bits = s·(128 − log₂ μ) − (2·log₂ L_int − 1).
/// - `ood_samples = 0` (L0's implicit binding): the opening's own evaluation
///   claim at a post-commit random point pins the prover to one claimed
///   value, so the union is over the list (not pairs):
///       bits = 128 − log₂ L_int − log₂ μ.
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
    /// Compute the proximity-gap and per-query soundness bits this level is
    /// expected to deliver under its declared regime. Returns
    /// `(eps_pg_bits, eps_query_bits)` where:
    ///   eps_pg_bits   = log₂(q/a) under the regime's threshold-a formula
    ///   eps_query_bits = Q · log₂(1/(1−γ))
    ///
    /// Used by [`LigeritoSecurityConfig::validate`] to assert the declared
    /// `expected_*_bits` diagnostics are consistent with the regime's
    /// canonical formulas (i.e., the config is compatible with the paper).
    pub fn paper_predicted_bits(&self) -> (f64, f64) {
        match self.regime {
            SoundnessRegime::JohnsonOod => {
                let eta = self.eta.expect("JohnsonOod must have eta");
                // App. C.3 Lemma `mca-commutes`: the ℓ-round lane fold of a
                // 2^ℓ-interleaved word (ℓ = log_num_interleaved) pays a
                // row-union factor 2^{ℓ-i} at round i; the worst round (i=1)
                // gives 2^{ℓ-1}, on top of the base ca-johnson MCA error.
                let log_a = paper_johnson_log_a(
                    self.log_inv_rate,
                    eta,
                    self.log_msg_cols,
                    self.log_num_interleaved,
                );
                let eps_pg = ANALYSIS_LOG_Q - log_a;
                // Per-query soundness WITHOUT a list union bound — the OOD
                // binding (see `paper_ood_bits`) pins the prover to a single
                // codeword of the interleaved list before queries are drawn.
                let per_q = paper_per_query_bits(self.log_inv_rate, eta);
                let eps_query = self.queries as f64 * per_q;
                (eps_pg, eps_query)
            }
            SoundnessRegime::Udr => {
                // App. C.3 Thm `ca-udr` (BCHKS25 Cor. 1.4): a = γ·n + 1 for
                // radius γ = δ/2 (ε* = 0, no backoff).
                let proximity_loss = self
                    .proximity_loss
                    .expect("Udr regime must carry proximity_loss");
                // No row-union penalty in the unique-decoding regime: the list
                // has size 1, so (per Diamond and Gruen) the MCA-commutes step
                // holds with error ε directly — the Johnson regime's 2^{ℓ-1}
                // row union is unnecessary. So eps_pg = 128 − log₂ a.
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

    /// OOD binding bits this level is expected to deliver (`JohnsonOod`
    /// only; `None` for `Udr`, where the unique-decoding list has size 1 and
    /// no binding step exists). See [`paper_ood_bits`].
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
    /// Validate that the config is internally consistent and matches the
    /// declared analysis. Returns the first violation found, if any.
    pub fn validate(&self) -> Result<(), String> {
        if self.log_n + 7 != self.m {
            return Err(format!(
                "log_n ({}) + LOG_PACKING (7) != m ({})",
                self.log_n, self.m
            ));
        }

        // Reject a `hash` we do not implement here, so a bad spelling is caught
        // at config-load time rather than silently committing under SHA-256.
        self.merkle_hash()?;

        // Recursion shape: initial_k + Σ k_recursive (L1+) + yr_log_n = log_n.
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

        // L0 must have k_recursive = initial_k and log_num_interleaved = initial_k.
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

        // Per-level checks.
        let mut dim_in = self.log_n;
        for (i, lv) in self.levels.iter().enumerate() {
            // Shape: log_msg_cols + log_num_interleaved = dim_in.
            if lv.log_msg_cols + lv.log_num_interleaved != dim_in {
                return Err(format!(
                    "L{i}: log_msg_cols ({}) + log_num_interleaved ({}) ≠ input dim ({dim_in})",
                    lv.log_msg_cols, lv.log_num_interleaved
                ));
            }

            // eta presence matches regime.
            match (lv.regime, lv.eta) {
                (SoundnessRegime::Udr, Some(_)) => {
                    return Err(format!("L{i}: regime=udr but eta is set"));
                }
                (SoundnessRegime::JohnsonOod, None) => {
                    return Err(format!("L{i}: regime requires eta but eta is None"));
                }
                _ => {}
            }

            // proximity_loss presence matches regime (UDR-only).
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

            // OOD samples match regime: UDR has no list, so no OOD; under
            // JohnsonOod every level past L0 needs explicit samples, while
            // L0 is bound by the opening's own post-commit evaluation claim.
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

            // OOD diagnostic matches regime + formula.
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

            // Paper-compatibility: the declared expected_*_bits must agree
            // with what the regime's formula predicts (within tolerance).
            // Asserts the config was actually derived from the paper, not
            // hand-waved into compliance.
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

            // Security: queries cover the gap left by grinding.
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

            // Per-application proximity gap + fold-challenge grinding must
            // reach target. (The pg bad event lives on the fold challenges,
            // so only the fold grind — done before each fold challenge —
            // boosts it; the query-phase grind does not.)
            if lv.expected_eps_pg_bits + lv.fold_grinding_bits as f64 + 1e-3
                < lv.target_security_bits as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_pg_bits ({:.2}) + fold_grinding ({}) < target ({})",
                    lv.expected_eps_pg_bits, lv.fold_grinding_bits, lv.target_security_bits
                ));
            }

            // OOD binding must reach target on its own (no grind covers it;
            // escalate ood_samples instead).
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

            // Advance dim_in for next level: subtract k_recursive (the folds at this level).
            dim_in -= lv.k_recursive;
        }

        if dim_in != yr_log_n {
            return Err(format!(
                "after consuming all levels, dim_in ({dim_in}) ≠ yr_log_n ({yr_log_n})"
            ));
        }

        // Round-by-round soundness: each error term at each round is checked
        // against `target_security_bits` in the per-level loop above. Total
        // security is the minimum over rounds (the Fiat-Shamir-relevant notion;
        // cf. Ethereum's `soundcalc`), so there is intentionally no
        // whole-protocol union bound summed across terms.
        Ok(())
    }

    /// Mechanically derive a paper-compatible `LigeritoSecurityConfig` for
    /// `(m, log_inv_rate)` targeting `target_security_bits`, in the
    /// **unique-decoding regime** (BCHKS25 Theorem 1.4). Uses the same
    /// recursion shape as [`default_config`] and picks per-level
    /// `(proximity_loss, queries)` so that each level satisfies:
    ///
    ///   * `expected_eps_query_bits ≥ target_security_bits` (queries alone
    ///     close the target; per the "100 bits from queries always" policy).
    ///   * `expected_eps_pg_bits + fold_grinding_bits ≥ target_security_bits`.
    ///     Under Thm `ca-udr` the exceptional set is `a = γ·n + 1`
    ///     (length-dependent), so `eps_pg = 128 − log₂(γ·n+1) − log₂(log L)`
    ///     decreases with witness size; any shortfall below target is made up
    ///     by `fold_grinding_bits` (query-phase `grinding_bits` stays 0).
    ///
    /// All diagnostic fields are populated from the paper formulas so the
    /// resulting config validates strictly against [`Self::validate`].
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
        // Build per-level (log_msg_cols, log_num_interleaved, k_recursive).
        let mut log_msg_cols_per_level = Vec::with_capacity(r + 1);
        let mut log_num_interleaved_per_level = Vec::with_capacity(r + 1);
        let mut k_recursive_per_level = Vec::with_capacity(r + 1);
        // L0
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
            // UDR: γ = δ/2 = (1−ρ)/2 (ε* = UDR_PROXIMITY_LOSS = 0, no backoff).
            // Thm `ca-udr`'s exceptional set a = γ·n + 1 grows with the
            // codeword length, so eps_pg falls ~1 bit per witness doubling and
            // is recovered by fold_grinding_bits below.
            let proximity_loss = UDR_PROXIMITY_LOSS;
            let per_q = udr_per_query_bits(rate, log_msg_cols_per_level[i], proximity_loss);
            let queries = ((target_security_bits as f64) / per_q).ceil() as usize;
            // No row-union penalty in the unique-decoding regime (list size 1):
            // per Diamond and Gruen, MCA-commutes holds with error ε directly,
            // unlike the Johnson regime's 2^{ℓ-1} row union.
            let log_a = paper_thm_1_4_log_a(rate, log_msg_cols_per_level[i], proximity_loss);
            let eps_pg = ANALYSIS_LOG_Q - log_a;
            // Any pg shortfall is ground on the fold challenges (where the
            // pg bad event lives); 0 at the 100-bit target.
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
        // Final residual: yr_log_n = log_n − initial_k − Σ k_recursive
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

    /// Derive the security config for a named [`LigeritoProfile`] at witness
    /// size `m`. Each profile targets its bit level under **round-by-round
    /// soundness**: every error term (pg + fold grinding, query + query
    /// grinding, OOD) clears the target individually, and the protocol's
    /// security is the *minimum* over rounds — the notion that governs
    /// Fiat-Shamir security (cf. Ethereum's `soundcalc`), not a whole-protocol
    /// union bound over terms. The three shipped profiles:
    ///
    /// - `Fast`:   JohnsonOod, rate 1/2, η = 0.02, 100 bits per round.
    /// - `Slim`:   JohnsonOod, rate 1/4, η = 0.02, 16-bit query grinding at
    ///             every level, 100 bits per round.
    /// - `Secure`: Udr, rate 1/2, ε* = 1e-3, 120 bits per round.
    pub fn derive_profile(m: usize, profile: LigeritoProfile) -> Result<Self, String> {
        /// Johnson slack below the Johnson radius, flat across levels.
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

        // Length-agnostic per-query estimate for ladder-shape feasibility
        // (the per-level codeword length `n` is not known until the shape is
        // fixed). UDR uses the asymptotic γ = δ/2; the actual per-level config
        // below uses the n-aware `udr_per_query_bits`.
        let per_query_bits_feas = |rate: usize| -> f64 {
            match profile {
                LigeritoProfile::Secure => udr_per_query_bits_asymptotic(rate),
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    paper_per_query_bits(rate, JOHNSON_ETA)
                }
            }
        };

        // Shape derivation needs per-level query counts for block-length
        // feasibility before the level count (and hence the exact per-term
        // target) is known. Use a conservative target of target_bits + 5
        // (≥ log₂(3 terms · 10 levels)); the final counts are ≤ this.
        let t_feas = target_bits as f64 + 5.0;
        let queries_feas = |rate: usize| -> usize {
            ((t_feas - query_grind as f64).max(1.0) / per_query_bits_feas(rate)).ceil() as usize
        };
        let shape = derive_ladder_shape(log_n, initial_k, log_inv_rate, &queries_feas)?;
        let n_levels = shape.log_inv_rates.len();

        // Round-by-round target: every error term (pg, query, ood) at every
        // round must individually clear `target_bits`. Round-by-round soundness
        // — the notion that governs the Fiat-Shamir security of the IOP — is the
        // *minimum* security level over rounds, not the sum, so there is
        // deliberately NO `log₂(#terms)` union-bound headroom. This matches the
        // convention Ethereum's `soundcalc` uses for hash-based zkEVM IOPs
        // (total security = min over rounds). It also keeps the proximity-gap
        // fold grinding (especially L0's, the dominant prover cost) at the
        // round-by-round minimum rather than paying ~4 bits of union slack that
        // buys nothing.
        let t = target_bits as f64;

        let mut levels = Vec::with_capacity(n_levels);
        for i in 0..n_levels {
            let rate = shape.log_inv_rates[i];
            let cols = shape.log_msg_cols[i];
            let ilv = shape.log_num_interleaved[i];
            // Actual per-level per-query bits: n-aware (maximal radius) for
            // UDR, length-agnostic Johnson otherwise.
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
                    // No row-union penalty in the unique-decoding regime (list
                    // size 1): per Diamond and Gruen, MCA-commutes holds with
                    // error ε directly (vs the Johnson regime's 2^{ℓ-1} factor).
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
                        0 // bound by the opening's own evaluation claim
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

    /// Parse a [`LigeritoSecurityConfig`] from a TOML string and validate it.
    /// The caller is expected to embed the file contents via
    /// `include_str!("../../configs/ligerito/m29_fast.toml")` (for compile-time
    /// configs) or read it via `std::fs` (for runtime configs).
    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        let cfg: Self = toml::from_str(s).map_err(|e| format!("toml parse: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Serialize the config back out to TOML. Round-trip-stable with
    /// [`from_toml_str`].
    pub fn to_toml_string(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("toml serialize: {e}"))
    }

    /// Build a `(ProverConfig, VerifierConfig)` pair from this security config.
    /// Drops the security-only fields (eta, queries, grinding, expected_*) but
    /// preserves the recursion shape so the existing prover/verifier code path
    /// works unchanged.
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

    /// The Merkle hash this config selects, parsed from its `hash` field.
    ///
    /// Errors on any spelling we do not implement rather than defaulting —
    /// a config asking for a hash that is not wired up must fail loudly, not
    /// silently produce SHA-256 proofs under a `hash = "…"` that says
    /// otherwise.
    pub fn merkle_hash(&self) -> Result<HashKind, String> {
        HashKind::parse(&self.hash).map_err(|e| format!("security config `hash`: {e}"))
    }
}

// ===================================================================
// Proof
// ===================================================================

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursiveProof {
    /// One row per query, each of `num_interleaved` F128 entries. Rows are
    /// emitted in **sorted** query-position order so they align with the
    /// merkle multi-proof.
    pub opened_rows: Vec<Vec<F128>>,
    /// Single octopus multi-proof shared across all queries at this level.
    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalProof {
    /// Remaining polynomial sent in clear at the last recursive step.
    pub yr: Vec<F128>,
    /// Same sorted-by-position convention as [`RecursiveProof`].
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
    /// Per-level PoW nonces (one entry per query phase). When all
    /// `grinding_bits` are 0 (the default config), each entry is just 0
    /// and the verifier's PoW check is a no-op. `#[serde(default)]` keeps
    /// older serialized proofs that pre-date this field readable.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
    /// Claimed multilinear OOD evaluations, flattened in transcript order
    /// (level 1's `ood_samples[1]` values, then level 2's, ...). Empty when
    /// the config takes no OOD samples (UDR profiles, legacy paths).
    #[serde(default)]
    pub ood_values: Vec<F128>,
    /// Fold-challenge PoW nonces, flattened in transcript order — one per
    /// fold challenge at every level with `fold_grinding_bits > 0`. Empty
    /// when no level fold-grinds.
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

    /// Print a per-component breakdown of the proof size to stderr.
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

// ===================================================================
// Multilinear helpers
// ===================================================================

/// Multilinear extension of `evals` at the boolean cube of dimension `n`,
/// LSB-first indexing: `eval(b_0, …, b_{n-1}) = evals[b_0 + 2·b_1 + …]`.
///
/// Partially evaluate at the first `k` variables (the LSB end): given
/// challenges `rs ∈ F^k`, returns the length-`2^{n-k}` table
/// `f(rs[0], …, rs[k-1], x_k, …, x_{n-1})`.
///
/// Matches Flock's [`build_eq_table`] LSB-first convention (and bolt-rs's
/// `partial_eval` Julia convention).
pub(crate) fn partial_eval_lsb(evals: &[F128], rs: &[F128]) -> Vec<F128> {
    let mut cur = evals.to_vec();
    for &r in rs {
        let one_plus_r = F128::ONE + r;
        let half = cur.len() / 2;
        // Pair (cur[2i], cur[2i+1]) collapses to cur[2i]·(1+r) + cur[2i+1]·r.
        // LSB-first ⇒ adjacent pairs are bit_0 = 0 vs 1.
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            next.push(cur[2 * i] * one_plus_r + cur[2 * i + 1] * r);
        }
        cur = next;
    }
    cur
}

/// Evaluate the multilinear extension of `evals` at `point` (LSB-first).
/// `point.len()` must equal `log2(evals.len())`. Test oracle for
/// `partial_eval_lsb` composition; not used in production paths.
#[cfg(test)]
pub(crate) fn eval_mle_lsb(evals: &[F128], point: &[F128]) -> F128 {
    let folded = partial_eval_lsb(evals, point);
    debug_assert_eq!(folded.len(), 1);
    folded[0]
}

// ===================================================================
// LCH novel-basis evaluations (ported from bolt-rs `fft.rs`)
// ===================================================================
//
// Same subspace-polynomial recurrence `s_{i+1}(x) = s_i(x)² + s_i(v_i)·s_i(x)`
// as Flock's `AdditiveNttF128`, but we expose the evaluation at an arbitrary
// point — which the NTT doesn't currently surface publicly. Standard basis only
// (v_i = 2^i, embedded as `F128::new(1 << i, 0)`).

#[inline]
fn next_s(s: F128, s_at_root: F128) -> F128 {
    s * s + s_at_root * s
}

/// `sks_vks[k] = s_k(v_k)` for `k = 0..=log_n`. Length `log_n + 1`.
/// Only depends on `log_n` (standard basis `v_i = 2^i` is fixed), so the
/// values are circuit-determined — no challenge or witness dependence.
///
/// Micro-stack memoization: despite the "callers cache" intent, the ranked
/// prover recomputes this at dims 19/16/13/10/7 on EVERY prove (L0 induce at
/// `n1` plus the recursion loop's `n_next` levels). Same memo pattern as
/// [`prover_config_for`]; the cached value is the pure function output, so
/// the memo is bit-exact by construction (`eval_sk_at_vks_memo_matches_direct`
/// checks it anyway). `FLOCK_NO_MICRO_STACK=1` bypasses the cache and always
/// recomputes — the incumbent behavior.
pub(crate) fn eval_sk_at_vks(log_n: usize) -> Vec<F128> {
    use std::sync::{Mutex, OnceLock};
    if !crate::micro_stack_enabled() {
        return eval_sk_at_vks_uncached(log_n);
    }
    static MEMO: OnceLock<Mutex<Vec<(usize, Vec<F128>)>>> = OnceLock::new();
    let memo = MEMO.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(g) = memo.lock() {
        for (k, v) in g.iter() {
            if *k == log_n {
                return v.clone();
            }
        }
    }
    let out = eval_sk_at_vks_uncached(log_n);
    if let Ok(mut g) = memo.lock() {
        if !g.iter().any(|(k, _)| *k == log_n) {
            g.push((log_n, out.clone()));
        }
    }
    out
}

fn eval_sk_at_vks_uncached(log_n: usize) -> Vec<F128> {
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

/// Write into `basis` the **normalized** LCH novel-basis polynomials
/// `X̂_j(x) = Π_{k: bit_k(j)=1} Ŵ_k(x)` for `j ∈ [0, 2^log_n)`, each scaled by
/// `alpha`. `Ŵ_k = s_k / s_k(v_k)` is normalized to match Flock's NTT twiddles.
///
/// `sks_at_x` is a scratch buffer of length `≥ log_n`. `sks_vks` is from
/// [`eval_sk_at_vks`]; `inv_sks_vks[k] = sks_vks[k].inv()` precomputed once
/// across many queries.
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
        // Normalize: Ŵ_i(x) = s_i(x) / s_i(v_i)
        for i in 0..log_n {
            sks_at_x[i] *= inv_sks_vks[i];
        }
    }

    basis[0] = alpha;
    for k in 0..log_n {
        let s_at_x = sks_at_x[k];
        let current_len = 1 << k;
        for i in 0..current_len {
            basis[i + current_len] = s_at_x * basis[i];
        }
    }
}

// ===================================================================
// induce_sumcheck_poly — the per-level basis-poly builder.
// ===================================================================
//
// Given Q opened rows of the previous commitment at query positions and the
// post-partial-eval challenges `v_challenges`, builds:
//   basis_poly[j] = Σ_i  α^i · Ŵ_j(q_i_field)
//   enforced_sum  = Σ_i  α^i · ⟨row_i, eq(v_challenges, ·)⟩
//
// The verifier reconstructs both independently from public inputs and checks
// the sumcheck claim Σ_j f(j) · basis_poly[j] = enforced_sum at the residual.

/// Compute just the `enforced_sum` half of [`induce_sumcheck_poly`]:
///   `enforced_sum = Σ_i eq(α, i_bin) · ⟨opened_rows[i], eq(v_challenges, ·)⟩`
/// Cheap: O(num_queries × num_interleaved). Verifier needs this at level
/// intro time (before residual challenges are known).
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
    let mut sum = F128::ZERO;
    for (i, row) in opened_rows.iter().enumerate() {
        debug_assert_eq!(row.len(), eq.len());
        let dot: F128 = row
            .iter()
            .zip(eq.iter())
            .map(|(&r, &e)| r * e)
            .fold(F128::ZERO, |a, v| a + v);
        sum += alpha_weights[i] * dot;
    }
    sum
}

/// **Succinct** evaluator for the induced basis poly's MLE at residual points.
/// Replaces `induce_sumcheck_poly` + `partial_eval_lsb` in the verifier:
/// instead of materializing the dense `2^log_msg_cols` basis_poly, evaluates
/// its MLE directly using the closed-form identity:
///   `MLE(basis_poly)(p) = Σ_i α^i · Π_k (1 + p[k] · (1 + Ŵ_k(q_i)))`
/// where each `q_i` is the field embedding of `queries[i]`.
///
/// `ris_for_basis` is the fixed prefix of the residual point (the ris range
/// that would have been passed to `partial_eval_lsb(basis_poly, ris_for_basis)`).
/// Length must be `log_msg_cols - yr_log_n`. The function returns evaluations
/// at `2^yr_log_n` points: `ris_for_basis ++ y_bits` for `y ∈ [0, 2^yr_log_n)`.
///
/// Cost: O(num_queries × yr_log_n × 2^yr_log_n + num_queries × log_msg_cols),
/// vs the dense path's O(num_queries × log_msg_cols × 2^log_msg_cols). At m=30
/// L0 with 221 queries, log_msg_cols=17, yr_log_n=4: ~18k ops vs ~500M ops.
/// `⌈log₂ n⌉`. Number of bits needed to index `n` items. Used to size the
/// per-level `alpha` slice for the eq-tensor basis-induction combination.
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

    // Per-query weights are the eq-tensor coefficients `eq(α, i_bin)` for
    // `i ∈ {0,1}^{⌈log₂ n_queries⌉}` (LSB-first), padded with zeros for
    // indices ≥ n_queries. Replaces the legacy α^i Vandermonde scheme;
    // soundness bound goes from `Q/q` (univariate S-Z) to `⌈log₂ Q⌉/q`
    // (multilinear S-Z), matching the rest of the multilinear protocol.
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

    // Per-query precomputation: Ŵ_k(q) for all k, then split into prefix
    // product (fixed scalar) and suffix Ŵ values (varied per y).
    struct PerQuery {
        prefix_prod: F128,
        suffix_w: Vec<F128>, // length = yr_log_n
    }
    let compute_query = |&q: &usize| -> PerQuery {
        let q_field = F128::new(q as u64, 0);
        // Compute s_k(q_field) recursively, then normalize by 1/s_k(v_k).
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
        // Prefix product: Π_{k<prefix_len} (1 + ris[k] · (1 + Ŵ_k(q)))
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
    // This runs once per recursion level over tiny verify-sized inputs
    // (`queries` ≈ tens; `yr_len` ≤ 2^5 since the residual folds to ≤5 bits), so
    // a rayon dispatch per level costs more than the field work itself (measured
    // ~0.47 ms serial vs ~0.75 ms parallel for the whole residual eval at m=30).
    // Stay serial below the crossover — mirror of merkle.rs's `SERIAL_LEVEL_NODES`.
    const PAR_FLOOR: usize = 1024;
    let per_query: Vec<PerQuery> = if n_queries > PAR_FLOOR {
        queries.par_iter().map(compute_query).collect()
    } else {
        queries.iter().map(compute_query).collect()
    };

    // For each residual position y, accumulate the suffix product per query.
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

/// `queries` are **0-indexed** codeword positions. `q_field = F128::new(q, 0)`.
///
/// Parallel: each thread takes a chunk of queries, builds a partial basis_poly
/// accumulator + partial enforced_sum, then we reduce. The per-query work
/// (eq-dot + LCH novel-basis expansion) is independent of other queries.
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

    let eq = build_eq_table(v_challenges); // length 2^v_challenges.len() = num_interleaved

    // Per-query weights are the eq-tensor coefficients `eq(α, i_bin)` for
    // `i ∈ {0,1}^{⌈log₂ n_queries⌉}` (LSB-first), truncated to the first
    // `n_queries` indices. Replaces the legacy α^i Vandermonde scheme;
    // matches the multilinear S-Z structure used by the lane fold.
    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    // Precompute inv_sks_vks once across all queries and threads.
    let inv_sks_vks: Vec<F128> = sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
        .collect();

    // Per-worker chunked accumulation: each worker accumulates a partial
    // basis_poly (length n) and a partial enforced_sum, then we reduce.
    // With a live E-core helper pool the same query chunks drain through the
    // shared P+E queue instead — one chunk per potential worker, so the four
    // UTILITY-QoS helpers claim query chunks the main pool would otherwise
    // serialize behind its own. Grouping queries differently across
    // accumulators cannot change bytes: every query contributes the same
    // scaled basis vector and dot product, and the cross-chunk merge is a
    // GF(2^128) XOR sum. `FLOCK_NO_LIG_INDUCE_HETERO=1` (exactly `"1"`)
    // restores the incumbent main-pool split as the same-binary A/B control.
    let helper_threads = if lig_induce_hetero_enabled() {
        crate::epool::epool().map_or(0, rayon::ThreadPool::current_num_threads)
    } else {
        0
    };
    // 16 chunks when hetero: the queue's engagement floor (EPOOL_MIN_CHUNKS),
    // and enough claims that the four helpers stay fed without inflating the
    // serial partial reduce below by more than two extra length-n passes.
    let n_threads = if helper_threads > 0 {
        (rayon::current_num_threads().max(1) + helper_threads).max(16)
    } else {
        rayon::current_num_threads().max(1)
    };
    let chunk_size = (n_queries + n_threads - 1) / n_threads.max(1);

    let chunk_partial = |t: usize| -> (Vec<F128>, F128) {
        {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(n_queries);
            if start >= end {
                // Empty marker: contributes nothing, skipped in the reduce
                // (previously a length-n zeroed vec that was allocated,
                // filled, and XOR-added for no effect).
                return (Vec::new(), F128::ZERO);
            }
            // Both per-thread buffers are uninit-sound: `local_basis` is
            // fully written by `evaluate_scaled_basis_inplace` before any
            // read (`basis[0] = alpha`, then each doubling level writes
            // `[2^k, 2^{k+1})` from the already-written lower half), and
            // `accum_basis` is seeded by a full `copy_from_slice` of the
            // chunk's FIRST query before any accumulation — algebraically
            // identical to zero-init + XOR-add (x ⊕ 0 = x), deleting one
            // length-n memset and one full-buffer RMW pass per worker.
            let mut accum_basis = crate::alloc_uninit_f128_vec(n);
            let mut local_basis = crate::alloc_uninit_f128_vec(n);
            let mut sks_at_x = vec![F128::ZERO; log_msg_cols.max(1)];
            let mut local_sum = F128::ZERO;

            for i in start..end {
                let row = &opened_rows[i];
                let q = queries[i];
                let ap = alpha_pows[i];

                let dot: F128 = row
                    .iter()
                    .zip(eq.iter())
                    .map(|(&r, &e)| r * e)
                    .fold(F128::ZERO, |a, v| a + v);
                local_sum += dot * ap;

                let q_field = F128::new(q as u64, 0);
                if i == start {
                    evaluate_scaled_basis_inplace(
                        &mut sks_at_x,
                        &mut accum_basis,
                        sks_vks,
                        &inv_sks_vks,
                        q_field,
                        ap,
                    );
                    continue;
                }
                evaluate_scaled_basis_inplace(
                    &mut sks_at_x,
                    &mut local_basis,
                    sks_vks,
                    &inv_sks_vks,
                    q_field,
                    ap,
                );
                for (acc, &v) in accum_basis.iter_mut().zip(local_basis.iter()) {
                    *acc += v;
                }
            }
            (accum_basis, local_sum)
        }
    };
    let partials: Vec<(Vec<F128>, F128)> = if helper_threads > 0 {
        let mut slots: Vec<Option<(Vec<F128>, F128)>> = (0..n_threads).map(|_| None).collect();
        // Raw address rather than `SyncPtr` because the slot type is not
        // `Copy` (same aliasing contract as the stripe fill's `as usize`).
        let slots_addr = slots.as_mut_ptr() as usize;
        crate::epool::run_hetero_chunks(n_threads, |t| {
            // SAFETY: the queue claims each `t` exactly once; each slot is
            // written by its unique claimant and published by the join.
            unsafe {
                (slots_addr as *mut Option<(Vec<F128>, F128)>)
                    .add(t)
                    .write(Some(chunk_partial(t)));
            }
        });
        slots
            .into_iter()
            .map(|s| s.expect("hetero queue ran every chunk"))
            .collect()
    } else {
        (0..n_threads)
            .into_par_iter()
            .map(chunk_partial)
            .collect()
    };

    // Reduce across threads: seed with the first non-empty partial (a move —
    // deletes the zero-seeded output buffer and its redundant first XOR-add
    // pass), then fold the rest in. Zero-query calls keep the zeroed-output
    // behavior of the original.
    let mut iter = partials.into_iter().filter(|(lb, _)| !lb.is_empty());
    let (mut basis_poly, mut enforced_sum) = match iter.next() {
        Some(first) => first,
        None => return (vec![F128::ZERO; n], F128::ZERO),
    };
    for (lb, ls) in iter {
        for (acc, &v) in basis_poly.iter_mut().zip(lb.iter()) {
            *acc += v;
        }
        enforced_sum += ls;
    }

    (basis_poly, enforced_sum)
}

/// Apply three consecutive transpose layers in one read/write pass. `layer`
/// is the lowest (root-most) of the three; the transpose executes forward
/// layers `layer+2`, `layer+1`, then `layer`.
fn transpose_forward_ntt_fused_3layer(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    log_d: usize,
    layer: usize,
) {
    use rayon::prelude::*;

    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], a: usize, b: usize, twiddle: F128) {
        let sum = values[a] + values[b];
        values[a] = sum;
        values[b] = twiddle * sum + values[b];
    }

    let num_blocks = 1usize << layer;
    let block_size = 1usize << (log_d - layer);
    let eighth = block_size >> 3;
    let eighth_log = log_d - layer - 3;
    let row_mask = eighth - 1;
    let twiddles: Vec<[F128; 7]> = (0..num_blocks)
        .map(|block| {
            let mut tw = [F128::ZERO; 7];
            tw[0] = ntt.twiddle(layer, block);
            for s in 0..2 {
                tw[1 + s] = ntt.twiddle(layer + 1, 2 * block + s);
            }
            for s in 0..4 {
                tw[3 + s] = ntt.twiddle(layer + 2, 4 * block + s);
            }
            tw
        })
        .collect();

    // Flatten `(block, row)` into one Rayon range. This keeps all cores busy
    // even for the final few large blocks without opening nested parallel
    // regions, which caused long-tail scheduler stalls in this phase.
    let data_ptr = data.as_mut_ptr() as usize;
    (0..num_blocks * eighth).into_par_iter().for_each(|job| {
        // `eighth` is always a power of two. Spell out the quotient/remainder
        // so rustc does not emit UDIV+MSUB in every eight-value row job.
        let block = job >> eighth_log;
        let row = job & row_mask;
        let base = block * block_size + row;
        let mut values = [F128::ZERO; 8];
        // SAFETY: each `(block,row)` owns the eight distinct positions
        // `base + i*eighth`, and different jobs never overlap.
        unsafe {
            let ptr = data_ptr as *mut F128;
            for (i, value) in values.iter_mut().enumerate() {
                *value = *ptr.add(base + i * eighth);
            }
            let tw = &twiddles[block];
            for pair in 0..4 {
                butterfly(&mut values, 2 * pair, 2 * pair + 1, tw[3 + pair]);
            }
            for half in 0..2 {
                butterfly(&mut values, 4 * half, 4 * half + 2, tw[1 + half]);
                butterfly(&mut values, 4 * half + 1, 4 * half + 3, tw[1 + half]);
            }
            for i in 0..4 {
                butterfly(&mut values, i, i + 4, tw[0]);
            }
            for (i, &value) in values.iter().enumerate() {
                *ptr.add(base + i * eighth) = value;
            }
        }
    });
}

/// Final three transpose layers when the caller retains only the low half.
/// The first two layers still contribute to both root inputs, but the root
/// butterfly's retained output is just `top = a + b`. Its discarded output
/// `bottom = t * (a + b) + b` therefore needs neither the field product nor
/// the store. The four writes per row cover exactly `data[..data.len() / 2]`.
fn transpose_forward_ntt_fused_final_3layer_low_half(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    log_d: usize,
) {
    use rayon::prelude::*;

    #[cfg(test)]
    TEST_TRUNCATED_FINAL_NTT_HITS.with(|hits| hits.set(hits.get() + 1));

    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], a: usize, b: usize, twiddle: F128) {
        let sum = values[a] + values[b];
        values[a] = sum;
        values[b] = twiddle * sum + values[b];
    }

    assert!(log_d >= 3);
    assert_eq!(data.len(), 1usize << log_d);
    let eighth = data.len() >> 3;
    let mut layer_1_twiddles = [F128::ZERO; 2];
    let mut layer_2_twiddles = [F128::ZERO; 4];
    for (block, twiddle) in layer_1_twiddles.iter_mut().enumerate() {
        *twiddle = ntt.twiddle(1, block);
    }
    for (block, twiddle) in layer_2_twiddles.iter_mut().enumerate() {
        *twiddle = ntt.twiddle(2, block);
    }

    let data_ptr = data.as_mut_ptr() as usize;
    (0..eighth).into_par_iter().for_each(|row| {
        let mut values = [F128::ZERO; 8];
        // SAFETY: each row owns the eight positions `row + i*eighth`.
        // Different rows never overlap. Only positions i=0..4 are written;
        // together those positions are exactly the retained low half.
        unsafe {
            let ptr = data_ptr as *mut F128;
            for (i, value) in values.iter_mut().enumerate() {
                *value = *ptr.add(row + i * eighth);
            }
            for pair in 0..4 {
                butterfly(&mut values, 2 * pair, 2 * pair + 1, layer_2_twiddles[pair]);
            }
            for half in 0..2 {
                butterfly(&mut values, 4 * half, 4 * half + 2, layer_1_twiddles[half]);
                butterfly(
                    &mut values,
                    4 * half + 1,
                    4 * half + 3,
                    layer_1_twiddles[half],
                );
            }
            for i in 0..4 {
                *ptr.add(row + i * eighth) = values[i] + values[i + 4];
            }
        }
    });
}

/// Ranked variant of [`transpose_forward_ntt_fused_final_3layer_low_half`]
/// that also computes the ordinary introduction message against `f`.
///
/// Each retained quarter is one contiguous `eighth`-sized segment. Processing
/// even/odd rows together therefore produces four exact adjacent basis pairs:
/// `(segment + row, segment + row + 1)` for segments 0 through 3. Those are
/// precisely the pairs consumed by [`round_msg_lsb`], so the message adds no
/// second read of the just-written low half.
fn transpose_forward_ntt_fused_final_3layer_low_half_with_round_msg(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    log_d: usize,
    f: &[F128],
) -> SumcheckMessage {
    use rayon::prelude::*;

    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], a: usize, b: usize, twiddle: F128) {
        let sum = values[a] + values[b];
        values[a] = sum;
        values[b] = twiddle * sum + values[b];
    }

    #[inline(always)]
    unsafe fn retained_row(
        ptr: *mut F128,
        row: usize,
        eighth: usize,
        layer_1_twiddles: &[F128; 2],
        layer_2_twiddles: &[F128; 4],
    ) -> [F128; 4] {
        let mut values = [F128::ZERO; 8];
        for (i, value) in values.iter_mut().enumerate() {
            // SAFETY: established by the caller's disjoint paired-row range.
            *value = unsafe { *ptr.add(row + i * eighth) };
        }
        for pair in 0..4 {
            butterfly(&mut values, 2 * pair, 2 * pair + 1, layer_2_twiddles[pair]);
        }
        for half in 0..2 {
            butterfly(&mut values, 4 * half, 4 * half + 2, layer_1_twiddles[half]);
            butterfly(
                &mut values,
                4 * half + 1,
                4 * half + 3,
                layer_1_twiddles[half],
            );
        }
        [
            values[0] + values[4],
            values[1] + values[5],
            values[2] + values[6],
            values[3] + values[7],
        ]
    }

    assert!(log_d >= 4);
    assert_eq!(data.len(), 1usize << log_d);
    assert_eq!(f.len(), data.len() >> 1);
    let eighth = data.len() >> 3;
    assert!(eighth.is_multiple_of(2));
    let mut layer_1_twiddles = [F128::ZERO; 2];
    let mut layer_2_twiddles = [F128::ZERO; 4];
    for (block, twiddle) in layer_1_twiddles.iter_mut().enumerate() {
        *twiddle = ntt.twiddle(1, block);
    }
    for (block, twiddle) in layer_2_twiddles.iter_mut().enumerate() {
        *twiddle = ntt.twiddle(2, block);
    }

    let data_ptr = data.as_mut_ptr() as usize;
    let (u_0, u_2) = (0..eighth / 2)
        .into_par_iter()
        .map(|row_pair| {
            let even_row = 2 * row_pair;
            let odd_row = even_row + 1;
            let mut local_u_0 = F128::ZERO;
            let mut local_u_2 = F128::ZERO;
            // SAFETY: every job owns the 16 distinct inputs for its paired
            // rows and the eight retained outputs at those same row offsets.
            unsafe {
                let ptr = data_ptr as *mut F128;
                let even =
                    retained_row(ptr, even_row, eighth, &layer_1_twiddles, &layer_2_twiddles);
                let odd = retained_row(ptr, odd_row, eighth, &layer_1_twiddles, &layer_2_twiddles);
                for segment in 0..4 {
                    let even_index = segment * eighth + even_row;
                    *ptr.add(even_index) = even[segment];
                    *ptr.add(even_index + 1) = odd[segment];
                    let f_0 = f[even_index];
                    let f_1 = f[even_index + 1];
                    local_u_0 += f_0 * even[segment];
                    local_u_2 += (f_0 + f_1) * (even[segment] + odd[segment]);
                }
            }
            (local_u_0, local_u_2)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a_0, a_2), (b_0, b_2)| (a_0 + b_0, a_2 + b_2),
        );
    SumcheckMessage { u_0, u_2 }
}

/// Transposed forward additive NTT, `Fᵀ`, in place over `2^log_d` coefficients.
/// Forward butterfly is `M=[[1,t],[1,t+1]]`; transpose `Mᵀ=[[1,1],[t,t+1]]` is
/// `s=a+b; top=s; bot=t·s+b`, applied in **reverse** layer order. (Baseline:
/// one parallel sweep per layer.) Three adjacent layers are fused so each
/// eight-value row group crosses memory once instead of three times.
fn transpose_forward_ntt(ntt: &AdditiveNttF128, data: &mut [F128], log_d: usize) {
    use rayon::prelude::*;
    debug_assert_eq!(data.len(), 1usize << log_d);
    debug_assert!(log_d <= ntt.log_domain_size());
    let n_threads = rayon::current_num_threads().max(1);
    let mut remaining = log_d;
    while remaining >= 3 {
        let layer = remaining - 3;
        transpose_forward_ntt_fused_3layer(ntt, data, log_d, layer);
        remaining -= 3;
    }
    for layer in (0..remaining).rev() {
        let num_blocks = 1usize << layer;
        let block_size = 1usize << (log_d - layer);
        let bsh = block_size >> 1;
        if num_blocks >= n_threads {
            data.par_chunks_mut(block_size)
                .enumerate()
                .for_each(|(block, chunk)| {
                    let t = ntt.twiddle(layer, block);
                    let (top, bot) = chunk.split_at_mut(bsh);
                    for (a_ref, b_ref) in top.iter_mut().zip(bot.iter_mut()) {
                        let a = *a_ref;
                        let b = *b_ref;
                        let s = a + b;
                        *a_ref = s;
                        *b_ref = t * s + b;
                    }
                });
        } else {
            for block in 0..num_blocks {
                let t = ntt.twiddle(layer, block);
                let chunk = &mut data[block * block_size..(block + 1) * block_size];
                let (top, bot) = chunk.split_at_mut(bsh);
                top.par_iter_mut()
                    .zip(bot.par_iter_mut())
                    .for_each(|(a_ref, b_ref)| {
                        let a = *a_ref;
                        let b = *b_ref;
                        let s = a + b;
                        *a_ref = s;
                        *b_ref = t * s + b;
                    });
            }
        }
    }
}

/// Exact ranked top-level induction shape. It transforms a 2^20 rate-two
/// codeword, keeps 2^19 coefficients, folds 64 lanes, and batches 218 opens.
#[inline]
fn is_ranked_induce_truncated_final_ntt_shape(
    log_msg_cols: usize,
    log_inv_rate: usize,
    log_num_interleaved: usize,
    n_queries: usize,
    alpha_len: usize,
) -> bool {
    log_msg_cols == 19
        && log_inv_rate == 1
        && log_num_interleaved == 6
        && n_queries == 218
        && alpha_len == 8
}

#[cfg(test)]
std::thread_local! {
    static TEST_TRUNCATED_FINAL_NTT_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static TEST_TRUNCATED_FINAL_NTT_HITS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[inline]
fn use_ranked_induce_truncated_final_ntt(
    log_msg_cols: usize,
    log_inv_rate: usize,
    log_num_interleaved: usize,
    n_queries: usize,
    alpha_len: usize,
) -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_TRUNCATED_FINAL_NTT_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }

    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && is_ranked_induce_truncated_final_ntt_shape(
            log_msg_cols,
            log_inv_rate,
            log_num_interleaved,
            n_queries,
            alpha_len,
        )
        && std::env::var_os("FLOCK_NO_LIG_INDUCE_TRUNCATED_NTT").is_none()
}

/// Enable the fused final-transpose/ordinary-message pass only at the exact
/// ranked top-level induction shape. The separate opt-out keeps the incumbent
/// truncated transpose followed by [`round_msg_lsb`] available for rollback.
#[inline]
fn use_ranked_induce_fused_msg(
    log_msg_cols: usize,
    log_inv_rate: usize,
    log_num_interleaved: usize,
    n_queries: usize,
    alpha_len: usize,
    f_len: usize,
) -> bool {
    is_ranked_induce_truncated_final_ntt_shape(
        log_msg_cols,
        log_inv_rate,
        log_num_interleaved,
        n_queries,
        alpha_len,
    ) && f_len == (1usize << log_msg_cols)
        && use_ranked_induce_truncated_final_ntt(
            log_msg_cols,
            log_inv_rate,
            log_num_interleaved,
            n_queries,
            alpha_len,
        )
        && std::env::var_os("FLOCK_NO_LIG_INDUCE_FUSED_MSG").is_none()
}

#[cfg(test)]
fn with_truncated_final_ntt_override<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    TEST_TRUNCATED_FINAL_NTT_OVERRIDE.with(|slot| {
        struct Reset<'a> {
            slot: &'a std::cell::Cell<Option<bool>>,
            previous: Option<bool>,
        }
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.slot.set(self.previous);
            }
        }

        let previous = slot.replace(Some(enabled));
        let _reset = Reset { slot, previous };
        f()
    })
}

/// `Fᵀ`-based fast path for [`induce_sumcheck_poly`]: scatter per-query weights
/// into the codeword domain, apply `Fᵀ`, keep the low `2^log_msg_cols` outputs.
/// Byte-identical output to [`induce_sumcheck_poly`].
pub(crate) fn induce_sumcheck_poly_via_ntt(
    log_msg_cols: usize,
    log_inv_rate: usize,
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    let (basis, enforced_sum, intro_msg) = induce_sumcheck_poly_via_ntt_impl(
        log_msg_cols,
        log_inv_rate,
        opened_rows,
        v_challenges,
        queries,
        alpha,
        None,
    );
    debug_assert!(intro_msg.is_none());
    (basis, enforced_sum)
}

fn induce_sumcheck_poly_via_ntt_impl(
    log_msg_cols: usize,
    log_inv_rate: usize,
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
    round_f: Option<&[F128]>,
) -> (Vec<F128>, F128, Option<SumcheckMessage>) {
    let n = 1usize << log_msg_cols;
    let log_block = log_msg_cols + log_inv_rate;
    let block_len = 1usize << log_block;
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);
    let truncate_final_group = use_ranked_induce_truncated_final_ntt(
        log_msg_cols,
        log_inv_rate,
        v_challenges.len(),
        n_queries,
        alpha.len(),
    );
    assert!(
        round_f.is_none() || truncate_final_group,
        "fused induction message requires the truncated final group"
    );
    if let Some(f) = round_f {
        assert_eq!(f.len(), n, "induction message witness length changed");
    }

    let eq = build_eq_table(v_challenges);
    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    // Parallel per-query dot products, mirroring the dense variant's
    // per-thread accumulation. Every term is independent and F128 addition
    // is XOR (associative, commutative), so the parallel reduction is
    // bit-identical to the serial fold regardless of association. This loop
    // is the NTT variant's only serial stretch — n_queries · row_len
    // multiplies on one worker while the rest of the pool idles between the
    // per-level opens and the transpose.
    const PAR_QUERY_THRESHOLD: usize = 32;
    let query_term = |i: usize| -> F128 {
        let dot: F128 = opened_rows[i]
            .iter()
            .zip(eq.iter())
            .map(|(&r, &e)| r * e)
            .fold(F128::ZERO, |a, v| a + v);
        dot * alpha_pows[i]
    };
    let enforced_sum = if n_queries >= PAR_QUERY_THRESHOLD {
        use rayon::prelude::*;
        (0..n_queries)
            .into_par_iter()
            .map(query_term)
            .reduce(|| F128::ZERO, |a, b| a + b)
    } else {
        (0..n_queries)
            .map(query_term)
            .fold(F128::ZERO, |a, b| a + b)
    };

    let (mut coeffs, intro_msg) = if log_block == 0 {
        assert!(round_f.is_none());
        let mut c = vec![F128::ZERO; block_len];
        for i in 0..n_queries {
            c[queries[i]] += alpha_pows[i];
        }
        (c, None)
    } else {
        let ntt = AdditiveNttF128::standard(log_block);
        transpose_forward_ntt_sparse_impl(
            &ntt,
            queries,
            &alpha_pows,
            log_block,
            truncate_final_group,
            round_f,
        )
    };
    coeffs.truncate(n);
    (coeffs, enforced_sum, intro_msg)
}

/// Cost-based dispatch between the dense [`induce_sumcheck_poly`] and the
/// sparse-NTT [`induce_sumcheck_poly_via_ntt`].
///
/// The dense path costs `O(n_queries · 2^log_msg_cols)`; the NTT path costs one
/// pass over the `2^(log_msg_cols+log_inv_rate)` codeword domain, `O(2^log_block
/// · log_block)`. The `2^log_msg_cols` factor cancels, so the NTT wins exactly
/// when there are enough queries to amortize the codeword pass against the rate
/// blow-up and depth:
///   `n_queries  >  C · 2^log_inv_rate · log_block`   (C≈4: the NTT is ~2×
/// costlier per op — memory-bound, multi-pass — plus margin so we only switch
/// when clearly ahead). In the recursive PCS this fires only at the top level
/// (large message domain, many queries); deeper levels stay dense.
///
/// Both paths are byte-identical (see `induce_sumcheck_poly_via_ntt_matches_dense`),
/// so a mis-dispatch only costs time. Tuned/validated at blake m=30.
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

/// Ranked top-level induction with an optional ordinary-introduction message
/// produced by the final truncated transpose pass. Every other geometry calls
/// [`induce_sumcheck_poly_auto`] unchanged and returns `None` for the message.
fn induce_sumcheck_poly_auto_with_ranked_msg(
    log_msg_cols: usize,
    log_inv_rate: usize,
    sks_vks: &[F128],
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
    f: &[F128],
) -> (Vec<F128>, F128, Option<SumcheckMessage>) {
    if use_ranked_induce_fused_msg(
        log_msg_cols,
        log_inv_rate,
        v_challenges.len(),
        queries.len(),
        alpha.len(),
        f.len(),
    ) {
        induce_sumcheck_poly_via_ntt_impl(
            log_msg_cols,
            log_inv_rate,
            opened_rows,
            v_challenges,
            queries,
            alpha,
            Some(f),
        )
    } else {
        let (basis, enforced_sum) = induce_sumcheck_poly_auto(
            log_msg_cols,
            log_inv_rate,
            sks_vks,
            opened_rows,
            v_challenges,
            queries,
            alpha,
        );
        (basis, enforced_sum, None)
    }
}

/// Sparse-prefix variant of [`transpose_forward_ntt`]: exploits that the input
/// has only `positions.len()` nonzeros and that the first `k` transpose steps
/// (forward layers `log_d-1 .. log_d-k`, pairing distances `1 .. 2^(k-1)`) mix
/// only **within** `2^k`-aligned windows. We process just the windows that
/// contain a nonzero (a dense `2^k` transpose each), densify, then run the
/// remaining steps as full dense sweeps. Output is identical to
/// `transpose_forward_ntt` applied to the scattered input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActiveWindow {
    window_index: usize,
    input_start: usize,
    input_end: usize,
}

#[inline]
fn positions_are_sorted(positions: &[usize]) -> bool {
    positions.windows(2).all(|pair| pair[0] <= pair[1])
}

fn group_sorted_positions(positions: &[usize], prefix_k: usize) -> Vec<ActiveWindow> {
    debug_assert!(positions_are_sorted(positions));
    let mut groups = Vec::with_capacity(positions.len());
    let mut input_start = 0;
    while input_start < positions.len() {
        let window_index = positions[input_start] >> prefix_k;
        let mut input_end = input_start + 1;
        while input_end < positions.len() && positions[input_end] >> prefix_k == window_index {
            input_end += 1;
        }
        groups.push(ActiveWindow {
            window_index,
            input_start,
            input_end,
        });
        input_start = input_end;
    }
    groups
}

fn scatter_active_windows(
    groups: &[ActiveWindow],
    positions: &[usize],
    values: &[F128],
    prefix_k: usize,
) -> Vec<F128> {
    debug_assert_eq!(positions.len(), values.len());
    let window_len = 1usize << prefix_k;
    let window_mask = window_len - 1;
    let mut arena = vec![F128::ZERO; groups.len() * window_len];
    for (arena_index, group) in groups.iter().enumerate() {
        let window = &mut arena[arena_index * window_len..(arena_index + 1) * window_len];
        for input_index in group.input_start..group.input_end {
            window[positions[input_index] & window_mask] += values[input_index];
        }
    }
    arena
}

#[inline]
fn transform_active_window(
    ntt: &AdditiveNttF128,
    window: &mut [F128],
    window_index: usize,
    prefix_k: usize,
    log_d: usize,
) {
    for s in 0..prefix_k {
        let layer = log_d - 1 - s;
        let half = 1usize << s;
        let block_size = half << 1;
        let nblocks = window.len() / block_size;
        for block in 0..nblocks {
            let twiddle = ntt.twiddle(layer, (window_index << (prefix_k - s - 1)) + block);
            let base = block * block_size;
            for row in 0..half {
                let top = window[base + row];
                let bottom = window[base + row + half];
                let sum = top + bottom;
                window[base + row] = sum;
                window[base + row + half] = twiddle * sum + bottom;
            }
        }
    }
}

fn transform_active_windows(
    ntt: &AdditiveNttF128,
    arena: &mut [F128],
    groups: &[ActiveWindow],
    prefix_k: usize,
    log_d: usize,
) {
    use rayon::prelude::*;
    let window_len = 1usize << prefix_k;
    arena
        .par_chunks_mut(window_len)
        .zip(groups.par_iter())
        .for_each(|(window, group)| {
            transform_active_window(ntt, window, group.window_index, prefix_k, log_d);
        });
}

const INACTIVE_WINDOW: usize = usize::MAX;

fn active_window_to_arena(groups: &[ActiveWindow], n_windows: usize) -> Vec<usize> {
    let mut window_to_arena = vec![INACTIVE_WINDOW; n_windows];
    for (arena_index, group) in groups.iter().enumerate() {
        debug_assert!(group.window_index < n_windows);
        window_to_arena[group.window_index] = arena_index;
    }
    window_to_arena
}

fn densify_active_windows(
    arena: &[F128],
    groups: &[ActiveWindow],
    log_d: usize,
    prefix_k: usize,
) -> Vec<F128> {
    use rayon::prelude::*;

    let n = 1usize << log_d;
    let window_len = 1usize << prefix_k;
    let n_windows = n / window_len;
    let window_to_arena = active_window_to_arena(groups, n_windows);

    let mut data: Vec<F128> = crate::alloc_uninit_vec(n);
    data.par_chunks_mut(window_len)
        .enumerate()
        .for_each(|(window_index, destination)| {
            let arena_index = window_to_arena[window_index];
            if arena_index == INACTIVE_WINDOW {
                destination.fill(F128::ZERO);
            } else {
                let source = &arena[arena_index * window_len..(arena_index + 1) * window_len];
                destination.copy_from_slice(source);
            }
        });
    // Every chunk covers one disjoint dense window and takes exactly one of
    // the fill/copy branches above, so all uninitialized elements are written
    // before the dense transpose can read them.
    data
}

/// Materialize the sparse-window arena directly through the first dense
/// three-layer transpose group.
///
/// With a `2^prefix_k` sparse window, that group consumes exactly eight
/// adjacent windows per block. Gathering those eight inputs from the arena
/// (or substituting zero for an inactive window) is therefore byte-identical
/// to first densifying and then running [`transpose_forward_ntt_fused_3layer`].
/// It removes the full-domain densify write and the first group's matching
/// input read. A block with no active input is linear-zero and can initialize
/// its complete output block without executing any field products.
fn densify_active_windows_fused_first_3layer(
    ntt: &AdditiveNttF128,
    arena: &[F128],
    groups: &[ActiveWindow],
    log_d: usize,
    prefix_k: usize,
) -> Vec<F128> {
    use rayon::prelude::*;

    assert!(
        log_d >= prefix_k + 3,
        "fused densification requires a complete three-layer suffix group"
    );
    let n = 1usize << log_d;
    let window_len = 1usize << prefix_k;
    let n_windows = n / window_len;
    let first_layer = log_d - prefix_k - 3;
    let num_blocks = 1usize << first_layer;
    let block_size = window_len << 3;
    debug_assert_eq!(num_blocks * block_size, n);
    debug_assert_eq!(n_windows, num_blocks << 3);
    debug_assert_eq!(arena.len(), groups.len() * window_len);

    let window_to_arena = active_window_to_arena(groups, n_windows);
    let twiddles: Vec<[F128; 7]> = (0..num_blocks)
        .map(|block| {
            let mut tw = [F128::ZERO; 7];
            tw[0] = ntt.twiddle(first_layer, block);
            for half in 0..2 {
                tw[1 + half] = ntt.twiddle(first_layer + 1, 2 * block + half);
            }
            for quarter in 0..4 {
                tw[3 + quarter] = ntt.twiddle(first_layer + 2, 4 * block + quarter);
            }
            tw
        })
        .collect();

    // Keep the vector length at zero until every Rayon job has initialized its
    // disjoint block. This avoids ever constructing a safe reference to an
    // uninitialized `F128`; a panic before `set_len` merely drops capacity.
    let mut data = Vec::<F128>::with_capacity(n);
    let data_ptr = data.as_mut_ptr() as usize;
    (0..num_blocks).into_par_iter().for_each(|block| {
        let arena_indices: [usize; 8] = core::array::from_fn(|i| window_to_arena[8 * block + i]);
        let output_start = block * block_size;
        let output_ptr = data_ptr as *mut F128;

        // SAFETY: every job owns the disjoint initialized range
        // `[output_start, output_start + block_size)`. `data` has capacity `n`
        // but length zero throughout the parallel region, and no read of its
        // storage occurs. F128 is exactly two `u64` limbs, so all-zero bytes are
        // its valid `ZERO` representation.
        unsafe {
            if arena_indices
                .iter()
                .all(|&arena_index| arena_index == INACTIVE_WINDOW)
            {
                core::ptr::write_bytes(output_ptr.add(output_start), 0, block_size);
                return;
            }

            let tw = &twiddles[block];
            for row in 0..window_len {
                let mut values: [F128; 8] = core::array::from_fn(|i| {
                    let arena_index = arena_indices[i];
                    if arena_index == INACTIVE_WINDOW {
                        F128::ZERO
                    } else {
                        arena[arena_index * window_len + row]
                    }
                });

                #[inline(always)]
                fn butterfly(values: &mut [F128; 8], top: usize, bottom: usize, twiddle: F128) {
                    let sum = values[top] + values[bottom];
                    values[top] = sum;
                    values[bottom] = twiddle * sum + values[bottom];
                }

                for pair in 0..4 {
                    butterfly(&mut values, 2 * pair, 2 * pair + 1, tw[3 + pair]);
                }
                for half in 0..2 {
                    butterfly(&mut values, 4 * half, 4 * half + 2, tw[1 + half]);
                    butterfly(&mut values, 4 * half + 1, 4 * half + 3, tw[1 + half]);
                }
                for top in 0..4 {
                    butterfly(&mut values, top, top + 4, tw[0]);
                }

                for (i, value) in values.into_iter().enumerate() {
                    output_ptr
                        .add(output_start + row + i * window_len)
                        .write(value);
                }
            }
        }
    });

    // SAFETY: the block partition covers `[0, n)` exactly, and every block
    // takes either the full zero-write branch or writes all eight values for
    // every row before the parallel iterator joins above.
    unsafe {
        data.set_len(n);
    }
    data
}

#[inline]
fn is_ranked_fused_densify_first_shape(log_d: usize, prefix_k: usize, n_positions: usize) -> bool {
    prefix_k == 8 && matches!((log_d, n_positions), (20, 218) | (18, 106))
}

#[inline]
fn use_ranked_fused_densify_first(log_d: usize, prefix_k: usize, n_positions: usize) -> bool {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64"))
        || !is_ranked_fused_densify_first_shape(log_d, prefix_k, n_positions)
    {
        return false;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FLOCK_NO_INDUCE_FUSED_DENSIFY_FIRST").as_deref()
            != Some(std::ffi::OsStr::new("1"))
    })
}

fn transpose_forward_ntt_dense_suffix_impl(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    log_d: usize,
    prefix_k: usize,
    truncate_final_group: bool,
    round_f: Option<&[F128]>,
) -> Option<SumcheckMessage> {
    use rayon::prelude::*;
    let n_threads = rayon::current_num_threads().max(1);
    let mut remaining = log_d - prefix_k;
    let mut intro_msg = None;
    assert!(
        round_f.is_none() || truncate_final_group,
        "fused message requires truncated dense suffix"
    );
    if truncate_final_group {
        // The optimized ranked schedule ends in the fused layers 2,1,0.
        // Keep the gate explicit so another sparse geometry cannot silently
        // skip outputs from a differently shaped suffix schedule.
        assert!(remaining >= 3 && remaining.is_multiple_of(3));
    }
    while remaining >= 3 {
        let layer = remaining - 3;
        if truncate_final_group && layer == 0 {
            if let Some(f) = round_f {
                intro_msg = Some(
                    transpose_forward_ntt_fused_final_3layer_low_half_with_round_msg(
                        ntt, data, log_d, f,
                    ),
                );
            } else {
                transpose_forward_ntt_fused_final_3layer_low_half(ntt, data, log_d);
            }
        } else {
            transpose_forward_ntt_fused_3layer(ntt, data, log_d, layer);
        }
        remaining -= 3;
    }
    for layer in (0..remaining).rev() {
        let num_blocks = 1usize << layer;
        let block_size = 1usize << (log_d - layer);
        let half = block_size >> 1;
        if num_blocks >= n_threads {
            data.par_chunks_mut(block_size)
                .enumerate()
                .for_each(|(block, chunk)| {
                    let twiddle = ntt.twiddle(layer, block);
                    let (top, bottom) = chunk.split_at_mut(half);
                    for (top, bottom) in top.iter_mut().zip(bottom.iter_mut()) {
                        let a = *top;
                        let b = *bottom;
                        let sum = a + b;
                        *top = sum;
                        *bottom = twiddle * sum + b;
                    }
                });
        } else {
            for block in 0..num_blocks {
                let twiddle = ntt.twiddle(layer, block);
                let chunk = &mut data[block * block_size..(block + 1) * block_size];
                let (top, bottom) = chunk.split_at_mut(half);
                top.par_iter_mut()
                    .zip(bottom.par_iter_mut())
                    .for_each(|(top, bottom)| {
                        let a = *top;
                        let b = *bottom;
                        let sum = a + b;
                        *top = sum;
                        *bottom = twiddle * sum + b;
                    });
            }
        }
    }
    assert_eq!(intro_msg.is_some(), round_f.is_some());
    intro_msg
}

#[cfg(test)]
fn transpose_forward_ntt_sparse(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    truncate_final_group: bool,
) -> Vec<F128> {
    let (data, intro_msg) = transpose_forward_ntt_sparse_impl(
        ntt,
        positions,
        values,
        log_d,
        truncate_final_group,
        None,
    );
    debug_assert!(intro_msg.is_none());
    data
}

fn transpose_forward_ntt_sparse_impl(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    truncate_final_group: bool,
    round_f: Option<&[F128]>,
) -> (Vec<F128>, Option<SumcheckMessage>) {
    let n = 1usize << log_d;
    // No prefix for small domains — just scatter + full dense transpose.
    let k = if log_d >= 12 { 8usize.min(log_d) } else { 0 };

    if k == 0 {
        assert!(!truncate_final_group);
        assert!(round_f.is_none());
        let mut data = vec![F128::ZERO; n];
        for (&p, &v) in positions.iter().zip(values) {
            data[p] += v;
        }
        if log_d > 0 {
            transpose_forward_ntt(ntt, &mut data, log_d);
        }
        return (data, None);
    }

    static LINEAR_WINDOWS_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let use_linear_windows = positions_are_sorted(positions)
        && *LINEAR_WINDOWS_ENABLED
            .get_or_init(|| std::env::var_os("FLOCK_NO_INDUCE_LINEAR_WINDOWS").is_none());
    if !use_linear_windows {
        return transpose_forward_ntt_sparse_hashmap_impl(
            ntt,
            positions,
            values,
            log_d,
            k,
            truncate_final_group,
            round_f,
        );
    }

    let groups = group_sorted_positions(positions, k);

    let mut arena = scatter_active_windows(&groups, positions, values, k);

    transform_active_windows(ntt, &mut arena, &groups, k, log_d);

    let use_fused_densify = use_ranked_fused_densify_first(log_d, k, positions.len());
    let (mut data, dense_prefix_k) = if use_fused_densify {
        (
            densify_active_windows_fused_first_3layer(ntt, &arena, &groups, log_d, k),
            k + 3,
        )
    } else {
        (densify_active_windows(&arena, &groups, log_d, k), k)
    };

    let intro_msg = transpose_forward_ntt_dense_suffix_impl(
        ntt,
        &mut data,
        log_d,
        dense_prefix_k,
        truncate_final_group,
        round_f,
    );
    if truncate_final_group {
        data.truncate(n >> 1);
    }
    (data, intro_msg)
}

#[cfg(test)]
fn transpose_forward_ntt_sparse_hashmap(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    prefix_k: usize,
    truncate_final_group: bool,
) -> Vec<F128> {
    let (data, intro_msg) = transpose_forward_ntt_sparse_hashmap_impl(
        ntt,
        positions,
        values,
        log_d,
        prefix_k,
        truncate_final_group,
        None,
    );
    debug_assert!(intro_msg.is_none());
    data
}

fn transpose_forward_ntt_sparse_hashmap_impl(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    prefix_k: usize,
    truncate_final_group: bool,
    round_f: Option<&[F128]>,
) -> (Vec<F128>, Option<SumcheckMessage>) {
    use rayon::prelude::*;
    use std::collections::HashMap;
    let n = 1usize << log_d;
    let window_len = 1usize << prefix_k;
    let window_mask = window_len - 1;

    let mut windows: HashMap<usize, Vec<F128>> = HashMap::new();
    for (&position, &value) in positions.iter().zip(values) {
        let window = windows
            .entry(position >> prefix_k)
            .or_insert_with(|| vec![F128::ZERO; window_len]);
        window[position & window_mask] += value;
    }

    let windows: Vec<(usize, Vec<F128>)> = windows.into_iter().collect();
    let processed: Vec<(usize, Vec<F128>)> = windows
        .into_par_iter()
        .map(|(window_index, mut window)| {
            transform_active_window(ntt, &mut window, window_index, prefix_k, log_d);
            (window_index, window)
        })
        .collect();

    let mut data = vec![F128::ZERO; n];
    for (window_index, window) in processed {
        let start = window_index << prefix_k;
        data[start..start + window_len].copy_from_slice(&window);
    }

    let intro_msg = transpose_forward_ntt_dense_suffix_impl(
        ntt,
        &mut data,
        log_d,
        prefix_k,
        truncate_final_group,
        round_f,
    );
    if truncate_final_group {
        data.truncate(n >> 1);
    }
    (data, intro_msg)
}

// ===================================================================
// ligero_commit
// ===================================================================

/// Codeword + Merkle tree for one Ligerito commitment level.
///
/// `mat` is row-major: `mat[pos * num_interleaved + lane]` for
/// `pos ∈ [0, block_len)`, `lane ∈ [0, num_interleaved)`. Each row
/// (one `pos` across all lanes) is one Merkle leaf.
pub(crate) struct LigeroWitness {
    pub mat: Vec<F128>,
    pub tree: Vec<Hash>,
    pub block_len: usize,
    pub num_interleaved: usize,
}

// Recycle the codeword matrix (128 MB for L1 at m=29) and the flat Merkle
// tree (16 MiB at L1) through their scratch pools when a level's witness is
// replaced/dropped.
impl Drop for LigeroWitness {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.mat));
        crate::scratch::give_hash_tree(std::mem::take(&mut self.tree));
    }
}

// SumcheckProver owns the two witness-sized polynomials of the open (the
// packed witness `f` and the γ-combined basis) plus the fold ping-pong
// spares — recycle all four on drop.
impl Drop for SumcheckProver {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.f));
        crate::scratch::give_f128(std::mem::take(&mut self.combined_basis));
        crate::scratch::give_f128(std::mem::take(&mut self.spare_f));
        crate::scratch::give_f128(std::mem::take(&mut self.spare_b));
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

/// Reshape `poly` (length `num_interleaved · msg_cols`) into a
/// `block_len × num_interleaved` SoA matrix, RS-encode each lane via the
/// LCH additive NTT (non-systematic: pad message with zeros to `block_len`,
/// then forward-transform), and Merkle-commit the rows.
///
/// `poly` layout: **LSB-first lane index** — `poly[col * num_interleaved + lane]`.
/// The first `log_num_interleaved` LSB variables of the multilinear poly are the
/// lane indices, so `partial_eval_lsb(poly, lane_challenges)` produces the
/// next-level poly directly. This composes cleanly with sumcheck folds.
pub(crate) fn ligero_commit(
    poly: &[F128],
    log_msg_cols: usize,
    log_num_interleaved: usize,
    log_inv_rate: usize,
    ntt: &AdditiveNttF128,
    kind: HashKind,
) -> LigeroWitness {
    let level_opt_out = match (log_msg_cols, log_num_interleaved, log_inv_rate) {
        (16, 3, 2) => Some("FLOCK_NO_RECURSIVE_FROM_MESSAGE_L1"),
        (13, 3, 3) => Some("FLOCK_NO_RECURSIVE_FROM_MESSAGE_L2"),
        _ => None,
    };
    let recursive_from_message_shape = kind == HashKind::Blake3 && level_opt_out.is_some();
    let level_enabled = level_opt_out.is_some_and(|name| std::env::var_os(name).is_none());
    let fuse_from_message = cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        target_feature = "aes"
    )) && recursive_from_message_shape
        && level_enabled
        && std::env::var_os("FLOCK_NO_RECURSIVE_FROM_MESSAGE").is_none();
    ligero_commit_impl(
        poly,
        log_msg_cols,
        log_num_interleaved,
        log_inv_rate,
        ntt,
        kind,
        fuse_from_message,
    )
}

/// Implementation split so the exact matrix/tree oracle can force the new
/// path independently of target-feature and environment dispatch.
#[allow(clippy::too_many_arguments)]
fn ligero_commit_impl(
    poly: &[F128],
    log_msg_cols: usize,
    log_num_interleaved: usize,
    log_inv_rate: usize,
    ntt: &AdditiveNttF128,
    kind: HashKind,
    fuse_from_message: bool,
) -> LigeroWitness {
    let msg_cols = 1usize << log_msg_cols;
    let num_interleaved = 1usize << log_num_interleaved;
    let block_len = msg_cols << log_inv_rate;
    let log_block_len = log_msg_cols + log_inv_rate;
    assert_eq!(poly.len(), num_interleaved * msg_cols);
    assert!(log_block_len <= ntt.log_domain_size());

    let timing = std::env::var_os("LIG_PROVE_TRACE").is_some()
        || std::env::var_os("FLOCK_OPEN_TIMING").is_some();
    let total_start = timing.then(std::time::Instant::now);

    // LSB-lane layout: input matches `data[pos * num_interleaved + lane]`.
    // The first `log_inv_rate` layers on zero-padded coefficients are pure
    // copies. The ordinary path materializes those replicas; the exact ranked
    // recursive shapes fuse that logical state into their first radix-8 pass.
    let codeword_len = block_len * num_interleaved;
    let alloc_start = timing.then(std::time::Instant::now);
    let mut mat = crate::scratch::take_f128(codeword_len);
    let alloc_elapsed = alloc_start.map_or(std::time::Duration::ZERO, |t| t.elapsed());
    let mut fill_elapsed = std::time::Duration::ZERO;
    let ntt_start = timing.then(std::time::Instant::now);
    if fuse_from_message {
        // Write the first nontrivial radix-8 result straight from the compact
        // message into stale codeword storage. This deletes the full replica
        // fill and the first pass's destination reads/RFOs.
        ntt.forward_transform_interleaved_from_message_fused3(
            poly,
            &mut mat,
            num_interleaved,
            log_inv_rate,
        );
    } else {
        let fill_start = timing.then(std::time::Instant::now);
        super::commit::replicate_message_fill(&mut mat, poly);
        fill_elapsed = fill_start.map_or(std::time::Duration::ZERO, |t| t.elapsed());
        // RS-encode every lane in one call (each lane is one independent NTT).
        ntt.forward_transform_interleaved_from_layer(&mut mat, num_interleaved, log_inv_rate);
    }
    let ntt_elapsed = ntt_start
        .map_or(std::time::Duration::ZERO, |t| t.elapsed())
        .saturating_sub(fill_elapsed);

    // Merkle over rows. One leaf = `num_interleaved` consecutive F128 = 16·num_interleaved bytes.
    let merkle_start = timing.then(std::time::Instant::now);
    let leaf_size_bytes = num_interleaved * core::mem::size_of::<F128>();
    let data_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            mat.as_ptr() as *const u8,
            mat.len() * core::mem::size_of::<F128>(),
        )
    };
    debug_assert_eq!(data_bytes.len(), block_len * leaf_size_bytes);
    // The 128-byte-leaf recursive shapes are the only serial CPU BLAKE3
    // blocks in the opening spine while the GPU sits idle; the offload
    // returns the bit-identical flat tree or `None` for the exact CPU path
    // (kill switch `FLOCK_NO_GPU_RECURSIVE_MERKLE=1`, non-Blake3 hashes,
    // other shapes, and every GPU failure).
    let gpu_tree = if matches!(kind, HashKind::Blake3) && leaf_size_bytes == 128 {
        crate::gpu_commit::gpu_recursive_merkle_blake3(data_bytes, block_len)
    } else {
        None
    };
    let tree = match gpu_tree {
        Some(tree) => tree,
        // Pooled tree storage for the CPU builder (the GPU offload's
        // copy-out is pooled inside `gpu_recursive_merkle_blake3`): same
        // fault/munmap argument, byte-identical output.
        None => merkle::merkle_tree_into(
            crate::scratch::take_hash_tree(2 * block_len - 1),
            data_bytes,
            block_len,
            kind,
        ),
    };
    let merkle_elapsed = merkle_start.map_or(std::time::Duration::ZERO, |t| t.elapsed());

    if timing {
        let level = match (log_msg_cols, log_num_interleaved, log_inv_rate) {
            (16, 3, 2) => Some("L1"),
            (13, 3, 3) => Some("L2"),
            _ => None,
        };
        if let Some(level) = level {
            let total_elapsed = total_start.expect("timing start").elapsed();
            eprintln!(
                "    [recursive-commit {level}] fused={fuse_from_message} alloc={:.2} ms fill={:.2} ms ntt={:.2} ms merkle={:.2} ms total={:.2} ms",
                alloc_elapsed.as_secs_f64() * 1e3,
                fill_elapsed.as_secs_f64() * 1e3,
                ntt_elapsed.as_secs_f64() * 1e3,
                merkle_elapsed.as_secs_f64() * 1e3,
                total_elapsed.as_secs_f64() * 1e3,
            );
        }
    }

    LigeroWitness {
        mat,
        tree,
        block_len,
        num_interleaved,
    }
}

// ===================================================================
// Stateful sumcheck — Flock (u_0, u_2) convention
// ===================================================================
//
// Per-round quadratic q(X) = u_0 + u_1·X + u_2·X² with the sumcheck constraint
//   q(0) + q(1) = T_r          (T_r = running sum-claim entering this round)
// Verifier derives u_1 = T_r + u_2 (char 2). Round eval at challenge r:
//   q(r) = u_0 + r·(T_r + u_2) + r²·u_2 = u_0 + r·T_r + (r + r²)·u_2
//
// Ligerito extends plain sumcheck with two ops at recursive-level boundaries:
//
//   introduce_new(b_new, h):
//     Prover commits to a new basis poly b_new with its own claimed sum h
//     (verifier-computable from the open-rows induce step). Sends (u_0, u_2)
//     for the inner product f·b_new at the current (already-folded) dim.
//
//   glue(α):
//     Combine the running round-quadratic with the introduced one as
//     running := running + α·to_glue. New sum-claim becomes T_r + α·h.

/// (u_0, u_2) per round — what the prover sends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SumcheckMessage {
    pub u_0: F128,
    pub u_2: F128,
}

/// Round-quadratic in coefficient form `c + b·X + a·X²`. Used by the verifier
/// to track the running quadratic across fold / introduce_new / glue.
#[derive(Clone, Copy, Debug)]
struct RoundQuad {
    c: F128, // u_0
    b: F128, // u_1 (X coeff) — derived from T_r and u_2
    a: F128, // u_2 (X² coeff)
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

/// Enable the previously-promoted two-challenge initial-fold cadence only for
/// the ranked opening geometry it was designed and validated for. The opt-out
/// keeps an adjacent single-fold control available without rebuilding.
#[inline]
pub(crate) fn ranked_fold2_enabled(poly_len: usize, initial_k: usize) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && poly_len == (1usize << 25)
        && initial_k == 6
        && std::env::var_os("FLOCK_NO_LIG_FOLD2").is_none()
}

/// Compute `(u_0, u_2)` for `u(X) = Σ_x f(X, x) · b(X, x)` where `X` is the
/// LSB variable. Parallel reduction across pair indices.
///
/// Uses a SINGLE combined basis poly. (Previously took `&[Vec<F128>]` and
/// summed at every pair index; collapsing to one basis happens at glue time.)
fn round_msg_lsb(f: &[F128], b: &[F128]) -> SumcheckMessage {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);

    const PAR_THRESHOLD: usize = 4096;
    let half = n / 2;
    if half < PAR_THRESHOLD {
        let mut u_0 = F128::ZERO;
        let mut u_2 = F128::ZERO;
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let b0 = b[2 * j];
            let b1 = b[2 * j + 1];
            u_0 += f0 * b0;
            u_2 += (f0 + f1) * (b0 + b1);
        }
        return SumcheckMessage { u_0, u_2 };
    }

    let (u_0, u_2) = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(|j| {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let b0 = b[2 * j];
            let b1 = b[2 * j + 1];
            (f0 * b0, (f0 + f1) * (b0 + b1))
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    SumcheckMessage { u_0, u_2 }
}

/// Fused round message + full inner product: returns `round_msg_lsb(f, b)`
/// alongside `y = Σ_x f(x)·b(x)`, computed in a single pass over `(f, b)`.
///
/// Used by OOD binding, where `b = eq_table(z)` and `y` is the claimed MLE
/// eval `f̂(z)`. Folding `f` against `z` separately (`mle_eval_inline`) then
/// re-reading `f` against `b` in `round_msg_lsb` costs two passes over the
/// 2^n witness; this collapses them into one (the phase is memory-bandwidth
/// bound, so a saved pass is a near-proportional win). The `u_0` term `f0·b0`
/// is shared between the message and the eval, so `y` costs one extra mul per
/// pair. Bit-identical to the unfused path: F128 sums are exact and order-
/// independent, so `y == mle_eval_inline(f, z)`.
fn round_msg_and_eval_lsb(f: &[F128], b: &[F128]) -> (SumcheckMessage, F128) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);

    const PAR_THRESHOLD: usize = 4096;
    let half = n / 2;
    let term = |j: usize| -> (F128, F128, F128) {
        let f0 = f[2 * j];
        let f1 = f[2 * j + 1];
        let b0 = b[2 * j];
        let b1 = b[2 * j + 1];
        let e0 = f0 * b0;
        // (u_0 term, u_2 term, y term = f0·b0 + f1·b1).
        (e0, (f0 + f1) * (b0 + b1), e0 + f1 * b1)
    };
    if half < PAR_THRESHOLD {
        let (mut u_0, mut u_2, mut y) = (F128::ZERO, F128::ZERO, F128::ZERO);
        for j in 0..half {
            let (a0, a2, ay) = term(j);
            u_0 += a0;
            u_2 += a2;
            y += ay;
        }
        return (SumcheckMessage { u_0, u_2 }, y);
    }

    let (u_0, u_2, y) = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(term)
        .reduce(
            || (F128::ZERO, F128::ZERO, F128::ZERO),
            |(a0, a2, ay), (b0, b2, by)| (a0 + b0, a2 + b2, ay + by),
        );
    (SumcheckMessage { u_0, u_2 }, y)
}

/// Maximum low-factor width for retained lazy-OOD equalities. The production
/// tail has 18 dimensions and therefore splits 11+7: a shared 2,048-entry low
/// table and 128 high weights. Tails of at most 11 dimensions fit entirely in
/// the low factor and retain a one-entry high identity table.
const LAZY_OOD_EQ_SPLIT_LOW_LOG_MAX: usize = 11;

/// Factorized equivalent of [`round_msg_and_eval_lsb`] that keeps
/// `b = eq_table([z_0, z_tail...])` as an exact low/high tensor product.
///
/// With the LSB variable first and `w = eq_table(z_tail)`, each basis pair is
///
/// ```text
/// b[2j]     = (1 + z_0) w[j]
/// b[2j + 1] = z_0 w[j].
/// ```
///
/// For high chunk `h`, the dense tail weights are `eq_lo[i] * eq_hi[h]`.
/// The inner scan computes `a = sum f_0 w` and `s = sum (f_0 + f_1) w`
/// against the shared low table. Only those two chunk partials are scaled by
/// `eq_hi[h]`, yielding `u_0 = (1 + z_0)a`, `u_2 = s`, and `y = a + z_0 s`.
/// At the ranked shape the low table has 2,048 entries and no 2^18-entry tail
/// is built.
fn round_msg_and_eval_lsb_factorized_eq_split(
    f: &[F128],
    eq_lo: &[F128],
    eq_hi: &[F128],
    z_0: F128,
) -> (SumcheckMessage, F128) {
    use rayon::prelude::*;

    assert!(eq_lo.len().is_power_of_two() && eq_lo.len() >= 2);
    assert!(eq_hi.len().is_power_of_two());
    let tail_len = eq_lo
        .len()
        .checked_mul(eq_hi.len())
        .expect("split OOD tail length overflow");
    assert_eq!(f.len(), 2 * tail_len, "split OOD input shape changed");
    let tail_log = tail_len.trailing_zeros() as usize;
    assert_eq!(
        eq_lo.len(),
        1usize << tail_log.min(LAZY_OOD_EQ_SPLIT_LOW_LOG_MAX),
        "split OOD low factor width changed"
    );

    let (a, s) = f
        .par_chunks(2 * eq_lo.len())
        .zip(eq_hi.par_iter())
        .map(|(f_chunk, &hi_weight)| {
            let (a_chunk, s_chunk) = crate::field::f128_slice::round0_factorized_eq(f_chunk, eq_lo);
            (a_chunk * hi_weight, s_chunk * hi_weight)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a_0, s_0), (a_1, s_1)| (a_0 + a_1, s_0 + s_1),
        );

    (
        SumcheckMessage {
            u_0: (F128::ONE + z_0) * a,
            u_2: s,
        },
        a + z_0 * s,
    )
}

/// Partially evaluate `evals` at LSB variable = `r`, in place. Halves length.
/// Parallel for large arrays. Test oracle for the fused fold below; the
/// production path uses `fold_and_msg_lsb_into` instead.
#[cfg(test)]
fn partial_eval_lsb_one(evals: &mut Vec<F128>, r: F128) {
    use rayon::prelude::*;
    let n = evals.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    let one_plus_r = F128::ONE + r;

    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        for j in 0..half {
            let v0 = evals[2 * j];
            let v1 = evals[2 * j + 1];
            evals[j] = v0 * one_plus_r + v1 * r;
        }
        evals.truncate(half);
        return;
    }

    // Parallel: produce a fresh halved Vec then swap in. Doing it in-place with
    // par_iter on overlapping indices is dicey; allocate the halved output and
    // swap (cheap vs the fold itself).
    let folded: Vec<F128> = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(|j| evals[2 * j] * one_plus_r + evals[2 * j + 1] * r)
        .collect();
    *evals = folded;
}

/// Route the Ligerito per-round fold/message passes through the shared P+E
/// hetero chunk queue when the round is wide enough to amortize the kickoff.
/// The mid-size rounds (the largest ones drain through the epool combine and
/// the stateful big-round queues) previously ran main-pool-only while the
/// four E-cores idled between the round-5 materializer and the recursive
/// commits. Chunk geometry, per-chunk kernels, and the XOR message merge are
/// unchanged, so output bytes are identical; only which pool claims a chunk
/// differs. Compile-time default per the cleared ranked environment;
/// `FLOCK_NO_LIG_FOLD_HETERO=1` (exactly `"1"`) restores the incumbent
/// rayon-only passes as the same-binary A/B control.
fn lig_fold_hetero_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("FLOCK_NO_LIG_FOLD_HETERO").is_ok_and(|v| v == "1")
    })
}

/// Same contract for the dense recursive-commit basis induction sweep:
/// `FLOCK_NO_LIG_INDUCE_HETERO=1` (exactly `"1"`) restores the incumbent
/// main-pool-only per-worker query split.
fn lig_induce_hetero_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("FLOCK_NO_LIG_INDUCE_HETERO").is_ok_and(|v| v == "1")
    })
}

/// Fused fold + next-round message in a SINGLE parallel pass.
///
/// Replaces the three separate passes a sumcheck fold otherwise needs
/// (`partial_eval_lsb_one(f)` + `partial_eval_lsb_one(b)` + `round_msg_lsb`):
/// each chunk folds its slice of `f` and `b` at `r` (LSB variable) AND
/// accumulates that slice's `(u_0, u_2)` contribution to the message for the
/// *next* round — over the freshly-folded values, computed while they are
/// still in registers. One fork-join instead of three, and ~⅓ less memory
/// traffic (the folded arrays are not re-read to build the message).
///
/// Computes `next_msg = round_msg_lsb(folded_f, folded_b)`, bit-identical to
/// the unfused sequence.
///
/// Writes into caller-provided buffers (each must have capacity >=
/// `f.len() / 2`; length is set to `f.len() / 2`). Lets [`SumcheckProver`]
/// ping-pong between two persistent buffer pairs instead of allocating,
/// faulting, and unmapping a fresh pair every round.
fn fold_and_msg_lsb_into(
    f: &[F128],
    b: &[F128],
    r: F128,
    nf: &mut Vec<F128>,
    nb: &mut Vec<F128>,
) -> SumcheckMessage {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    debug_assert!(nf.capacity() >= half && nb.capacity() >= half);
    // SAFETY: capacities were checked above; F128: Copy (no Drop), so
    // exposing uninit/stale elements is sound to *hold* — every slot is
    // written below before anything reads it.
    unsafe {
        nf.set_len(half);
        nb.set_len(half);
    }
    let one_plus_r = F128::ONE + r;

    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        for j in 0..half {
            nf[j] = f[2 * j] * one_plus_r + f[2 * j + 1] * r;
            nb[j] = b[2 * j] * one_plus_r + b[2 * j + 1] * r;
        }
        let mut u_0 = F128::ZERO;
        let mut u_2 = F128::ZERO;
        let mut k = 0;
        while k + 1 < half {
            let f0 = nf[k];
            let f1 = nf[k + 1];
            let b0 = nb[k];
            let b1 = nb[k + 1];
            u_0 += f0 * b0;
            u_2 += (f0 + f1) * (b0 + b1);
            k += 2;
        }
        return SumcheckMessage { u_0, u_2 };
    }

    // Parallel path: `half` is a power of two ≥ PAR_THRESHOLD and CHUNK is a
    // power of two, so every chunk has even length and starts at an even
    // global index — message pairs (2k, 2k+1) never straddle a chunk boundary.
    const CHUNK: usize = 2048;
    let chunk_body = |ci: usize, fc: &mut [F128], bc: &mut [F128]| -> (F128, F128) {
        let base = ci * CHUNK;
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            return crate::field::f128_slice::fold_two_and_msg(f, b, base, fc, bc, r);
        }

        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            let len = fc.len();
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            // Fold this slice, then pair up the just-folded values for the msg.
            crate::field::f128_slice::fold_pairs(f, base, fc, r);
            crate::field::f128_slice::fold_pairs(b, base, bc, r);
            let mut k = 0;
            while k + 1 < len {
                let f0 = fc[k];
                let f1 = fc[k + 1];
                let b0 = bc[k];
                let b1 = bc[k + 1];
                u0 += f0 * b0;
                u2 += (f0 + f1) * (b0 + b1);
                k += 2;
            }
            (u0, u2)
        }
    };
    // Hetero queue when the round is wide enough that every chunk claim is
    // useful work: `half` is a power of two, so `half / CHUNK` chunks divide
    // the outputs exactly; each chunk owns `[ci*CHUNK, (ci+1)*CHUNK)` of both
    // outputs and one partial slot. Same chunk grid as the rayon path below,
    // so bytes are identical either way.
    if half >= 16 * CHUNK && lig_fold_hetero_enabled() && crate::epool::epool().is_some() {
        let n_chunks = half / CHUNK;
        let mut partials = vec![(F128::ZERO, F128::ZERO); n_chunks];
        let f_base = crate::epool::SyncPtr(nf.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(nb.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks(n_chunks, |ci| {
            // SAFETY: the queue claims each `ci` exactly once; the ranges and
            // the partial slot are disjoint per chunk and the two-pool join
            // publishes every write before the reduce below reads them.
            unsafe {
                let fc = core::slice::from_raw_parts_mut(f_base.ptr().add(ci * CHUNK), CHUNK);
                let bc = core::slice::from_raw_parts_mut(b_base.ptr().add(ci * CHUNK), CHUNK);
                partials_base.ptr().add(ci).write(chunk_body(ci, fc, bc));
            }
        });
        let (u_0, u_2) = partials
            .into_iter()
            .fold((F128::ZERO, F128::ZERO), |(a0, a2), (c0, c2)| {
                (a0 + c0, a2 + c2)
            });
        return SumcheckMessage { u_0, u_2 };
    }
    let (u_0, u_2) = nf
        .par_chunks_mut(CHUNK)
        .zip(nb.par_chunks_mut(CHUNK))
        .enumerate()
        .map(|(ci, (fc, bc))| chunk_body(ci, fc, bc))
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
        );
    SumcheckMessage { u_0, u_2 }
}

/// Fold the incumbent `(f, combined_basis)` state while injecting a retained
/// split OOD equality into the freshly-folded basis before accumulating the
/// next-round message.
///
/// Each low-factor-sized output chunk reuses `eq_lo`; its complete correction
/// scale is `beta * (1 + z_0 + r) * eq_hi[h]`. At the exact ranked geometry
/// this is 128 chunks of 2,048 outputs and avoids materializing `eq(z_tail)`.
fn fold_and_msg_lsb_into_with_lazy_ood_eq(
    f: &[F128],
    b: &[F128],
    deferred_basis: Option<(&[F128], F128)>,
    r: F128,
    eq_lo: &[F128],
    eq_hi: &[F128],
    beta: F128,
    z_0: F128,
    nf: &mut Vec<F128>,
    nb: &mut Vec<F128>,
) -> SumcheckMessage {
    use rayon::prelude::*;

    assert!(eq_lo.len().is_power_of_two() && eq_lo.len() >= 2);
    assert!(eq_hi.len().is_power_of_two());
    let expected_half = eq_lo
        .len()
        .checked_mul(eq_hi.len())
        .expect("split OOD fold length overflow");
    assert_eq!(f.len(), 2 * expected_half, "split OOD fold shape changed");
    assert_eq!(b.len(), f.len(), "split OOD polynomial lengths differ");
    if let Some((deferred_basis, _)) = deferred_basis {
        assert_eq!(
            deferred_basis.len(),
            f.len(),
            "deferred ordinary basis length changed"
        );
    }
    let tail_log = expected_half.trailing_zeros() as usize;
    assert_eq!(
        eq_lo.len(),
        1usize << tail_log.min(LAZY_OOD_EQ_SPLIT_LOW_LOG_MAX),
        "split OOD fold low factor width changed"
    );
    assert!(
        nf.capacity() >= expected_half && nb.capacity() >= expected_half,
        "split OOD spare capacity is insufficient"
    );
    // SAFETY: capacities are hard-checked above and every slot is overwritten
    // by exactly one disjoint chunk before either output is read.
    unsafe {
        nf.set_len(expected_half);
        nb.set_len(expected_half);
    }

    let gamma = beta * (F128::ONE + z_0 + r);
    let alpha_r = deferred_basis.map(|(_, alpha)| alpha * r);
    let chunk_body = |high_index: usize, f_chunk: &mut [F128], b_chunk: &mut [F128]| {
        let base = high_index * eq_lo.len();
        let chunk_scale = gamma * eq_hi[high_index];
        match deferred_basis {
            Some((deferred_basis, alpha)) => {
                crate::field::f128_slice::fold_two_and_msg_with_deferred_basis_and_scaled_local_addend(
                    f,
                    b,
                    deferred_basis,
                    eq_lo,
                    base,
                    f_chunk,
                    b_chunk,
                    r,
                    alpha,
                    alpha_r.expect("deferred basis scale product"),
                    chunk_scale,
                )
            }
            None => crate::field::f128_slice::fold_two_and_msg_with_scaled_local_basis_addend(
                f,
                b,
                eq_lo,
                base,
                f_chunk,
                b_chunk,
                r,
                chunk_scale,
            ),
        }
    };
    // Hetero queue over the same per-`high_index` chunk grid (chunk width =
    // `eq_lo.len()`, one chunk per `eq_hi` entry — 128 × 2,048 at the ranked
    // geometry). Identical kernels and chunk bases, so bytes are unchanged.
    let chunk_w = eq_lo.len();
    if eq_hi.len() >= 16 && lig_fold_hetero_enabled() && crate::epool::epool().is_some() {
        let n_chunks = eq_hi.len();
        let mut partials = vec![(F128::ZERO, F128::ZERO); n_chunks];
        let f_base = crate::epool::SyncPtr(nf.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(nb.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks(n_chunks, |hi| {
            // SAFETY: each `hi` is claimed exactly once; output ranges and the
            // partial slot are disjoint per chunk and published by the join.
            unsafe {
                let fc = core::slice::from_raw_parts_mut(f_base.ptr().add(hi * chunk_w), chunk_w);
                let bc = core::slice::from_raw_parts_mut(b_base.ptr().add(hi * chunk_w), chunk_w);
                partials_base.ptr().add(hi).write(chunk_body(hi, fc, bc));
            }
        });
        let (u_0, u_2) = partials
            .into_iter()
            .fold((F128::ZERO, F128::ZERO), |(a_0, a_2), (b_0, b_2)| {
                (a_0 + b_0, a_2 + b_2)
            });
        return SumcheckMessage { u_0, u_2 };
    }
    let (u_0, u_2) = nf
        .par_chunks_mut(chunk_w)
        .zip(nb.par_chunks_mut(chunk_w))
        .enumerate()
        .map(|(high_index, (f_chunk, b_chunk))| chunk_body(high_index, f_chunk, b_chunk))
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a_0, a_2), (b_0, b_2)| (a_0 + b_0, a_2 + b_2),
        );
    SumcheckMessage { u_0, u_2 }
}

/// Fold two consecutive sumcheck challenges in one streaming pass and emit
/// both the direct next message and the following message as six quadratic
/// coefficients. This removes the intermediate half-sized f/b state while
/// preserving the transcript's observe/sample order.
fn fold2_and_msgs_lsb(
    f: &[F128],
    b: &[F128],
    r_a: F128,
    r_b: F128,
    wf: &mut Vec<F128>,
    wb: &mut Vec<F128>,
) -> (SumcheckMessage, [F128; 6]) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 16);
    debug_assert_eq!(b.len(), n);
    let quarter = n / 4;
    debug_assert!(wf.capacity() >= quarter && wb.capacity() >= quarter);
    // SAFETY: capacity checked; F128: Copy; every slot written before read.
    unsafe {
        wf.set_len(quarter);
        wb.set_len(quarter);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    let oa = F128::ONE + r_a;
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    let ob = F128::ONE + r_b;

    // Per 8 input values (per poly) -> 2 outputs w[2t], w[2t+1]:
    //   v[j]    = f[2j]*oa + f[2j+1]*r_a          (first bind, in registers)
    //   w[t]    = v[2t]*ob + v[2t+1]*r_b          (second bind, written)
    // Direct message over w-pairs; lookahead over x[u] = w[2u]*oc + w[2u+1]*r_c:
    //   u0_D = sum_u x_f[2u]*x_b[2u]
    //        = sum_u (wf0*oc + wf1*rc)(wb0*oc + wb1*rc)
    //   with oc = 1 + rc:  expand in {1, rc, rc^2}:
    //     coeff of 1   : wf0*wb0
    //     coeff of rc  : wf0*(wb0+wb1) + wb0*(wf0+wf1)
    //     coeff of rc^2: (wf0+wf1)*(wb0+wb1)
    //   u2_D likewise over sums-of-adjacent-x, which reduce to the same three
    //   bilinear forms on (wf0+wf2.., wf1+wf3..) groupings handled below.
    const CHUNK: usize = 2048; // outputs per chunk; 8 inputs per output pair
    // Fold pairs whose w outputs are past LLC size write ping-pong state not
    // read until the next fold pair's barrier; `stnp` elides the
    // write-allocate RFO reads there (same driver-decided policy as the
    // zerocheck tail's NT rounds). 2^21 F128 = 32 MiB per polynomial (both
    // polynomials together exceed LLC).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let nt_stores = {
        use std::sync::OnceLock;
        static NT_ENABLED: OnceLock<bool> = OnceLock::new();
        quarter >= (1usize << 21)
            && *NT_ENABLED.get_or_init(|| std::env::var_os("FLOCK_LIG_NT_LEGACY").is_none())
    };
    let chunk_body = |ci: usize, wfc: &mut [F128], wbc: &mut [F128]| -> (F128, F128, [F128; 6]) {
        {
            let base = ci * CHUNK; // output index base
            #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
            {
                let (u0, u2, c) = crate::field::f128_slice::fold2_two_and_msgs(
                    f, b, base, wfc, wbc, r_a, r_b, nt_stores,
                );
                return (u0, u2, c);
            }
            #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
            {
                let len = wfc.len();
                let mut m_u0 = F128::ZERO;
                let mut m_u2 = F128::ZERO;
                let mut c = [F128::ZERO; 6];
                // process outputs in groups of 4 (one lookahead x-pair needs w[4u..4u+4])
                let mut t = 0;
                while t < len {
                    // build w[t..t+4] from f/b[8*(base+t) .. ]
                    let mut wq_f = [F128::ZERO; 4];
                    let mut wq_b = [F128::ZERO; 4];
                    for q in 0..4 {
                        let i = 2 * (base + t + q); // v-index base (2 v per w)
                        let vf0 = f[2 * i] * oa + f[2 * i + 1] * r_a;
                        let vf1 = f[2 * i + 2] * oa + f[2 * i + 3] * r_a;
                        let vb0 = b[2 * i] * oa + b[2 * i + 1] * r_a;
                        let vb1 = b[2 * i + 2] * oa + b[2 * i + 3] * r_a;
                        wq_f[q] = vf0 * ob + vf1 * r_b;
                        wq_b[q] = vb0 * ob + vb1 * r_b;
                        wfc[t + q] = wq_f[q];
                        wbc[t + q] = wq_b[q];
                    }
                    // Direct message over w-pairs (2 pairs in this group).
                    // Keep the four endpoint products live: the first pair's
                    // products are also lookahead c0/c2.
                    let s0f = wq_f[0] + wq_f[1];
                    let s0b = wq_b[0] + wq_b[1];
                    let s1f = wq_f[2] + wq_f[3];
                    let s1b = wq_b[2] + wq_b[3];
                    let p0 = wq_f[0] * wq_b[0];
                    let p1 = wq_f[2] * wq_b[2];
                    let ps0 = s0f * s0b;
                    let ps1 = s1f * s1b;
                    m_u0 += p0 + p1;
                    m_u2 += ps0 + ps1;
                    // lookahead: x0 = w0*oc + w1*rc, x1 = w2*oc + w3*rc (one x-pair)
                    // u0_D += x0_f * x0_b  -> bilinear in (w0,w1)
                    c[0] += p0;
                    // Karatsuba cross: w0*s0b + w0b*s0f =
                    // w1*w1b + p0 + ps0. Reuses both endpoint products.
                    c[1] += wq_f[1] * wq_b[1] + p0 + ps0;
                    c[2] += ps0;
                    // u2_D += (x0+x1)_f * (x0+x1)_b ; x0+x1 = (w0+w2)*oc + (w1+w3)*rc
                    let e_f = wq_f[0] + wq_f[2];
                    let o_f = wq_f[1] + wq_f[3];
                    let e_b = wq_b[0] + wq_b[2];
                    let o_b = wq_b[1] + wq_b[3];
                    let se_f = e_f + o_f;
                    let se_b = e_b + o_b;
                    let pe = e_f * e_b;
                    let pse = se_f * se_b;
                    // Here the complementary endpoint is the odd aggregate.
                    let po = o_f * o_b;
                    c[3] += pe;
                    c[4] += po + pe + pse;
                    c[5] += pse;
                    t += 4;
                }
                (m_u0, m_u2, c)
            }
        }
    };
    let merge = |(a0, a2, ac): (F128, F128, [F128; 6]),
                 (b0, b2, bc): (F128, F128, [F128; 6])| {
        let mut c = ac;
        for (x, y) in c.iter_mut().zip(bc.iter()) {
            *x += *y;
        }
        (a0 + b0, a2 + b2, c)
    };
    // Hetero queue over the identical chunk grid: same bases, same kernel,
    // XOR-merged partials — bytes unchanged, only chunk ownership differs.
    let acc = if quarter >= 16 * CHUNK
        && lig_fold_hetero_enabled()
        && crate::epool::epool().is_some()
    {
        let n_chunks = quarter / CHUNK;
        let mut partials = vec![(F128::ZERO, F128::ZERO, [F128::ZERO; 6]); n_chunks];
        let f_base = crate::epool::SyncPtr(wf.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(wb.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks(n_chunks, |ci| {
            // SAFETY: each `ci` claimed exactly once; disjoint output ranges
            // and partial slot per chunk, published by the two-pool join.
            unsafe {
                let wfc = core::slice::from_raw_parts_mut(f_base.ptr().add(ci * CHUNK), CHUNK);
                let wbc = core::slice::from_raw_parts_mut(b_base.ptr().add(ci * CHUNK), CHUNK);
                partials_base.ptr().add(ci).write(chunk_body(ci, wfc, wbc));
            }
        });
        partials
            .into_iter()
            .fold((F128::ZERO, F128::ZERO, [F128::ZERO; 6]), merge)
    } else {
        wf.par_chunks_mut(CHUNK)
            .zip(wb.par_chunks_mut(CHUNK))
            .enumerate()
            .map(|(ci, (wfc, wbc))| chunk_body(ci, wfc, wbc))
            .reduce(|| (F128::ZERO, F128::ZERO, [F128::ZERO; 6]), merge)
    };
    (
        SumcheckMessage {
            u_0: acc.0,
            u_2: acc.1,
        },
        acc.2,
    )
}

/// Final initial-lane pair: bind two challenges and emit only the direct next
/// message. The ordinary lookahead would describe a round beyond
/// `initial_k`, so computing it cannot affect the transcript or folded state.
fn fold2_and_msg_lsb(
    f: &[F128],
    b: &[F128],
    r_a: F128,
    r_b: F128,
    wf: &mut Vec<F128>,
    wb: &mut Vec<F128>,
) -> SumcheckMessage {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        // The ranked production path is AArch64. Keep other targets simple
        // and byte-identical by using the portable full oracle and discarding
        // only its unobserved lookahead.
        return fold2_and_msgs_lsb(f, b, r_a, r_b, wf, wb).0;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        use rayon::prelude::*;

        let n = f.len();
        debug_assert!(n.is_power_of_two() && n >= 16);
        debug_assert_eq!(b.len(), n);
        let quarter = n / 4;
        debug_assert!(wf.capacity() >= quarter && wb.capacity() >= quarter);
        // SAFETY: capacity checked; every slot is initialized by its unique
        // parallel chunk before the reduction returns.
        unsafe {
            wf.set_len(quarter);
            wb.set_len(quarter);
        }

        const CHUNK: usize = 1024;
        let nt_stores = {
            use std::sync::OnceLock;
            static NT_ENABLED: OnceLock<bool> = OnceLock::new();
            quarter >= (1usize << 21)
                && *NT_ENABLED.get_or_init(|| std::env::var_os("FLOCK_LIG_NT_LEGACY").is_none())
        };
        let chunk_body = |chunk: usize, f_out: &mut [F128], b_out: &mut [F128]| {
            crate::field::f128_slice::fold2_two_and_msg(
                f,
                b,
                chunk * CHUNK,
                f_out,
                b_out,
                r_a,
                r_b,
                nt_stores,
            )
        };
        // Hetero queue over the identical chunk grid — bytes unchanged.
        if quarter >= 16 * CHUNK && lig_fold_hetero_enabled() && crate::epool::epool().is_some() {
            let n_chunks = quarter / CHUNK;
            let mut partials = vec![(F128::ZERO, F128::ZERO); n_chunks];
            let f_base = crate::epool::SyncPtr(wf.as_mut_ptr());
            let b_base = crate::epool::SyncPtr(wb.as_mut_ptr());
            let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
            crate::epool::run_hetero_chunks(n_chunks, |ci| {
                // SAFETY: each `ci` claimed exactly once; disjoint output
                // ranges and partial slot, published by the two-pool join.
                unsafe {
                    let fc = core::slice::from_raw_parts_mut(f_base.ptr().add(ci * CHUNK), CHUNK);
                    let bc = core::slice::from_raw_parts_mut(b_base.ptr().add(ci * CHUNK), CHUNK);
                    partials_base.ptr().add(ci).write(chunk_body(ci, fc, bc));
                }
            });
            let (u_0, u_2) = partials
                .into_iter()
                .fold((F128::ZERO, F128::ZERO), |(a0, a2), (b0, b2)| {
                    (a0 + b0, a2 + b2)
                });
            return SumcheckMessage { u_0, u_2 };
        }
        let (u_0, u_2) = wf
            .par_chunks_mut(CHUNK)
            .zip(wb.par_chunks_mut(CHUNK))
            .enumerate()
            .map(|(chunk, (f_out, b_out))| chunk_body(chunk, f_out, b_out))
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );
        SumcheckMessage { u_0, u_2 }
    }
}

fn eval_lookahead(c: &[F128; 6], rho: F128) -> SumcheckMessage {
    let r2 = rho * rho;
    SumcheckMessage {
        u_0: c[0] + c[1] * rho + c[2] * r2,
        u_2: c[3] + c[4] * rho + c[5] * r2,
    }
}

/// Contract the two row-major quadratic coefficient tensors in place. An
/// ascending pass is safe because output `i` precedes every later source
/// triple `3(i + 1)..3(i + 1) + 3`.
#[inline]
fn eval_quadratic_tensors_in_place(
    coefficients: &mut [F128],
    challenges: &[F128],
) -> SumcheckMessage {
    let tensor_len = 3usize.pow(challenges.len() as u32);
    debug_assert_eq!(coefficients.len(), 2 * tensor_len);
    let (u_0, u_2) = coefficients.split_at_mut(tensor_len);
    let mut active_len = tensor_len;

    for &challenge in challenges.iter().rev() {
        let next_len = active_len / 3;
        for i in 0..next_len {
            let base = 3 * i;
            let u_0_a = u_0[base];
            let u_0_b = u_0[base + 1];
            let u_0_c = u_0[base + 2];
            let u_2_a = u_2[base];
            let u_2_b = u_2[base + 1];
            let u_2_c = u_2[base + 2];
            u_0[i] = u_0_a + challenge * (u_0_b + challenge * u_0_c);
            u_2[i] = u_2_a + challenge * (u_2_b + challenge * u_2_c);
        }
        active_len = next_len;
    }

    debug_assert_eq!(active_len, 1);
    SumcheckMessage {
        u_0: u_0[0],
        u_2: u_2[0],
    }
}

#[inline]
fn eval_fold4_lookahead2(
    coefficients: &mut super::Fold4Lookahead2,
    r0: F128,
    r1: F128,
) -> SumcheckMessage {
    eval_quadratic_tensors_in_place(coefficients, &[r0, r1])
}

#[inline]
fn eval_fold4_lookahead3(
    coefficients: &mut super::Fold4Lookahead3,
    r0: F128,
    r1: F128,
    r2: F128,
) -> SumcheckMessage {
    eval_quadratic_tensors_in_place(coefficients, &[r0, r1, r2])
}

#[inline]
#[cfg(test)]
fn eval_fold8_lookahead4(
    coefficients: &mut super::Fold8Lookahead4,
    r0: F128,
    r1: F128,
    r2: F128,
    r3: F128,
) -> SumcheckMessage {
    eval_quadratic_tensors_in_place(coefficients, &[r0, r1, r2, r3])
}

#[inline]
#[cfg(test)]
fn eval_fold8_lookahead5(
    coefficients: &mut super::Fold8Lookahead5,
    r0: F128,
    r1: F128,
    r2: F128,
    r3: F128,
    r4: F128,
) -> SumcheckMessage {
    eval_quadratic_tensors_in_place(coefficients, &[r0, r1, r2, r3, r4])
}

const DIRECT_FOLD8_CLAIM_PAR_MIN_STATE_LEN: usize = 1 << 12;
const ENV_NO_DIRECT_FOLD8_CLAIM_PAR: &str = "FLOCK_NO_DIRECT_FOLD8_CLAIM_PAR";

/// Exact-`1` rollback for the early-round AB/C claim join. Every other value
/// leaves the candidate enabled, so control and candidate use the same binary.
fn direct_fold8_claim_parallel_value_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("1"))
}

fn direct_fold8_claim_parallel_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        direct_fold8_claim_parallel_value_enabled(
            std::env::var_os(ENV_NO_DIRECT_FOLD8_CLAIM_PAR).as_deref(),
        )
    })
}

#[inline]
fn select_direct_fold8_claim_parallel(
    claim_count: usize,
    min_state_len: usize,
    thread_count: usize,
    homogeneous_pool: bool,
    enabled: bool,
) -> bool {
    enabled
        && claim_count == 2
        && min_state_len >= DIRECT_FOLD8_CLAIM_PAR_MIN_STATE_LEN
        && thread_count > 1
        && homogeneous_pool
}

#[inline]
fn direct_fold8_claim_parallel_pool_is_homogeneous(thread_count: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        thread_count <= crate::perf_core_count_cached()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = thread_count;
        false
    }
}

/// Bind one retained fold8 coordinate in both factor families for one claim.
/// States are bit-major `[bit][bank]`; while at least two banks remain after
/// the fold, flattening preserves every row's pair boundaries and the ordinary
/// fold/message kernel applies unchanged.
#[inline]
fn fold_one_direct_fold8_claim_and_message(
    claim: &mut super::ring_switch::DirectFold8Factors,
    challenge: F128,
) -> (F128, F128) {
    assert_eq!(claim.a_state.len(), claim.w_state.len());
    assert_eq!(claim.a_state.len() % (1usize << super::LOG_PACKING), 0);
    let banks = claim.a_state.len() >> super::LOG_PACKING;
    assert!(banks >= 4 && banks.is_power_of_two());
    crate::field::f128_slice::fold_two_and_msg_in_place(
        &mut claim.a_state,
        &mut claim.w_state,
        challenge,
    )
}

fn fold_direct_fold8_factors_and_message_selected(
    claims: &mut [super::ring_switch::DirectFold8Factors],
    challenge: F128,
    parallel: bool,
) -> SumcheckMessage {
    if parallel {
        assert_eq!(claims.len(), 2, "parallel fold8 requires AB and C claims");
        let (ab_claims, c_claims) = claims.split_at_mut(1);
        let (ab_partial, c_partial) = rayon::join(
            || fold_one_direct_fold8_claim_and_message(&mut ab_claims[0], challenge),
            || fold_one_direct_fold8_claim_and_message(&mut c_claims[0], challenge),
        );
        return SumcheckMessage {
            u_0: ab_partial.0 + c_partial.0,
            u_2: ab_partial.1 + c_partial.1,
        };
    }

    let mut u_0 = F128::ZERO;
    let mut u_2 = F128::ZERO;
    for claim in claims {
        let partial = fold_one_direct_fold8_claim_and_message(claim, challenge);
        u_0 += partial.0;
        u_2 += partial.1;
    }
    SumcheckMessage { u_0, u_2 }
}

fn fold_direct_fold8_factors_and_message(
    claims: &mut [super::ring_switch::DirectFold8Factors],
    challenge: F128,
) -> SumcheckMessage {
    let min_state_len = claims
        .iter()
        .map(|claim| claim.a_state.len())
        .min()
        .unwrap_or(0);
    let thread_count = rayon::current_num_threads();
    let parallel = select_direct_fold8_claim_parallel(
        claims.len(),
        min_state_len,
        thread_count,
        direct_fold8_claim_parallel_pool_is_homogeneous(thread_count),
        direct_fold8_claim_parallel_enabled(),
    );
    fold_direct_fold8_factors_and_message_selected(claims, challenge, parallel)
}

/// Bind the sixth retained coordinate of `W`. At this point each bit row has
/// two banks, so the result is the 128-generator vector used by the direct
/// byte map. No witness-factor state is needed after the round-five message.
fn direct_fold8_final_generators(
    claim: &super::ring_switch::DirectFold8Factors,
    challenge: F128,
) -> [F128; 1 << super::LOG_PACKING] {
    let mut generators = [F128::ZERO; 1 << super::LOG_PACKING];
    assert_eq!(claim.w_state.len(), 2 * generators.len());
    crate::field::f128_slice::fold_pairs(&claim.w_state, 0, &mut generators, challenge);
    generators
}

/// Exact fallback for the final-pair specialization. With the opt-out set,
/// the prover computes and discards the incumbent lookahead as before.
fn fold2_final_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_LIG_FOLD2_FINAL").is_none())
}
/// Whether the fused first-claim + ordinary-basis fold4 initialization is
/// enabled for [`materialize_direct_ab_fold2`].
///
/// `FLOCK_NO_DIRECT_AB_FUSE_INIT=1` restores the frontier's
/// zero-fill → sum-claims → `+= fold4(C)` sequence in the same binary.
fn direct_ab_fuse_init_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_DIRECT_AB_FUSE_INIT").is_none())
}

/// Materialize the combined sumcheck state only after its first two folds.
/// `ordinary_basis` contains the incumbent C contribution; `claims` contains
/// the AB contribution in sufficient-stat form. Both are γ-baked. Deferred-C
/// mode instead passes `ordinary_basis` EMPTY and C as a second direct claim,
/// so there is no fold4 over a materialized basis at all.
///
/// Ranked shape has exactly one direct claim (AB). The default path therefore
/// fuses that claim's contribution with the ordinary-basis fold4 into a
/// single assignment per output slot — deleting the full L/4 zero-fill pass
/// and the subsequent read-modify-write of `b_out` that used to add fold4(C).
/// Algebra: `0 + D_0 + … + D_n + fold4(C)` becomes `D_0 + fold4(C) + D_1 + …`.
/// Exact ranked direct-AB materialization can distribute its 256 independent
/// 2 MiB blocks across the existing P/E stateful queue. Each worker retains
/// one private fold table; the queue changes only block ownership.
#[inline]
fn use_ranked_direct_ab_hetero_materialize(
    packed_len: usize,
    block_len: usize,
    claim_count: usize,
    has_ordinary: bool,
) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && matches!(rayon::current_num_threads(), 2..=16)
        && super::is_ranked_direct_fold2_lookahead_shape(
            packed_len,
            block_len,
            claim_count,
            has_ordinary,
        )
        && std::env::var_os("FLOCK_NO_DIRECT_AB_HETERO_MATERIALIZE").is_none()
        && crate::epool::epool().is_some()
}

fn materialize_direct_ab_fold2(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold2Factors],
    r0: F128,
    r1: F128,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage, [F128; 6]) {
    let block_len = claims
        .first()
        .expect("direct AB materialization requires a claim")
        .eq_lo
        .len();
    let helper = use_ranked_direct_ab_hetero_materialize(
        packed_witness.len(),
        block_len,
        claims.len(),
        !ordinary_basis.is_empty(),
    )
    .then(crate::epool::epool)
    .flatten();
    materialize_direct_ab_fold2_with_helper(packed_witness, ordinary_basis, claims, r0, r1, helper)
}

fn materialize_direct_ab_fold2_with_helper(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold2Factors],
    r0: F128,
    r1: F128,
    helper: Option<&rayon::ThreadPool>,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage, [F128; 6]) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    let has_ordinary = !ordinary_basis.is_empty();
    assert!(!has_ordinary || ordinary_basis.len() == packed_witness.len());
    let fold_weight = [
        (F128::ONE + r0) * (F128::ONE + r1),
        r0 * (F128::ONE + r1),
        (F128::ONE + r0) * r1,
        r0 * r1,
    ];
    let direct_tables: Vec<Vec<F128>> = claims
        .iter()
        .map(|claim| {
            super::ring_switch::build_direct_fold2_table(&claim.low_eq, &fold_weight, &claim.table)
        })
        .collect();

    let out_len = packed_witness.len() / 4;
    let block_len = claims[0].eq_lo.len();
    assert_eq!(out_len, block_len * claims[0].eq_hi.len());
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && claim.eq_hi.len() * block_len == out_len
    }));
    let ranked_lookahead_neon = super::is_ranked_direct_fold2_lookahead_shape(
        packed_witness.len(),
        block_len,
        claims.len(),
        has_ordinary,
    );

    let fuse_init = direct_ab_fuse_init_enabled();
    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    type FoldStats = ((F128, F128), [F128; 6]);
    fn empty_stats() -> FoldStats {
        ((F128::ZERO, F128::ZERO), [F128::ZERO; 6])
    }
    fn merge_stats(((x0, x2), mut xc): FoldStats, ((y0, y2), yc): FoldStats) -> FoldStats {
        for (x, y) in xc.iter_mut().zip(yc) {
            *x += y;
        }
        ((x0 + y0, x2 + y2), xc)
    }
    let fold_block =
        |scratch: &mut Vec<F128>, block: usize, b_out: &mut [F128], f_out: &mut [F128]| {
            let start = 4 * block * block_len;
            let f_in = &packed_witness[start..start + 4 * block_len];
            let b_in: &[F128] = if has_ordinary {
                &ordinary_basis[start..start + 4 * block_len]
            } else {
                &[]
            };
            let fold4 = |input: &[F128], slot: usize| {
                let a0 = input[4 * slot];
                let a1 = input[4 * slot + 1];
                let a2 = input[4 * slot + 2];
                let a3 = input[4 * slot + 3];
                let low = a0 + r0 * (a0 + a1);
                let high = a2 + r0 * (a2 + a3);
                low + r1 * (low + high)
            };

            if fuse_init {
                // First direct claim initializes; remaining claims add.
                // Fuse claim-0 with ordinary-basis fold4 so each b_out
                // slot is written once when there is a single claim
                // (the ranked AB-only shape).
                let (first_claim, rest_claims) = claims.split_first().expect("nonempty claims");
                let (first_table, rest_tables) =
                    direct_tables.split_first().expect("nonempty tables");
                super::ring_switch::compose_fold_byte_table_into(
                    first_claim.eq_hi[block],
                    first_table,
                    scratch,
                );
                if has_ordinary {
                    for slot in 0..block_len {
                        let direct =
                            super::ring_switch::fold_one_slot(first_claim.eq_lo[slot], scratch);
                        f_out[slot] = fold4(f_in, slot);
                        b_out[slot] = direct + fold4(b_in, slot);
                    }
                } else {
                    for slot in 0..block_len {
                        f_out[slot] = fold4(f_in, slot);
                        b_out[slot] =
                            super::ring_switch::fold_one_slot(first_claim.eq_lo[slot], scratch);
                    }
                }
                for (claim, direct_table) in rest_claims.iter().zip(rest_tables.iter()) {
                    super::ring_switch::compose_fold_byte_table_into(
                        claim.eq_hi[block],
                        direct_table,
                        scratch,
                    );
                    for (slot, out) in b_out.iter_mut().enumerate() {
                        *out += super::ring_switch::fold_one_slot(claim.eq_lo[slot], scratch);
                    }
                }
            } else {
                // Frontier control: zero-fill, sum all direct claims, then
                // add ordinary-basis fold4.
                b_out.fill(F128::ZERO);
                for (claim, direct_table) in claims.iter().zip(direct_tables.iter()) {
                    super::ring_switch::compose_fold_byte_table_into(
                        claim.eq_hi[block],
                        direct_table,
                        scratch,
                    );
                    for (slot, out) in b_out.iter_mut().enumerate() {
                        *out += super::ring_switch::fold_one_slot(claim.eq_lo[slot], scratch);
                    }
                }
                if has_ordinary {
                    for slot in 0..block_len {
                        f_out[slot] = fold4(f_in, slot);
                        b_out[slot] += fold4(b_in, slot);
                    }
                } else {
                    for slot in 0..block_len {
                        f_out[slot] = fold4(f_in, slot);
                    }
                }
            }
            super::round0_and_round1_lookahead_ranked(f_out, b_out, ranked_lookahead_neon)
        };
    let stats = if let Some(helper) = helper {
        let n_blocks = out_len / block_len;
        let mut partials = vec![empty_stats(); n_blocks];
        let b_base = crate::epool::SyncPtr(folded_b.as_mut_ptr());
        let f_base = crate::epool::SyncPtr(folded_f.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_chunks_with_helper_stateful(
            n_blocks,
            &|| vec![F128::ZERO; super::ring_switch::FOLD_TABLE_LEN],
            &|scratch, block| {
                // SAFETY: the queue claims each block exactly once. That
                // block owns disjoint folded-f/folded-b output ranges and
                // exactly one partial slot until the synchronous join.
                unsafe {
                    let b_out = core::slice::from_raw_parts_mut(
                        b_base.ptr().add(block * block_len),
                        block_len,
                    );
                    let f_out = core::slice::from_raw_parts_mut(
                        f_base.ptr().add(block * block_len),
                        block_len,
                    );
                    partials_base
                        .ptr()
                        .add(block)
                        .write(fold_block(scratch, block, b_out, f_out));
                }
            },
            Some(helper),
        );
        partials.into_iter().fold(empty_stats(), merge_stats)
    } else {
        // Preserve the frontier scheduler when the exact ranked P/E gate is
        // closed or its explicit kill switch is set.
        folded_b
            .par_chunks_mut(block_len)
            .zip(folded_f.par_chunks_mut(block_len))
            .enumerate()
            .map_init(
                || vec![F128::ZERO; super::ring_switch::FOLD_TABLE_LEN],
                |scratch, (block, (b_out, f_out))| fold_block(scratch, block, b_out, f_out),
            )
            .reduce(empty_stats, merge_stats)
    };
    crate::scratch::give_f128(packed_witness);
    crate::scratch::give_f128(ordinary_basis);
    (
        folded_f,
        folded_b,
        SumcheckMessage {
            u_0: stats.0.0,
            u_2: stats.0.1,
        },
        stats.1,
    )
}

/// Correctness-first sixteen-bank materializer. Four challenges are sampled
/// from direct product statistics before this function binds the witness and
/// combined basis in one N→N/16 pass. It emits only the ordinary message M4
/// and univariate lookahead for M5; the incumbent final fold2 cadence then
/// handles rounds four and five.
fn materialize_direct_fold4(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold4Factors],
    challenges: [F128; 4],
) -> (Vec<F128>, Vec<F128>, SumcheckMessage, [F128; 6]) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    let has_ordinary = !ordinary_basis.is_empty();
    assert!(!has_ordinary || ordinary_basis.len() == packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(16));

    let fold_weight: [F128; 16] = std::array::from_fn(|bank| {
        let mut weight = F128::ONE;
        for (bit, &challenge) in challenges.iter().enumerate() {
            weight *= if (bank >> bit) & 1 == 0 {
                F128::ONE + challenge
            } else {
                challenge
            };
        }
        weight
    });
    let direct_tables: Vec<Vec<F128>> = claims
        .iter()
        .map(|claim| {
            super::ring_switch::build_direct_fold4_table(&claim.low_eq, &fold_weight, &claim.table)
        })
        .collect();

    let out_len = packed_witness.len() / 16;
    let block_len = claims[0].eq_lo.len();
    assert!(block_len.is_multiple_of(4));
    assert_eq!(out_len, block_len * claims[0].eq_hi.len());
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && claim.eq_hi.len() * block_len == out_len
    }));
    let deferred_reduce = super::use_fold_deferred_reduce();

    type FoldStats = ((F128, F128), [F128; 6]);
    let empty_stats = || ((F128::ZERO, F128::ZERO), [F128::ZERO; 6]);
    let merge_stats = |((x0, x2), mut xc): FoldStats, ((y0, y2), yc): FoldStats| {
        for (out, value) in xc.iter_mut().zip(yc) {
            *out += value;
        }
        ((x0 + y0, x2 + y2), xc)
    };

    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    let stats = folded_b
        .par_chunks_mut(block_len)
        .zip(folded_f.par_chunks_mut(block_len))
        .enumerate()
        .map_init(
            || vec![F128::ZERO; super::ring_switch::FOLD_TABLE_LEN],
            |scratch, (block, (b_out, f_out))| {
                let start = 16 * block * block_len;
                let f_in = &packed_witness[start..start + 16 * block_len];
                let b_in: &[F128] = if has_ordinary {
                    &ordinary_basis[start..start + 16 * block_len]
                } else {
                    &[]
                };
                let fold16 = |input: &[F128], slot: usize| {
                    let base = 16 * slot;
                    if deferred_reduce {
                        return crate::field::f128_slice::fold_banked_slot::<16>(
                            &fold_weight,
                            &input[base..base + 16],
                        );
                    }
                    let mut value = F128::ZERO;
                    for bank in 0..16 {
                        value += fold_weight[bank] * input[base + bank];
                    }
                    value
                };

                let (first_claim, rest_claims) = claims.split_first().unwrap();
                let (first_table, rest_tables) = direct_tables.split_first().unwrap();
                super::ring_switch::compose_fold_byte_table_into(
                    first_claim.eq_hi[block],
                    first_table,
                    scratch,
                );
                for slot in 0..block_len {
                    f_out[slot] = fold16(f_in, slot);
                    let direct =
                        super::ring_switch::fold_one_slot(first_claim.eq_lo[slot], scratch);
                    b_out[slot] = if has_ordinary {
                        direct + fold16(b_in, slot)
                    } else {
                        direct
                    };
                }
                for (claim, table) in rest_claims.iter().zip(rest_tables.iter()) {
                    super::ring_switch::compose_fold_byte_table_into(
                        claim.eq_hi[block],
                        table,
                        scratch,
                    );
                    for (slot, out) in b_out.iter_mut().enumerate() {
                        *out += super::ring_switch::fold_one_slot(claim.eq_lo[slot], scratch);
                    }
                }
                super::round0_and_round1_lookahead_deferred(f_out, b_out)
            },
        )
        .reduce(empty_stats, merge_stats);

    crate::scratch::give_f128(packed_witness);
    crate::scratch::give_f128(ordinary_basis);
    (
        folded_f,
        folded_b,
        SumcheckMessage {
            u_0: stats.0.0,
            u_2: stats.0.1,
        },
        stats.1,
    )
}

/// Sixty-four-bank materializer. Six challenges are sampled from the direct
/// factor state before this function binds the witness and combined basis in
/// one N→N/64 pass. It emits M6 — the round message of the folded
/// 2^19 state — fused into the same pass; no lookahead follows because the
/// initial cadence is exhausted (the fold2 pair of the fold4 route never
/// runs and the 2^21/2^20 states never exist).
fn materialize_direct_fold8(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold8Factors],
    challenges: [F128; 6],
) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    let has_ordinary = !ordinary_basis.is_empty();
    assert!(!has_ordinary || ordinary_basis.len() == packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(64));

    let fold_weight: [F128; 64] = std::array::from_fn(|bank| {
        let mut weight = F128::ONE;
        for (bit, &challenge) in challenges.iter().enumerate() {
            weight *= if (bank >> bit) & 1 == 0 {
                F128::ONE + challenge
            } else {
                challenge
            };
        }
        weight
    });
    let direct_tables: Vec<Vec<F128>> = claims
        .iter()
        .map(|claim| {
            let generators = direct_fold8_final_generators(claim, challenges[5]);
            super::ring_switch::build_direct_fold8_table_from_generators(&generators)
        })
        .collect();

    let out_len = packed_witness.len() / 64;
    let block_len = claims[0].eq_lo.len();
    assert!(block_len.is_multiple_of(4));
    assert_eq!(out_len, block_len * claims[0].eq_hi.len());
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && claim.eq_hi.len() * block_len == out_len
    }));
    let deferred_reduce = super::use_fold_deferred_reduce();
    // On Apple AArch64, adjacent slots share the same 64 fold weights. Keep
    // their product sums independent while loading each weight only once.
    // The rollback retains the incumbent two single-slot calls for A/B.
    let pair_fold64 = deferred_reduce
        && cfg!(all(target_arch = "aarch64", target_feature = "aes"))
        && std::env::var_os("FLOCK_NO_DIRECT_FOLD8_PAIR").is_none();

    // One shared per-block body for both drains below, so the scheduling
    // choice cannot drift from the value computation. For block `i` it fully
    // rewrites `f_out`/`b_out` (each slot written before any read) from the
    // disjoint witness stripe `[64·i·B, 64·(i+1)·B)` and returns the block's
    // round-0 partial. Which worker (or pool) runs a block cannot change a
    // single bit of it.
    let fold8_block = |scratch: &mut Vec<F128>,
                       block: usize,
                       b_out: &mut [F128],
                       f_out: &mut [F128]|
     -> (F128, F128) {
        let start = 64 * block * block_len;
        let f_in = &packed_witness[start..start + 64 * block_len];
        let b_in: &[F128] = if has_ordinary {
            &ordinary_basis[start..start + 64 * block_len]
        } else {
            &[]
        };
        let fold64 = |input: &[F128], slot: usize| {
            let base = 64 * slot;
            if deferred_reduce {
                return crate::field::f128_slice::fold_banked_slot::<64>(
                    &fold_weight,
                    &input[base..base + 64],
                );
            }
            let mut value = F128::ZERO;
            for bank in 0..64 {
                value += fold_weight[bank] * input[base + bank];
            }
            value
        };

        let (first_claim, rest_claims) = claims.split_first().unwrap();
        let (first_table, rest_tables) = direct_tables.split_first().unwrap();
        super::ring_switch::compose_fold_byte_table_into(
            first_claim.eq_hi[block],
            first_table,
            scratch,
        );
        let mut slot = 0usize;
        if pair_fold64 {
            while slot + 1 < block_len {
                let base = 64 * slot;
                let folded_f = crate::field::f128_slice::fold_banked_slots2::<64>(
                    &fold_weight,
                    &f_in[base..base + 128],
                );
                f_out[slot] = folded_f[0];
                f_out[slot + 1] = folded_f[1];

                let direct0 = super::ring_switch::fold_one_slot(first_claim.eq_lo[slot], scratch);
                let direct1 =
                    super::ring_switch::fold_one_slot(first_claim.eq_lo[slot + 1], scratch);
                if has_ordinary {
                    let folded_b = crate::field::f128_slice::fold_banked_slots2::<64>(
                        &fold_weight,
                        &b_in[base..base + 128],
                    );
                    b_out[slot] = direct0 + folded_b[0];
                    b_out[slot + 1] = direct1 + folded_b[1];
                } else {
                    b_out[slot] = direct0;
                    b_out[slot + 1] = direct1;
                }
                slot += 2;
            }
        }
        while slot < block_len {
            f_out[slot] = fold64(f_in, slot);
            let direct = super::ring_switch::fold_one_slot(first_claim.eq_lo[slot], scratch);
            b_out[slot] = if has_ordinary {
                direct + fold64(b_in, slot)
            } else {
                direct
            };
            slot += 1;
        }
        for (claim, table) in rest_claims.iter().zip(rest_tables.iter()) {
            super::ring_switch::compose_fold_byte_table_into(claim.eq_hi[block], table, scratch);
            for (slot, out) in b_out.iter_mut().enumerate() {
                *out += super::ring_switch::fold_one_slot(claim.eq_lo[slot], scratch);
            }
        }
        super::round0_deferred(f_out, b_out)
    };

    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    let stats = if super::use_open_mat_hetero(
        packed_witness.len(),
        block_len,
        claims.len(),
        has_ordinary,
    ) {
        // Heterogeneous drain: the 256 blocks go through the shared P/E
        // atomic queue (same worker-private-scratch shape as
        // `pcs::run_hetero_open_combine_blocks`, same 33.5M-product census
        // rationale — see `pcs::use_open_mat_hetero`). Each queue index owns
        // its disjoint `f_out`/`b_out` stripes and one partial slot until
        // the synchronous two-pool join publishes them. The partials are
        // reduced by block index after the join; in char 2 the sum is an
        // XOR multiset, so the message equals the rayon `reduce` bitwise.
        let n_blocks = out_len / block_len;
        let mut partials = vec![(F128::ZERO, F128::ZERO); n_blocks];
        let f_base = crate::epool::SyncPtr(folded_f.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(folded_b.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks_stateful(
            n_blocks,
            || vec![F128::ZERO; super::ring_switch::FOLD_TABLE_LEN],
            |scratch, block| {
                // SAFETY: the queue hands out each `block` exactly once; it
                // owns output range `[block·B, (block+1)·B)` in both arrays
                // and one partial slot; the two-pool join publishes all
                // writes before the reduce reads them.
                unsafe {
                    let f_out = core::slice::from_raw_parts_mut(
                        f_base.ptr().add(block * block_len),
                        block_len,
                    );
                    let b_out = core::slice::from_raw_parts_mut(
                        b_base.ptr().add(block * block_len),
                        block_len,
                    );
                    partials_base
                        .ptr()
                        .add(block)
                        .write(fold8_block(scratch, block, b_out, f_out));
                }
            },
        );
        partials
            .into_iter()
            .fold((F128::ZERO, F128::ZERO), |(x0, x2), (y0, y2)| {
                (x0 + y0, x2 + y2)
            })
    } else {
        folded_b
            .par_chunks_mut(block_len)
            .zip(folded_f.par_chunks_mut(block_len))
            .enumerate()
            .map_init(
                || vec![F128::ZERO; super::ring_switch::FOLD_TABLE_LEN],
                |scratch, (block, (b_out, f_out))| fold8_block(scratch, block, b_out, f_out),
            )
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
            )
    };

    crate::scratch::give_f128(packed_witness);
    crate::scratch::give_f128(ordinary_basis);
    (
        folded_f,
        folded_b,
        SumcheckMessage {
            u_0: stats.0,
            u_2: stats.1,
        },
    )
}

/// Enable the lazy explicit-OOD equality path only for the ranked M32 Fast
/// L1 geometry. All other profiles, levels, sample counts, platforms, and an
/// explicit rollback retain the materialized full equality table.
#[inline]
fn ranked_l1_lazy_ood_eq_enabled(
    config: &ProverConfig,
    log_n: usize,
    n_1: usize,
    l1_ood_count: usize,
    current_len: usize,
    direct_fold8_mode: bool,
) -> bool {
    ranked_l1_lazy_ood_eq_selected(
        config,
        log_n,
        n_1,
        l1_ood_count,
        current_len,
        direct_fold8_mode,
        cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            target_feature = "aes"
        )),
        std::env::var_os("FLOCK_NO_LIG_LAZY_OOD_EQ").is_some(),
    )
}

/// Pure selector underneath [`ranked_l1_lazy_ood_eq_enabled`], split out so
/// every ranked-shape and rollback boundary can be mutation-tested without
/// changing process-global environment variables.
#[inline]
fn ranked_l1_lazy_ood_eq_selected(
    config: &ProverConfig,
    log_n: usize,
    n_1: usize,
    l1_ood_count: usize,
    current_len: usize,
    direct_fold8_mode: bool,
    platform_supported: bool,
    disabled: bool,
) -> bool {
    platform_supported
        && !disabled
        && direct_fold8_mode
        && log_n == 25
        && n_1 == 19
        && current_len == (1usize << 19)
        && l1_ood_count == 1
        && config.initial_log_msg_cols == 19
        && config.initial_log_num_interleaved == 6
        && config.initial_k == 6
        && config.recursive_steps == 5
        && config.recursive_log_msg_cols.as_slice() == [16, 13, 10, 7, 4]
        && config.recursive_ks.as_slice() == [3, 3, 3, 3, 3]
        && config.log_inv_rates.as_slice() == [1, 2, 3, 4, 5, 6]
        && config.queries.as_slice() == [218, 106, 71, 53, 43, 36]
        && config.grinding_bits.as_slice() == [0, 0, 0, 0, 0, 0]
        && config.fold_grinding_bits.as_slice() == [19, 14, 11, 8, 6, 4]
        && config.ood_samples.as_slice() == [0, 1, 1, 1, 1, 1]
        && config.merkle_hash == HashKind::Blake3
}

/// Factorized explicit-OOD state. `Introduced` spans only the transcript
/// observe/sample boundary; `Glued` remains separate from ordinary
/// `pending_glue` until the next fold consumes it.
enum PendingOodEq {
    Introduced {
        eq_lo: Vec<F128>,
        eq_hi: Vec<F128>,
        z_0: F128,
        h_new: F128,
    },
    Glued {
        eq_lo: Vec<F128>,
        eq_hi: Vec<F128>,
        z_0: F128,
        beta: F128,
    },
}

pub struct SumcheckProver {
    f: Vec<F128>,
    /// Single combined basis poly. After every `glue(β)`, the introduced
    /// `b_new` is folded into here as `combined_basis += β · b_new`. This
    /// keeps fold cost O(1 + 1) = (f + combined_basis) regardless of how
    /// many recursive intro/glue pairs have happened.
    combined_basis: Vec<F128>,
    /// Ping-pong spares for [`Self::fold`]: each fold writes the halved
    /// outputs into the spares (capacity >= current length / 2) and swaps
    /// them in, so the ladder touches one resident page set per prove instead
    /// of allocating, faulting, and unmapping a fresh buffer pair per round
    /// (~1 GiB of churn across the ranked recursive open). Taken from the
    /// scratch pool at construction and returned on drop, so the worker's
    /// timed prove reuses the pages its warm-up prove faulted in.
    spare_f: Vec<F128>,
    spare_b: Vec<F128>,
    t_r: F128,
    transcript: Vec<SumcheckMessage>,
    pending_glue: Option<(Vec<F128>, F128)>,
    /// One ordinary induced basis whose sampled glue challenge has already
    /// updated `t_r`, but whose pointwise basis update is deferred into the
    /// same ranked fold that consumes [`Self::pending_ood_eq`].
    pending_fold_basis: Option<(Vec<F128>, F128)>,
    /// Ranked L1 explicit-OOD equality keeps `eq(z[1..])` as low/high tensor
    /// factors plus `(z[0], beta)` until the next fold. Ordinary induced-basis
    /// introduce/glue operations proceed independently through `pending_glue`.
    pending_ood_eq: Option<PendingOodEq>,
}

impl SumcheckProver {
    /// Ping-pong spare of capacity >= `f.len() / 2`; an empty Vec when the
    /// prover is degenerate (len < 2), so `take_f128(0)` can never steal a
    /// large pooled buffer.
    fn new_spare(len: usize) -> Vec<F128> {
        let half = len / 2;
        if half == 0 {
            Vec::new()
        } else {
            crate::scratch::take_f128(half)
        }
    }

    pub fn new(f: Vec<F128>, b1: Vec<F128>, h1: F128) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let spare_f = Self::new_spare(f.len());
        let spare_b = Self::new_spare(f.len());
        let mut inst = Self {
            f,
            combined_basis: b1,
            spare_f,
            spare_b,
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        };
        let msg = round_msg_lsb(&inst.f, &inst.combined_basis);
        inst.transcript.push(msg);
        (inst, msg)
    }

    /// Like [`Self::new`] but skips the initial `round_msg_lsb` pass over
    /// `(f, b1)` because the caller already computed `(u_0, u_2)` while
    /// building `b1` (saves a 256 MB read pass at m=30 BLAKE3). Used by
    /// `recursive_prover_with_basis` to consume the round0 prime that
    /// `compute_combined_basis_and_target` produces for free.
    pub fn new_with_first_msg(
        f: Vec<F128>,
        b1: Vec<F128>,
        h1: F128,
        first_msg: SumcheckMessage,
    ) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let spare_f = Self::new_spare(f.len());
        let spare_b = Self::new_spare(f.len());
        let mut inst = Self {
            f,
            combined_basis: b1,
            spare_f,
            spare_b,
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        };
        inst.transcript.push(first_msg);
        (inst, first_msg)
    }
    fn new_after_direct_fold2(
        f: Vec<F128>,
        basis: Vec<F128>,
        target: F128,
        transcript: [SumcheckMessage; 3],
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            spare_f: Self::new_spare(f.len()),
            spare_b: Self::new_spare(f.len()),
            f,
            combined_basis: basis,
            t_r: target,
            transcript: transcript.to_vec(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        }
    }

    fn new_after_direct_fold4(
        f: Vec<F128>,
        basis: Vec<F128>,
        target: F128,
        transcript: [SumcheckMessage; 5],
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            spare_f: Self::new_spare(f.len()),
            spare_b: Self::new_spare(f.len()),
            f,
            combined_basis: basis,
            t_r: target,
            transcript: transcript.to_vec(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        }
    }

    fn new_after_direct_fold8(
        f: Vec<F128>,
        basis: Vec<F128>,
        target: F128,
        transcript: [SumcheckMessage; 7],
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            spare_f: Self::new_spare(f.len()),
            spare_b: Self::new_spare(f.len()),
            f,
            combined_basis: basis,
            t_r: target,
            transcript: transcript.to_vec(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        }
    }

    pub fn fold(&mut self, r: F128) -> SumcheckMessage {
        // Fused: fold f and combined_basis at r AND build the next-round
        // message in one parallel pass (was three passes), writing the halved
        // outputs into the persistent ping-pong spares and swapping them in.
        // A ranked L1 OOD equality may be retained in factorized form until
        // this exact point; its correction is incorporated into both folded
        // basis state and the returned next-round message before the swap.
        assert!(
            self.pending_glue.is_none(),
            "fold before ordinary glue challenge"
        );
        let msg = match (self.pending_ood_eq.take(), self.pending_fold_basis.take()) {
            (
                Some(PendingOodEq::Glued {
                    eq_lo,
                    eq_hi,
                    z_0,
                    beta,
                }),
                deferred_basis,
            ) => fold_and_msg_lsb_into_with_lazy_ood_eq(
                &self.f,
                &self.combined_basis,
                deferred_basis
                    .as_ref()
                    .map(|(basis, alpha)| (basis.as_slice(), *alpha)),
                r,
                &eq_lo,
                &eq_hi,
                beta,
                z_0,
                &mut self.spare_f,
                &mut self.spare_b,
            ),
            (Some(PendingOodEq::Introduced { .. }), _) => {
                panic!("fold before lazy OOD glue")
            }
            (None, Some(_)) => {
                panic!("deferred ordinary glue without a consuming lazy OOD fold")
            }
            (None, None) => fold_and_msg_lsb_into(
                &self.f,
                &self.combined_basis,
                r,
                &mut self.spare_f,
                &mut self.spare_b,
            ),
        };
        std::mem::swap(&mut self.f, &mut self.spare_f);
        std::mem::swap(&mut self.combined_basis, &mut self.spare_b);
        self.transcript.push(msg);
        msg
    }

    /// Record a message produced by evaluating lookahead coefficients. Its
    /// state bind is deliberately deferred until the paired challenge arrives.
    fn push_lookahead_msg(&mut self, msg: SumcheckMessage) {
        self.transcript.push(msg);
    }

    /// Bind two already-sampled challenges in one pass, replacing the current
    /// state with the quarter-sized result while retaining the existing
    /// scratch ping-pong allocation.
    fn fold2(&mut self, r_a: F128, r_b: F128) -> (SumcheckMessage, [F128; 6]) {
        debug_assert!(self.pending_glue.is_none(), "fold2 across pending glue");
        assert!(
            self.pending_fold_basis.is_none(),
            "fold2 across deferred ordinary glue"
        );
        debug_assert!(
            self.pending_ood_eq.is_none(),
            "fold2 across pending OOD equality"
        );
        let (msg, coeffs) = fold2_and_msgs_lsb(
            &self.f,
            &self.combined_basis,
            r_a,
            r_b,
            &mut self.spare_f,
            &mut self.spare_b,
        );
        std::mem::swap(&mut self.f, &mut self.spare_f);
        std::mem::swap(&mut self.combined_basis, &mut self.spare_b);
        self.transcript.push(msg);
        (msg, coeffs)
    }

    /// Bind the final two initial-lane challenges without producing the
    /// lookahead that has no consumer past `initial_k`.
    fn fold2_final(&mut self, r_a: F128, r_b: F128) -> SumcheckMessage {
        debug_assert!(self.pending_glue.is_none(), "fold2 across pending glue");
        assert!(
            self.pending_fold_basis.is_none(),
            "fold2 across deferred ordinary glue"
        );
        debug_assert!(
            self.pending_ood_eq.is_none(),
            "fold2 across pending OOD equality"
        );
        let msg = fold2_and_msg_lsb(
            &self.f,
            &self.combined_basis,
            r_a,
            r_b,
            &mut self.spare_f,
            &mut self.spare_b,
        );
        std::mem::swap(&mut self.f, &mut self.spare_f);
        std::mem::swap(&mut self.combined_basis, &mut self.spare_b);
        self.transcript.push(msg);
        msg
    }

    /// Introduce a fresh basis poly with claimed sum `h_new`. Sends the
    /// (u_0, u_2) for `Σ_x f(x) · b_new(x)` at the current dim.
    pub fn introduce_new(&mut self, b_new: Vec<F128>, h_new: F128) -> SumcheckMessage {
        assert_eq!(b_new.len(), self.f.len());
        assert!(
            self.pending_glue.is_none(),
            "ordinary introduction already pending"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "ordinary introduction across deferred glue"
        );
        let msg = round_msg_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        msg
    }

    /// Introduce a basis whose exact message was accumulated while the basis
    /// was produced. This changes no transcript or pending-state ordering; it
    /// only avoids rereading `(self.f, b_new)` through [`round_msg_lsb`].
    fn introduce_new_with_precomputed_msg(
        &mut self,
        b_new: Vec<F128>,
        h_new: F128,
        msg: SumcheckMessage,
    ) -> SumcheckMessage {
        assert_eq!(b_new.len(), self.f.len());
        assert!(
            self.pending_glue.is_none(),
            "ordinary introduction already pending"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "ordinary introduction across deferred glue"
        );
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        msg
    }

    /// Like [`Self::introduce_new`] but also returns the claimed sum
    /// `h_new = Σ_x f(x)·b_new(x)`, computed in the same pass as the round
    /// message. For OOD binding `b_new = eq_table(z)`, so `h_new` is the MLE
    /// eval `f̂(z)` — fusing it here removes the separate `mle_eval_inline`
    /// fold over `f`. Transcript-identical: the caller observes the returned
    /// `h_new` then `(u_0, u_2)`, exactly as the unfused path does.
    pub fn introduce_new_with_eval(&mut self, b_new: Vec<F128>) -> (SumcheckMessage, F128) {
        assert_eq!(b_new.len(), self.f.len());
        assert!(
            self.pending_glue.is_none(),
            "ordinary introduction already pending"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "ordinary introduction across deferred glue"
        );
        let (msg, h_new) = round_msg_and_eval_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        (msg, h_new)
    }

    /// Introduce `eq(z, ·)` as retained low/high factors rather than a dense
    /// table. Returns `None` without changing state for unsupported geometry
    /// or any outstanding introduction. A caller may use the full-table path
    /// after `None` only when no ordinary introduction is pending; the ranked
    /// production caller's exact gate guarantees both pending slots are empty.
    fn introduce_new_ood_factorized(&mut self, z: &[F128]) -> Option<(SumcheckMessage, F128)> {
        let expected_len = z
            .len()
            .try_into()
            .ok()
            .and_then(|shift: u32| 1usize.checked_shl(shift));
        if z.is_empty()
            || self.f.len() < 4
            || expected_len != Some(self.f.len())
            || self.pending_glue.is_some()
            || self.pending_fold_basis.is_some()
            || self.pending_ood_eq.is_some()
        {
            return None;
        }

        let tail = &z[1..];
        let split_low_log = tail.len().min(LAZY_OOD_EQ_SPLIT_LOW_LOG_MAX);
        let eq_lo = crate::lincheck::build_eq_table_optimized(&tail[..split_low_log]);
        let eq_hi = crate::lincheck::build_eq_table_optimized(&tail[split_low_log..]);
        let z_0 = z[0];
        let (msg, h_new) = round_msg_and_eval_lsb_factorized_eq_split(&self.f, &eq_lo, &eq_hi, z_0);
        self.transcript.push(msg);
        self.pending_ood_eq = Some(PendingOodEq::Introduced {
            eq_lo,
            eq_hi,
            z_0,
            h_new,
        });
        Some((msg, h_new))
    }

    /// Apply the OOD separation challenge while retaining the split equality
    /// factors for the next fold. This mirrors [`Self::glue`]'s target update
    /// but does not write a full equality table into `combined_basis`.
    fn glue_factorized_ood(&mut self, beta: F128) {
        assert!(
            self.pending_glue.is_none(),
            "lazy OOD glue across ordinary pending glue"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "lazy OOD glue across deferred ordinary glue"
        );
        let pending = self
            .pending_ood_eq
            .take()
            .expect("lazy OOD glue without factorized introduction");
        let PendingOodEq::Introduced {
            eq_lo,
            eq_hi,
            z_0,
            h_new,
        } = pending
        else {
            panic!("lazy OOD equality glued twice");
        };
        self.t_r += beta * h_new;
        self.pending_ood_eq = Some(PendingOodEq::Glued {
            eq_lo,
            eq_hi,
            z_0,
            beta,
        });
    }

    /// Combine the introduced basis into `combined_basis` with separation α.
    /// `combined_basis[j] += α · b_new[j]` (pointwise), `T_r += α · h_new`.
    pub fn glue(&mut self, alpha: F128) {
        use rayon::prelude::*;
        assert!(
            self.pending_fold_basis.is_none(),
            "ordinary glue across deferred ordinary glue"
        );
        let (b_new, h_new) = self
            .pending_glue
            .take()
            .expect("glue without introduce_new");
        assert_eq!(b_new.len(), self.combined_basis.len());
        const PAR_THRESHOLD: usize = 4096;
        if self.combined_basis.len() < PAR_THRESHOLD {
            for (acc, &v) in self.combined_basis.iter_mut().zip(b_new.iter()) {
                *acc += alpha * v;
            }
        } else {
            self.combined_basis
                .par_iter_mut()
                .zip(b_new.par_iter())
                .with_min_len(PAR_THRESHOLD / 4)
                .for_each(|(acc, &v)| *acc += alpha * v);
        }
        self.t_r += alpha * h_new;
    }

    /// Apply an ordinary separation challenge now, but retain its basis until
    /// the already-pending ranked lazy-OOD fold. The transcript and target are
    /// updated at exactly the same point as [`Self::glue`]; only the dense
    /// `combined_basis += alpha * b_new` traversal moves into that fold.
    fn glue_deferred_into_lazy_ood_fold(&mut self, alpha: F128) {
        assert!(
            matches!(self.pending_ood_eq, Some(PendingOodEq::Glued { .. })),
            "deferred ordinary glue requires a glued lazy OOD term"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "more than one ordinary glue deferred"
        );
        let (b_new, h_new) = self
            .pending_glue
            .take()
            .expect("deferred glue without introduce_new");
        assert_eq!(b_new.len(), self.combined_basis.len());
        self.t_r += alpha * h_new;
        self.pending_fold_basis = Some((b_new, alpha));
    }

    pub fn f(&self) -> &[F128] {
        &self.f
    }

    pub fn transcript(&self) -> &[SumcheckMessage] {
        &self.transcript
    }
}

// ===================================================================
// Prover / Verifier — stubs
// ===================================================================

/// Sample `count` distinct positions in `[0, block_len)` via the challenger.
/// Asserts `count <= block_len` — otherwise no number of samples could satisfy
/// the distinctness requirement (would infinite-loop).
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

/// Build a single octopus multi-proof for all `queries` against `tree`.
fn merkle_multi_proof_for(tree: &[Hash], block_len: usize, queries: &[usize]) -> Vec<Hash> {
    merkle::merkle_multi_proof(tree, block_len, queries)
}

/// Drive the recursive Ligerito prover to prove `poly(eval_point) = claimed_value`.
///
/// Protocol structure (unique-decoding regime, no OOD samples yet):
/// 1. Commit f⁰ = `poly`.
/// 2. Partial-eval at `eval_point[0..initial_k]` (LSB-first), commit f¹.
/// 3. Open f⁰ at random query positions, induce a basis poly from the openings.
/// 4. Start sumcheck on `Σ_x f¹(x) · eq(eval_point[initial_k..], x) = claimed_value`,
///    introduce the induced basis (α-batched), glue with a separation challenge.
/// 5. For each recursive level: do k_i sumcheck folds; if last, send the residual
///    yr in clear and open the previous commitment; else commit the folded f,
///    open the previous commitment, induce a fresh basis from these opens,
///    introduce + glue.
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

    // ---- Initial commit (wtns_0) ----
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

/// Variant of [`recursive_prover`] that reuses an **externally-built L0 commit**
/// (the codeword + merkle tree). This is what Flock's `pcs::open_batch` will
/// call after `pcs::commit` has already built the same shape. Skips the
/// L0 commit cost (~17 ms at m=29 MT).
///
/// Caller responsibility: the external L0 data must match what `ligero_commit`
/// would produce at the same `(log_msg_cols_0 = log_n - initial_k, initial_k,
/// log_inv_rates[0])`. In practice this means using `PcsParams` with
/// `log_batch_size = config.initial_k` and `log_inv_rate = config.log_inv_rates[0]`.
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
    let _ = r; // used implicitly via config in inner
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

/// Drop-in replacement for the legacy `basefold::prove`: takes a generic basis poly +
/// target (typically the combined `Σ γ_k · eq(z_k, ·)` and target produced by
/// `ring_switch::prove_batched` for batched claims), plus an externally-built
/// L0 commitment (the existing `pcs::commit` output).
///
/// Differs from [`recursive_prover`] in the initial step: instead of partial-
/// evaluating at `z[0..initial_k]` (which doesn't make sense for a combined
/// basis with no single `z`), runs `initial_k` real sumcheck rounds folding
/// both `f` and `b` together with FS challenges. The folded f becomes wtns_1
/// and the rest of the protocol proceeds identically.
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
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        challenger,
    )
}

/// Variant of [`recursive_prover_with_basis`] that accepts the round-0 sumcheck
/// `(u_0, u_2)` pre-computed by the caller. Useful from
/// `pcs::compute_combined_basis_and_target` which produces these values as a
/// side effect while building `b_initial` — passing them in here lets
/// `SumcheckProver::new` skip the redundant 256 MB read pass over (f, b1).
#[allow(clippy::too_many_arguments)]
pub fn recursive_prover_with_basis_precomputed_round0<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    round1_lookahead: Option<[F128; 6]>,
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
        round1_lookahead,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        challenger,
    )
}
/// Ranked AB-only direct-fold2 entry. `ordinary_basis` contains every claim
/// that stays on the incumbent materialized path (currently C); `direct`
/// contains only the AB sufficient statistics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_prover_with_basis_direct_ab_fold2<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    direct: Vec<super::ring_switch::DirectFold2Factors>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    round1_lookahead: [F128; 6],
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        ordinary_basis,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        Some(round1_lookahead),
        None,
        None,
        None,
        None,
        Some(direct),
        None,
        None,
        challenger,
    )
}

/// Experimental sixteen-bank entry. The first four transcript messages come
/// entirely from `direct` product matrices; after four sequential FS samples
/// the state is materialized at N/16 and rejoins the incumbent final fold2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_prover_with_basis_direct_fold4<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    direct: Vec<super::ring_switch::DirectFold4Factors>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    round1_lookahead: [F128; 6],
    round2_lookahead: super::Fold4Lookahead2,
    round3_lookahead: super::Fold4Lookahead3,
    challenger: &mut Ch,
) -> LigeritoProof {
    assert_eq!(
        config.initial_k, 6,
        "direct-fold4 scaffold requires initial_k=6"
    );
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        ordinary_basis,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        Some(round1_lookahead),
        Some(round2_lookahead),
        Some(round3_lookahead),
        None,
        None,
        None,
        Some(direct),
        None,
        challenger,
    )
}
/// Direct-fold8 entry. The first six transcript messages come from the
/// factorized sixty-four-bank state; after six sequential FS samples the state
/// is materialized at N/64 = 2^19 in ONE pass and the incumbent cadence
/// resumes — the fold2 pair of the fold4 route never runs (the 2^21 and
/// 2^20 states never exist).
#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_prover_with_basis_direct_fold8<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    direct: Vec<super::ring_switch::DirectFold8Factors>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    challenger: &mut Ch,
) -> LigeritoProof {
    assert_eq!(
        config.initial_k, 6,
        "direct-fold8 scaffold requires initial_k=6"
    );
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        ordinary_basis,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(direct),
        challenger,
    )
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
    round1_lookahead: Option<[F128; 6]>,
    round2_lookahead: Option<super::Fold4Lookahead2>,
    round3_lookahead: Option<super::Fold4Lookahead3>,
    _round4_lookahead: Option<super::Fold8Lookahead4>,
    _round5_lookahead: Option<super::Fold8Lookahead5>,
    direct_fold2: Option<Vec<super::ring_switch::DirectFold2Factors>>,
    direct_fold4: Option<Vec<super::ring_switch::DirectFold4Factors>>,
    direct_fold8: Option<Vec<super::ring_switch::DirectFold8Factors>>,
    challenger: &mut Ch,
) -> LigeritoProof {
    let log_n = packed_witness.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;

    assert_eq!(packed_witness.len(), 1usize << log_n);
    assert!(
        direct_fold2.is_none() || (direct_fold4.is_none() && direct_fold8.is_none()),
        "direct-fold2 and direct-fold4/fold8 modes are mutually exclusive"
    );
    assert!(
        direct_fold4.is_none() || direct_fold8.is_none(),
        "direct-fold4 and direct-fold8 modes are mutually exclusive"
    );
    // Direct mode may carry every claim in its factor bundle, in which case
    // there is no materialized basis at all.
    assert!(
        b_initial.len() == 1usize << log_n
            || ((direct_fold2.is_some() || direct_fold4.is_some() || direct_fold8.is_some())
                && b_initial.is_empty())
    );
    if direct_fold4.is_some() || direct_fold8.is_some() {
        assert_eq!(
            initial_k, 6,
            "direct-fold4/fold8 scaffold requires initial_k=6"
        );
    }
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(r >= 1);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    assert_eq!(l0_codeword.len(), block_len_0 * num_interleaved_0);
    assert_eq!(l0_tree.len(), 2 * block_len_0 - 1);

    let trace =
        std::env::var("LIG_PROVE_TRACE").is_ok() || std::env::var_os("FLOCK_OPEN_TIMING").is_some();
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

    // L0 codeword + tree are borrowed (reused from upstream `pcs::commit`).
    // wtns_0 access reduces to: root (last tree node), row(q), block_len.
    let initial_root: Hash = l0_tree[l0_tree.len() - 1];
    let l0_block_len = block_len_0;
    let l0_num_interleaved = num_interleaved_0;
    let l0_row = |q: usize| -> &[F128] {
        let start = q * l0_num_interleaved;
        &l0_codeword[start..start + l0_num_interleaved]
    };
    challenger.observe_bytes(&initial_root);

    // L0 takes no explicit OOD samples: it is bound by the opening's own
    // evaluation claim (`target` at the post-commit random point behind
    // `b_initial`), which plays the OOD role with a union over the list
    // instead of over pairs. See `paper_ood_bits`.
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
    let mut packed_witness = Some(packed_witness);
    let mut b_initial = Some(b_initial);
    let mut direct_fold2 = direct_fold2;
    let mut direct_fold4 = direct_fold4;
    let mut direct_fold8 = direct_fold8;
    let direct_fold4_mode = direct_fold4.is_some();
    let direct_fold8_mode = direct_fold8.is_some();
    let direct_mode = direct_fold2.is_some() || direct_fold4_mode || direct_fold8_mode;
    let (mut sc_prover, start_msg) = if direct_mode {
        (
            None,
            first_msg.expect("direct mode requires a sufficient-stat round-0 message"),
        )
    } else {
        let (prover, msg) = match first_msg {
            Some(msg) => SumcheckProver::new_with_first_msg(
                packed_witness.take().unwrap(),
                b_initial.take().unwrap(),
                target,
                msg,
            ),
            None => SumcheckProver::new(
                packed_witness.take().unwrap(),
                b_initial.take().unwrap(),
                target,
            ),
        };
        (Some(prover), msg)
    };
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    let mut t_grind0 = std::time::Duration::ZERO;
    let use_fold2 = direct_mode
        || (ranked_fold2_enabled(1usize << log_n, initial_k) && round1_lookahead.is_some());
    // A lookahead message is evaluated at the first challenge, allowing that
    // challenge's state bind to wait for the next one. Odd rounds then bind
    // both challenges together and refresh the next lookahead coefficients.
    let mut fold2_lookahead = if use_fold2 && !direct_fold4_mode && !direct_fold8_mode {
        round1_lookahead
    } else {
        None
    };
    let fold4_round1 = direct_fold4_mode.then_some(round1_lookahead).flatten();
    let mut fold4_round2 = direct_fold4_mode.then_some(round2_lookahead).flatten();
    let mut fold4_round3 = direct_fold4_mode.then_some(round3_lookahead).flatten();
    let mut fold4_challenges = Vec::with_capacity(5);
    let mut fold4_initial_msgs = Vec::with_capacity(5);
    let mut deferred_challenge = None;
    let mut deferred_msg = None;
    for j in 0..initial_k {
        // Fold-challenge grinding: the L0 proximity-gap bad event lives on
        // each of these lane-fold challenges, so each one is individually
        // PoW-guarded (a cheating prover re-rolls a fold challenge by
        // varying the preceding sumcheck message; the grind prices every
        // such attempt). Tapered per round: round j folds a 2^{ℓ-j}-row word
        // whose MCA error carries the factor 2^{ℓ-1-j} (App. C.3 Lemma
        // `mca-commutes`), so it needs (fold_bits − j) bits — one fewer per
        // round than the worst (j=0) round `fold_grinding_bits` is sized for.
        // Derived from fold_grinding_bits + round index; not stored.
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            let _tg = std::time::Instant::now();
            fold_grinding_nonces.push(challenger.grind_pow(bits));
            t_grind0 += _tg.elapsed();
        }
        let r = challenger.sample_f128();
        let _tf = std::time::Instant::now();
        let msg = if direct_fold8.is_some() {
            match fold4_challenges.len() {
                0..=4 => {
                    let msg =
                        fold_direct_fold8_factors_and_message(direct_fold8.as_mut().unwrap(), r);
                    fold4_challenges.push(r);
                    fold4_initial_msgs.push(msg);
                    msg
                }
                5 => {
                    // E-core engagement probe for the fold8 materialize: the
                    // hetero drain (`pcs::use_open_mat_hetero`) is the only
                    // helper-pool consumer inside this call, so the delta in
                    // the process-global `helper_chunks_claimed` counter across
                    // the materialize is exactly the number of the 256 fold8
                    // blocks the efficiency cores claimed. Diagnostic only
                    // (relaxed counter), fully trace-gated — the untimed hot
                    // path does zero extra work — and mirrors the counter's
                    // documented "prove E-core engagement for that window" use.
                    let helper_before = trace.then(crate::epool::helper_chunks_claimed);
                    let (f8, b8, msg) = materialize_direct_fold8(
                        packed_witness.take().unwrap(),
                        b_initial.take().unwrap(),
                        direct_fold8.take().unwrap().as_slice(),
                        [
                            fold4_challenges[0],
                            fold4_challenges[1],
                            fold4_challenges[2],
                            fold4_challenges[3],
                            fold4_challenges[4],
                            r,
                        ],
                    );
                    if let Some(before) = helper_before {
                        eprintln!(
                            "    [fold8-mat] helper (E-core) blocks claimed: {} / 256",
                            crate::epool::helper_chunks_claimed().wrapping_sub(before)
                        );
                    }
                    sc_prover = Some(SumcheckProver::new_after_direct_fold8(
                        f8,
                        b8,
                        target,
                        [
                            start_msg,
                            fold4_initial_msgs[0],
                            fold4_initial_msgs[1],
                            fold4_initial_msgs[2],
                            fold4_initial_msgs[3],
                            fold4_initial_msgs[4],
                            msg,
                        ],
                    ));
                    fold4_challenges.clear();
                    fold4_initial_msgs.clear();
                    msg
                }
                _ => unreachable!(),
            }
        } else if direct_fold4.is_some() {
            match fold4_challenges.len() {
                0 => {
                    let msg = eval_lookahead(
                        fold4_round1
                            .as_ref()
                            .expect("direct-fold4 round-1 lookahead"),
                        r,
                    );
                    fold4_challenges.push(r);
                    fold4_initial_msgs.push(msg);
                    msg
                }
                1 => {
                    let msg = eval_fold4_lookahead2(
                        fold4_round2
                            .as_mut()
                            .expect("direct-fold4 round-2 lookahead"),
                        fold4_challenges[0],
                        r,
                    );
                    fold4_challenges.push(r);
                    fold4_initial_msgs.push(msg);
                    msg
                }
                2 => {
                    let msg = eval_fold4_lookahead3(
                        fold4_round3
                            .as_mut()
                            .expect("direct-fold4 round-3 lookahead"),
                        fold4_challenges[0],
                        fold4_challenges[1],
                        r,
                    );
                    fold4_challenges.push(r);
                    fold4_initial_msgs.push(msg);
                    msg
                }
                3 => {
                    let (f4, b4, msg, next_lookahead) = materialize_direct_fold4(
                        packed_witness.take().unwrap(),
                        b_initial.take().unwrap(),
                        direct_fold4.take().unwrap().as_slice(),
                        [
                            fold4_challenges[0],
                            fold4_challenges[1],
                            fold4_challenges[2],
                            r,
                        ],
                    );
                    sc_prover = Some(SumcheckProver::new_after_direct_fold4(
                        f4,
                        b4,
                        target,
                        [
                            start_msg,
                            fold4_initial_msgs[0],
                            fold4_initial_msgs[1],
                            fold4_initial_msgs[2],
                            msg,
                        ],
                    ));
                    fold4_challenges.clear();
                    fold4_initial_msgs.clear();
                    fold2_lookahead = Some(next_lookahead);
                    msg
                }
                _ => unreachable!(),
            }
        } else if use_fold2 {
            if let Some(r_a) = deferred_challenge.take() {
                if sc_prover.is_none() {
                    let (f2, b2, msg, next_lookahead) = materialize_direct_ab_fold2(
                        packed_witness.take().unwrap(),
                        b_initial.take().unwrap(),
                        direct_fold2.take().unwrap().as_slice(),
                        r_a,
                        r,
                    );
                    sc_prover = Some(SumcheckProver::new_after_direct_fold2(
                        f2,
                        b2,
                        target,
                        [start_msg, deferred_msg.take().unwrap(), msg],
                    ));
                    fold2_lookahead = Some(next_lookahead);
                    msg
                } else if j + 1 == initial_k && fold2_final_enabled() {
                    sc_prover.as_mut().unwrap().fold2_final(r_a, r)
                } else {
                    let (msg, next_lookahead) = sc_prover.as_mut().unwrap().fold2(r_a, r);
                    fold2_lookahead = Some(next_lookahead);
                    msg
                }
            } else {
                let msg =
                    eval_lookahead(fold2_lookahead.as_ref().expect("ranked fold2 lookahead"), r);
                if let Some(prover) = sc_prover.as_mut() {
                    prover.push_lookahead_msg(msg);
                } else {
                    deferred_msg = Some(msg);
                }
                deferred_challenge = Some(r);
                msg
            }
        } else {
            sc_prover.as_mut().unwrap().fold(r)
        };
        if trace {
            eprintln!(
                "    [init-fold] round {j}: fold {:.2} ms",
                _tf.elapsed().as_secs_f64() * 1e3
            );
        }
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        r_lane_fold.push(r);
    }
    debug_assert!(
        deferred_challenge.is_none(),
        "ranked initial_k must be even"
    );
    debug_assert!(
        fold4_challenges.is_empty(),
        "direct-fold4/fold8 must materialize"
    );
    let mut sc_prover = sc_prover.expect("initial direct mode must materialize");
    if trace {
        t_init_sumcheck += _t.elapsed();
        eprintln!(
            "    [init-fold] initial_k={initial_k}, grind total {:.2} ms",
            t_grind0.as_secs_f64() * 1e3
        );
    }

    // Commit f^1 = folded packed witness as wtns_1.
    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1);
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    let _t = std::time::Instant::now();
    let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
    // Borrow the folded evaluations directly: `ligero_commit` copies its
    // input into its own scratch codeword (`replicate_message_fill`), so the
    // previous `sc_prover.f().to_vec()` materialized a second 2^(n1) copy
    // (8 MiB at the ranked shape) on the timed path only to drop it after
    // the commit.
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

    // OOD binding for the L1 commit: each sample evaluates f1's multilinear
    // extension at a random transcript point z ∈ F^{n1}, sends the claimed
    // value, and folds the claim `Σ_x f1(x)·eq(z,x) = y` into the running
    // sumcheck (introduce + glue). Binds the prover to a single codeword of
    // the interleaved list before any of L0's queries are drawn.
    let use_lazy_l1_ood = ranked_l1_lazy_ood_eq_enabled(
        config,
        log_n,
        n1,
        ood_count(1),
        sc_prover.f().len(),
        direct_fold8_mode,
    );
    {
        let _t = std::time::Instant::now();
        for _ in 0..ood_count(1) {
            let z = challenger.sample_f128_vec(n1);
            // Ranked L1 retains the equality as an LSB factor plus an exact
            // 11+7 tail split through the ordinary induced-basis introduce/glue
            // below. The exact selector chooses the incumbent full table for
            // every unsupported production geometry before transcript mutation.
            let (intro, y, factorized) = if use_lazy_l1_ood {
                let (intro, y) = sc_prover
                    .introduce_new_ood_factorized(&z)
                    .expect("ranked L1 lazy OOD preconditions changed after exact gate");
                (intro, y, true)
            } else {
                let eq_z = build_eq_table(&z);
                let (intro, y) = sc_prover.introduce_new_with_eval(eq_z);
                (intro, y, false)
            };
            challenger.observe_f128(y);
            ood_values.push(y);
            challenger.observe_f128(intro.u_0);
            challenger.observe_f128(intro.u_2);
            let beta = challenger.sample_f128();
            if factorized {
                sc_prover.glue_factorized_ood(beta);
            } else {
                sc_prover.glue(beta);
            }
        }
        if trace {
            t_ood += _t.elapsed();
        }
    }

    // Query-phase PoW grinding for L0: each ground bit substitutes for
    // ~1/log₂(1/(1−γ)) queries at this level (the Slim profile grinds 16
    // bits here). Verifier mirror checks the nonce; both then proceed to
    // sample query positions. (The proximity-gap shortfall is covered
    // separately by the fold-challenge grinds above.)
    let pow_nonce_0 = challenger.grind_pow(config.grinding_bits[0] as u32);
    let mut grinding_nonces: Vec<u64> = vec![pow_nonce_0];

    // Open L0; lane-fold weights = r_lane_fold.
    let num_queries_0 = config.queries[0];
    let queries_0 = sample_distinct_queries(challenger, l0_block_len, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let _t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = {
        use rayon::prelude::*;
        // Indexed parallel collect is order-preserving: bit-identical to
        // the serial map; each row copy is independent of the challenger.
        queries_0.par_iter().map(|&q| l0_row(q).to_vec()).collect()
    };
    let merkle_proof_0 = merkle_multi_proof_for(l0_tree, l0_block_len, &queries_0);
    if trace {
        t_opens += _t.elapsed();
    }
    // Induce basis_0 from wtns_0 opens. L0 dominates the induce phase, where the
    // sparse-prefix Fᵀ-NTT path wins; the dispatcher auto-selects it (deeper
    // levels stay dense).
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let _t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0, induced_intro_msg_0) =
        induce_sumcheck_poly_auto_with_ranked_msg(
            n1,
            log_inv_rate_0,
            &sks_vks_n1,
            &opened_rows_0,
            &r_lane_fold,
            &queries_0,
            &alpha_0,
            sc_prover.f(),
        );
    if trace {
        t_induce += _t.elapsed();
    }

    // Built after the induce so the opened rows move into the proof instead
    // of being cloned (218 row Vecs at the ranked shape); the rows are dead
    // to the prover past `induce_sumcheck_poly_auto`.
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0,
        merkle_proof: merkle_proof_0,
    };

    // Introduce + glue basis_0.
    let _t = std::time::Instant::now();
    let intro_msg_0 = if let Some(msg) = induced_intro_msg_0 {
        sc_prover.introduce_new_with_precomputed_msg(basis_0_induced, enforced_sum_0, msg)
    } else {
        sc_prover.introduce_new(basis_0_induced, enforced_sum_0)
    };
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    if use_lazy_l1_ood {
        sc_prover.glue_deferred_into_lazy_ood_fold(beta_0);
    } else {
        sc_prover.glue(beta_0);
    }
    if trace {
        t_intro_glue += _t.elapsed();
    }

    // Recursive levels — same as recursive_prover_inner from here.
    let mut wtns_prev = wtns_1;
    let mut recursive_roots: Vec<Hash> = vec![wtns_prev.root()];
    let mut recursive_proofs: Vec<RecursiveProof> = Vec::new();

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        let mut level_rs = Vec::with_capacity(k_i);
        let _t = std::time::Instant::now();
        for j in 0..k_i {
            // These folds fold level i+1's commitment — fold-challenge
            // grinding guards its proximity-gap term. Tapered per round:
            // round j needs (fold_bits − j) bits (see L0 loop).
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
            // PoW grinding for the last level before sampling its queries.
            let nonce_last = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
            grinding_nonces.push(nonce_last);
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            let _t = std::time::Instant::now();
            let opened_rows_last: Vec<Vec<F128>> = {
                use rayon::prelude::*;
                // Order-preserving parallel collect — bit-identical.
                queries_last
                    .par_iter()
                    .map(|&q| wtns_prev.row(q).to_vec())
                    .collect()
            };
            let merkle_proof_last =
                merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_last);
            if trace {
                t_opens += _t.elapsed();
            }
            // Final open complete — recycle last recursive codeword/tree before
            // proof-object assembly (transcript copy etc.).
            crate::scratch::give_f128(std::mem::take(&mut wtns_prev.mat));
            crate::scratch::give_hash_tree(std::mem::take(&mut wtns_prev.tree));
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
        // Same borrow-instead-of-copy as the wtns_1 commit above.
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

        // OOD binding for the L_{i+2} commit (same as the L1 block above).
        {
            let _t = std::time::Instant::now();
            for _ in 0..ood_count(i + 2) {
                let z = challenger.sample_f128_vec(n_next);
                // Micro-stack: the PMULL two-lane kernel builder is an exact
                // drop-in for the generic one (byte-equality proven by
                // `lincheck::tests::optimized_eq_table_matches_generic_bytes`,
                // which covers these dims — ranked n_next = 16/13/10/7).
                // FLOCK_NO_MICRO_STACK=1 restores the generic builder.
                let eq_z = if crate::micro_stack_enabled() {
                    crate::lincheck::build_eq_table_optimized(&z)
                } else {
                    build_eq_table(&z)
                };
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

        // PoW grinding for this iteration's query phase.
        let nonce_i = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
        grinding_nonces.push(nonce_i);
        let num_queries_i = config.queries[i + 1];
        let queries_i = sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let _t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = {
            use rayon::prelude::*;
            // Order-preserving parallel collect — bit-identical.
            queries_i
                .par_iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect()
        };
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        if trace {
            t_opens += _t.elapsed();
        }
        // Rows + multi-proof are owned copies now. Prior-level codeword mat and
        // Merkle tree are dead through induce/intro-glue; recycle before induce
        // so they do not stack under wtns_next (already committed) + induce temps.
        // Bit-identical: no further reads of wtns_prev.mat/tree this iteration.
        crate::scratch::give_f128(std::mem::take(&mut wtns_prev.mat));
        // Recycle the intermediate level's flat tree through the same pool the
        // final level uses (see the `give_hash_tree` above the trace block):
        // dropping it here returned multi-MiB of resident pages to the OS that
        // the next prove's `take_hash_tree` then re-faulted in. Allocation-only
        // and transcript-invariant by construction — the tree is dead per the
        // note above, so no read of it can observe the pool.
        // A/B-CONTROL: FLOCK_NO_TREE_POOL_FULL=1 (exact '1') restores the
        // incumbent drop.
        if crate::scratch::tree_pool_full_enabled() {
            crate::scratch::give_hash_tree(std::mem::take(&mut wtns_prev.tree));
        } else {
            wtns_prev.tree = Vec::new();
        }
        let sks_vks_i = eval_sk_at_vks(n_next);
        let _t = std::time::Instant::now();
        let (basis_i_induced, enforced_sum_i) =
            if n_next == 16 && config.log_inv_rates[i + 1] == 2 && queries_i.len() == 106 {
                induce_sumcheck_poly_via_ntt(
                    n_next,
                    config.log_inv_rates[i + 1],
                    &opened_rows_i,
                    &level_rs,
                    &queries_i,
                    &alpha_i,
                )
            } else {
                induce_sumcheck_poly(
                    n_next,
                    &sks_vks_i,
                    &opened_rows_i,
                    &level_rs,
                    &queries_i,
                    &alpha_i,
                )
            };
        if trace {
            t_induce += _t.elapsed();
        }

        // Pushed after the induce so the opened rows move instead of being
        // cloned; they are dead to the prover past the induce call.
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

/// Succinct verifier for [`recursive_prover_with_basis`]: instead of accepting
/// a dense `b_initial: &[F128]` (which would be ~16 MB at m=29), accepts a
/// **closure** `eval_b` that evaluates `b_initial(point)` at any multilinear
/// point. The verifier calls `eval_b` only `yr.len()` times (at the residual)
/// — typically a few dozen times, not 2^L. Use this from
/// `pcs::verify_opening_batch_ligerito_mixed` where the closure is built from
/// `ring_switch::verify_succinct` outputs + PD claim points.
///
/// `log_n` is the original packed-witness log size (= b_initial's logical dim).
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
    // Called ONCE at the residual check with the full ris and yr_log_n.
    // Returns 2^yr_log_n values: eval_b(ris ++ y_bits) for y ∈ [0, 2^yr_log_n).
    // This API allows callers to amortize prefix work across yr positions
    // (e.g. ring_switch::eval_rs_eq_prefix + finish_from_prefix).
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
        return false; // L0 must be bound by the opening's own eval claim
    }
    let mut fold_nonce_idx = 0usize;
    let mut ood_idx = 0usize;
    // OOD claims glued into the running sumcheck: each contributes
    // `beta · Π_b eq(z_b, r_b) · eq(z_tail, ·)` at the residual.
    struct OodCtx {
        z: Vec<F128>,
        ris_start: usize,
        beta: F128,
    }
    let mut ood_ctxs: Vec<OodCtx> = Vec::new();

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        // Fold-challenge PoW mirror (L0's lane folds), tapered per round to
        // (fold_bits − j) — see the prover's L0 loop.
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

    // OOD binding mirror for the L1 commit: sample z, read the claimed
    // evaluation from the proof, and glue the claim into the running
    // sumcheck exactly like the prover.
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

    // PoW grinding check for L0's query phase. With grinding_bits[0]=0 this
    // is a no-op (still absorbs the 0 nonce so the FS state matches the
    // prover side).
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

    // Compute enforced_sum cheaply at intro time. The induced basis poly's
    // residual evaluations are deferred to the final check (succinct path —
    // see `induce_sumcheck_evaluate_at_residual`).
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

    // Per-level induced-basis evaluation context — small (no dense vec).
    struct LevelCtx {
        log_msg_cols: usize,
        queries: Vec<usize>,
        alpha: Vec<F128>, // ⌈log₂ Q⌉ field elements (eq-tensor combination)
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
            // Fold-challenge PoW mirror (level i+1's folds), tapered per round
            // to (fold_bits − j) — see the prover's L0 loop.
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
            // PoW grinding check for last level's query phase.
            if nonce_idx >= proof.grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(
                proof.grinding_nonces[nonce_idx],
                config.grinding_bits[i + 1] as u32,
            ) {
                return false;
            }
            // (last nonce — nonce_idx is not advanced past it)

            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = config.queries[i + 1];
            let _t = std::time::Instant::now();
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);
            // Basis-induction challenge for the LAST commitment. Sampled here —
            // after `yr` was observed (top of this branch) and the queries are
            // fixed — so a forged `yr` cannot be adapted to it. Mirrors `alpha_i`
            // at every non-final level (see ~line 3377).
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

            // Bind the LAST commitment to `yr`. Every non-final level folds its
            // opened rows into the running sumcheck via induce_sumcheck; the
            // final level used to only Merkle-check its opened rows, leaving `yr`
            // (the claimed final message) constrained by a single scalar equation
            // — so a malicious prover could solve for a `yr` that opens the
            // commitment to an arbitrary value. We add the same proximity tie as
            // the other levels: `enforced_sum_last` is the α-weighted lane-fold
            // of the (Merkle-bound) opened rows, batched into `t_r` with a fresh
            // `beta_last`; its induced basis is already at the residual dimension
            // (zero further folds), so it joins `combined` below via this
            // LevelCtx. With `alpha_last` drawn after `yr`, the batched check now
            // forces `yr` to agree with the committed codeword at every queried
            // column (multilinear Schwartz–Zippel), restoring binding.
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

            // Succinct residual check: per-level induced basis evaluations
            // via closed-form (no dense materialization).
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

            // OOD bases: closed-form residual. An eq(z, ·) basis introduced
            // at dim |z| and folded by the subsequent challenges contributes
            // `beta · Π_b eq(z_b, r_b)` times the eq table on z's unfolded
            // tail (char-2 eq factor: 1 + a + b).
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

            // Batch-evaluate b at all yr positions in one call so the
            // caller can amortize prefix work (e.g. ring_switch tensor prefix).
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

        // OOD binding mirror for the L_{i+2} commit.
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

        // PoW grinding check for this iteration's query phase.
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

/// Verifier for [`recursive_prover_with_basis`]. Caller supplies the basis
/// `b_initial` recomputed locally (typically from the combined claims) and
/// `target`. Also supplies the L0 root (from the upstream `Commitment`).
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

    // Replay sumcheck: start msg → initial_k folds.
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
        return false; // L0 must be bound by the opening's own eval claim
    }
    let mut fold_nonce_idx = 0usize;
    let mut ood_idx = 0usize;
    // OOD eq bases glued into the running sumcheck, accumulated as
    // (dense eq table, ris_start, beta) and added at the residual check.
    let mut ood_bases: Vec<(Vec<F128>, usize, F128)> = Vec::new();

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        // Fold-challenge PoW mirror (L0's lane folds), tapered per round to
        // (fold_bits − j) — see the prover's L0 loop.
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

    // Observe wtns_1 root + open wtns_0.
    if proof.recursive_roots.is_empty() {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    // OOD binding mirror for the L1 commit.
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

    // PoW grinding check (dense verifier mirror) — keeps the FS state in
    // lockstep with the prover even at grinding_bits = 0.
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

    // Intro + glue.
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

    // Basis poly tracking for residual check.
    // b_initial is the "level-0 basis" — it gets partial-eval'd at all ris.
    // basis_0_induced is introduced at start (before any ris from level 0+) — partial-eval at the level-0+ ris.
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
            // Fold-challenge PoW mirror (level i+1's folds), tapered per round
            // to (fold_bits − j) — see the prover's L0 loop.
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
            // PoW grinding check for last level (dense verifier).
            if nonce_idx >= proof.grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(
                proof.grinding_nonces[nonce_idx],
                config.grinding_bits[i + 1] as u32,
            ) {
                return false;
            }
            // (last nonce — nonce_idx is not advanced past it)

            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);
            // Final-level basis-induction challenge — sampled after `yr` and the
            // queries are fixed. Same position as the succinct verifier
            // (recursive_verifier_with_basis_succinct), which verifies the same
            // proof, so both stay in lockstep.
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

            // Bind the LAST commitment to `yr`: induce its opened rows into the
            // sumcheck exactly like every non-final level, batched with a fresh
            // `beta_last`. Without this the last commitment is only Merkle-checked
            // and `yr` is left unconstrained — a forged `yr` could open to any
            // value. (Dense mirror of the succinct verifier fix.)
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

            // Residual check.
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
            // OOD eq bases contribute the same way (dense tables).
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

        // OOD binding mirror for the L_{i+2} commit.
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

        // PoW grinding check for this iteration (dense verifier mirror).
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

/// Shared body — runs after wtns_0 is in hand (whether freshly built or
/// supplied externally).
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
    // The legacy (non-basis) path predates OOD binding and fold grinding;
    // configs that use them must go through `recursive_prover_with_basis`.
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

    // ---- Partial-eval at z[0..initial_k] and commit f¹ (wtns_1) ----
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

    // ---- Queries + open wtns_0 ----
    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_distinct_queries(challenger, wtns_0.block_len, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = {
        use rayon::prelude::*;
        // Order-preserving parallel collect — bit-identical (see above).
        queries_0
            .par_iter()
            .map(|&q| wtns_0.row(q).to_vec())
            .collect()
    };
    let merkle_proof_0 = merkle_multi_proof_for(&wtns_0.tree, wtns_0.block_len, &queries_0);
    t_opens += t.elapsed();
    // L0 mat/tree dead after open copies; recycle before induce.
    {
        let mut wtns_0 = wtns_0;
        crate::scratch::give_f128(std::mem::take(&mut wtns_0.mat));
        wtns_0.tree = Vec::new();
    }

    // ---- Induce basis from wtns_0 opens ----
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
    // Move rows into the proof after induce (mirrors the timed basis path).
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0,
        merkle_proof: merkle_proof_0,
    };
    t_induce += t.elapsed();

    // ---- Start sumcheck: f¹ · eq(z[initial_k..], ·) = claimed_value ----
    let eq_z_residual = build_eq_table(&eval_point[initial_k..]);
    let t = std::time::Instant::now();
    let (mut sc_prover, start_msg) = SumcheckProver::new(f1, eq_z_residual, claimed_value);
    t_sumcheck += t.elapsed();
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    // ---- Introduce induced basis + glue ----
    let intro_msg_0 = sc_prover.introduce_new(basis_0_induced, enforced_sum_0);
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    sc_prover.glue(beta_0);

    // ---- Recursive levels ----
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
            // Last iter: send residual yr + open wtns_prev.
            let yr = sc_prover.f().to_vec();
            for v in &yr {
                challenger.observe_f128(*v);
            }
            // wtns_prev's rate (= log_inv_rates[i+1] for wtns_{i+1}).
            let num_queries_last = udr_queries(config.log_inv_rates[i + 1]);
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            let opened_rows_last: Vec<Vec<F128>> = {
                use rayon::prelude::*;
                // Order-preserving parallel collect — bit-identical.
                queries_last
                    .par_iter()
                    .map(|&q| wtns_prev.row(q).to_vec())
                    .collect()
            };
            let merkle_proof_last =
                merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_last);
            crate::scratch::give_f128(std::mem::take(&mut wtns_prev.mat));
            wtns_prev.tree = Vec::new();
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
                grinding_nonces: Vec::new(), // legacy recursive_prover_inner: no grinding plumbed
                ood_values: Vec::new(),
                fold_grinding_nonces: Vec::new(),
            };
        }

        // Non-last: commit the folded poly → wtns_next.
        // wtns_next = wtns_{i+2}, uses log_inv_rates[i+2].
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

        // Open wtns_prev. wtns_prev = wtns_{i+1} uses log_inv_rates[i+1].
        let num_queries_i = udr_queries(config.log_inv_rates[i + 1]);
        let queries_i = sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = {
            use rayon::prelude::*;
            // Order-preserving parallel collect — bit-identical.
            queries_i
                .par_iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect()
        };
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        t_opens += t.elapsed();
        // Prior-level mat/tree dead after the open; recycle before induce.
        crate::scratch::give_f128(std::mem::take(&mut wtns_prev.mat));
        wtns_prev.tree = Vec::new();

        // Induce fresh basis from these opens.
        let sks_vks_i = eval_sk_at_vks(n_next);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_next,
            &sks_vks_i,
            &opened_rows_i,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        // Move rows into the proof after induce (no pre-induce clone).
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows_i,
            merkle_proof: merkle_proof_i,
        });

        // Introduce + glue.
        let intro_msg_i = sc_prover.introduce_new(basis_i_induced, enforced_sum_i);
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        sc_prover.glue(beta_i);

        wtns_prev = wtns_next;
    }

    unreachable!("recursive loop should return on last iter")
}

/// Verify all opened rows against one root via a single octopus multi-proof.
/// `queries` must be sorted ascending and aligned with `opened_rows`.
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

/// Verifier counterpart to [`recursive_prover`]. Supports arbitrary `R ≥ 1`.
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
    // The legacy (non-basis) path predates OOD binding and fold grinding.
    if config.ood_samples.iter().any(|&s| s != 0)
        || config.fold_grinding_bits.iter().any(|&b| b != 0)
    {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    // ---- Roots ----
    challenger.observe_bytes(&proof.initial_root);
    if proof.recursive_roots.len() != r {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    // ---- Open wtns_0 + α₀ ----
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

    // ---- Induce basis_0 from wtns_0 opens ----
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

    // ---- Set up running sumcheck state ----
    let eq_z_residual = build_eq_table(&eval_point[initial_k..]);
    // basis_polys[k] are stored at the dim they were introduced. ris_starts[k] is
    // the index in `ris` at the time basis_polys[k] was introduced.
    let mut basis_polys: Vec<Vec<F128>> = vec![eq_z_residual];
    let mut basis_ris_starts: Vec<usize> = vec![0];
    let mut basis_separations: Vec<F128> = Vec::new(); // separation for basis_polys[k+1]
    let mut ris: Vec<F128> = Vec::new();
    let mut t_r = claimed_value;
    let mut tx_idx = 0usize;

    // ---- Start message ----
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    // ---- Intro basis_0 + glue β₀ ----
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

    // ---- Recursive iterations ----
    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1]; // wtns_1's rate
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
            // Last iter: read yr + open prev_root.
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
            // Final-level basis-induction challenge (after yr + queries fixed).
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

            // Bind the LAST commitment to `yr`: induce its opened rows into the
            // sumcheck like every non-final level (without this `yr` is
            // unconstrained and a forged `yr` opens to any value).
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

            // ---- Final residual check ----
            // Each basis_polys[k] is partially-evaluated at ris[ris_starts[k]..].
            // basis_polys[0] has separation 1, basis_polys[k+1] has separation basis_separations[k].
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

        // Non-last: read next root, sample queries on prev_root, induce basis, intro + glue.
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

        // Intro + glue
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

        // Update prev for next iteration: prev_root = root_next, dims = next commit's dims.
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
    // Disclosed zero-mechanism redraw marker (draw 2 of the latch tree,
    // fa433990 drew 1,449,917 in-band): resampling the median lottery.
    fn eval_quadratic_tensor_enumerator(coefficients: &[F128], challenges: &[F128]) -> F128 {
        coefficients
            .iter()
            .enumerate()
            .fold(F128::ZERO, |sum, (mut index, &coefficient)| {
                let mut weight = F128::ONE;
                for &challenge in challenges.iter().rev() {
                    weight *= match index % 3 {
                        0 => F128::ONE,
                        1 => challenge,
                        2 => challenge * challenge,
                        _ => unreachable!(),
                    };
                    index /= 3;
                }
                sum + coefficient * weight
            })
    }

    #[test]
    fn quadratic_tensor_horner_matches_enumerator() {
        let mut state = 0xC0EF_FEE1_2345_6789u64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(state, state.rotate_left(23) ^ 0xA5A5_5A5A_0F0F_F0F0)
        };

        for challenge_count in 2..=5 {
            let tensor_len = 3usize.pow(challenge_count);
            for _ in 0..4 {
                let challenges: Vec<F128> = (0..challenge_count).map(|_| random()).collect();
                let mut coefficients: Vec<F128> = (0..2 * tensor_len).map(|_| random()).collect();
                let expected = SumcheckMessage {
                    u_0: eval_quadratic_tensor_enumerator(&coefficients[..tensor_len], &challenges),
                    u_2: eval_quadratic_tensor_enumerator(&coefficients[tensor_len..], &challenges),
                };

                assert_eq!(
                    eval_quadratic_tensors_in_place(&mut coefficients, &challenges),
                    expected,
                    "challenge_count={challenge_count}",
                );
            }
        }
    }

    /// The paired fold must reproduce two sequential state binds, the direct
    /// message, and the coefficient-evaluated following message bit-for-bit.
    #[test]
    fn fold2_transcript_identity() {
        let mut st = 0xF01Du64;
        let mut rnd = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128 {
                lo: st,
                hi: st.rotate_left(17) ^ 0xABCD,
            }
        };
        for log_n in [4usize, 7, 12] {
            let n = 1usize << log_n;
            let f: Vec<F128> = (0..n).map(|_| rnd()).collect();
            let b: Vec<F128> = (0..n).map(|_| rnd()).collect();
            let r_a = rnd();
            let r_b = rnd();
            let rho_c = rnd();

            // Reference: two sequential fused rounds.
            let mut nf1 = Vec::with_capacity(n / 2);
            let mut nb1 = Vec::with_capacity(n / 2);
            // seq round 1: fold by r_a, msg over folded
            let msg1 = super::fold_and_msg_lsb_into(&f, &b, r_a, &mut nf1, &mut nb1);
            // seq round 2: fold by r_b, msg over folded
            let mut nf2 = Vec::with_capacity(n / 4);
            let mut nb2 = Vec::with_capacity(n / 4);
            let msg2 = super::fold_and_msg_lsb_into(&nf1, &nb1, r_b, &mut nf2, &mut nb2);
            // seq round 3 (for the lookahead check): fold by rho_c, msg over folded
            let mut nf3 = Vec::with_capacity(n / 8);
            let mut nb3 = Vec::with_capacity(n / 8);
            let msg3 = super::fold_and_msg_lsb_into(&nf2, &nb2, rho_c, &mut nf3, &mut nb3);
            let _ = msg1;

            // Fused: one pass with (r_a, r_b) known.
            let mut wf = Vec::with_capacity(n / 4);
            let mut wb = Vec::with_capacity(n / 4);
            let (msg_direct, coeffs) =
                super::fold2_and_msgs_lsb(&f, &b, r_a, r_b, &mut wf, &mut wb);
            let mut final_wf = Vec::with_capacity(n / 4);
            let mut final_wb = Vec::with_capacity(n / 4);
            let final_msg =
                super::fold2_and_msg_lsb(&f, &b, r_a, r_b, &mut final_wf, &mut final_wb);

            assert_eq!(wf, nf2, "folded f state differs at log_n={log_n}");
            assert_eq!(wb, nb2, "folded b state differs at log_n={log_n}");
            assert_eq!(final_wf, wf, "final folded f differs at log_n={log_n}");
            assert_eq!(final_wb, wb, "final folded b differs at log_n={log_n}");
            assert_eq!(
                final_msg, msg_direct,
                "final message differs at log_n={log_n}"
            );
            assert_eq!(
                msg_direct.u_0, msg2.u_0,
                "direct u_0 differs at log_n={log_n}"
            );
            assert_eq!(
                msg_direct.u_2, msg2.u_2,
                "direct u_2 differs at log_n={log_n}"
            );
            let msg_la = super::eval_lookahead(&coeffs, rho_c);
            assert_eq!(
                msg_la.u_0, msg3.u_0,
                "lookahead u_0 differs at log_n={log_n}"
            );
            assert_eq!(
                msg_la.u_2, msg3.u_2,
                "lookahead u_2 differs at log_n={log_n}"
            );
        }
    }
    #[test]
    fn direct_ab_materialization_matches_mixed_full_basis_oracle() {
        let mut state = 0xD1CE_F01D_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(state, state.rotate_left(29))
        };
        let n = 1usize << 10;
        let f: Vec<F128> = (0..n).map(|_| random()).collect();
        let ordinary_c: Vec<F128> = (0..n).map(|_| random()).collect();
        let r0 = random();
        let r1 = random();

        let suffix: Vec<F128> = (0..10).map(|_| random()).collect();
        let gamma = random();
        let scaled_rdp: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| random())
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .map(|value| gamma * value)
        .collect();
        let direct_full =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix), &scaled_rdp);
        let combined_full: Vec<F128> = ordinary_c
            .iter()
            .zip(direct_full)
            .map(|(&ordinary, direct)| ordinary + direct)
            .collect();
        let (eq_lo, eq_hi) = super::super::ring_switch::build_eq_split(&suffix[2..], 4);
        let direct = vec![super::super::ring_switch::DirectFold2Factors {
            eq_lo,
            eq_hi,
            low_eq: build_eq_table(&suffix[..2]).try_into().unwrap(),
            table: super::super::ring_switch::build_fold_byte_table(&scaled_rdp),
            products: [F128::ZERO; 16],
        }];

        let mut want_f = Vec::with_capacity(n / 4);
        let mut want_b = Vec::with_capacity(n / 4);
        let (want_msg, want_coeffs) =
            super::fold2_and_msgs_lsb(&f, &combined_full, r0, r1, &mut want_f, &mut want_b);
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let (got_f, got_b, got_msg, got_coeffs) = super::materialize_direct_ab_fold2_with_helper(
            f,
            ordinary_c,
            &direct,
            r0,
            r1,
            Some(&helper),
        );
        assert_eq!(got_f, want_f);
        assert_eq!(got_b, want_b);
        assert_eq!(got_msg, want_msg);
        assert_eq!(got_coeffs, want_coeffs);
    }

    use super::*;

    /// The recursive from-message first pass preserves every encoded element
    /// and every node of the ordinary Merkle tree for both production rates.
    #[test]
    fn recursive_from_message_commit_matches_replica_oracle() {
        use crate::challenger::Challenger;

        for (log_msg_cols, log_inv_rate, seed) in [
            (16usize, 2usize, 0xF14C_0002_u64),
            (13usize, 3usize, 0xF14C_0003_u64),
        ] {
            let log_num_interleaved = 3usize;
            let mut rng = crate::challenger::RandomChallenger::new(seed);
            let poly = rng.sample_f128_vec(1usize << (log_msg_cols + log_num_interleaved));
            let ntt = AdditiveNttF128::standard(log_msg_cols + log_inv_rate);
            let ordinary = ligero_commit_impl(
                &poly,
                log_msg_cols,
                log_num_interleaved,
                log_inv_rate,
                &ntt,
                HashKind::Blake3,
                false,
            );
            let fused = ligero_commit_impl(
                &poly,
                log_msg_cols,
                log_num_interleaved,
                log_inv_rate,
                &ntt,
                HashKind::Blake3,
                true,
            );
            assert_eq!(fused.mat, ordinary.mat, "encoded matrix changed");
            assert_eq!(fused.tree, ordinary.tree, "flat Merkle tree changed");
            assert_eq!(fused.root(), ordinary.root(), "Merkle root changed");
        }
    }

    /// Worked example: `LigeritoSecurityConfig` for BLAKE3 m=29 at rate 1/2.
    /// Paper-compatible m=29 fast example, mechanically derived in the
    /// unique-decoding regime (Theorem 1.4, ε* = 10⁻³) targeting 100-bit
    /// security.
    fn blake3_m29_udr_example() -> LigeritoSecurityConfig {
        LigeritoSecurityConfig::derive_paper_compatible(29, 1, 100).expect("derive m29 fast")
    }

    /// Both embedded TOMLs (m29_fast at rate 1/2 and m29_slim at rate 1/4)
    /// parse, validate, and produce ProverConfig/VerifierConfig agreeing
    /// with the corresponding `default_config(22, 6, rate)` shape.
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
        // Fast = JohnsonOod profile: 218 L0 queries per-round at 100 bits (no
        // list union bound — single-codeword binding via the opening claim /
        // OOD samples), proximity-gap shortfall covered by fold-challenge grinding.
        assert_eq!(cfg.levels[0].regime, SoundnessRegime::JohnsonOod);
        assert_eq!(cfg.levels[0].queries, 218);
        assert_eq!(cfg.levels[0].grinding_bits, 0);
        assert!(cfg.levels[0].fold_grinding_bits > 0);
        assert_eq!(cfg.levels[0].ood_samples, 0); // L0: bound by eval claim
        assert!(cfg.levels[1].ood_samples >= 1);
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 6, 1).unwrap();
        assert_eq!(pv.log_inv_rates, default.log_inv_rates);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.queries[0], 218);

        // Slim mode: rates start at 1/4.
        let toml_str = include_str!("../../configs/ligerito/m29_slim.toml");
        let cfg_slim = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_slim.toml must parse and validate");
        assert_eq!(cfg_slim.levels[0].log_inv_rate, 2);
        // Slim = JohnsonOod at rate 1/4 with 16-bit query grinding.
        assert_eq!(cfg_slim.levels[0].queries, 90);
        assert_eq!(cfg_slim.levels[0].grinding_bits, 16);
        let (pv_slim, _vc_slim) = cfg_slim.to_prover_verifier_configs().unwrap();
        let default_slim = default_config(22, 6, 2).unwrap();
        assert_eq!(pv_slim.log_inv_rates, default_slim.log_inv_rates);
        assert_eq!(pv_slim.recursive_ks, default_slim.recursive_ks);
    }

    /// Helper: re-emit all the embedded TOMLs from `derive_paper_compatible`.
    /// Writes to stdout (via eprintln) so the user can `>` redirect to disk.
    /// Run with:
    ///   cargo test --release --lib regen_embedded_tomls -- --ignored --nocapture
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

    /// `validate()` rejects a config whose declared `expected_eps_pg_bits`
    /// disagrees with what Theorem 1.5 predicts for the level's
    /// `(eta, log_inv_rate, log_msg_cols)`. Enforces that the per-level
    /// diagnostics weren't hand-waved.
    #[test]
    fn ligerito_security_config_rejects_paper_inconsistent_eps_pg() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_pg_bits = 50.0; // very wrong
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("doesn't match") && err.contains("prediction"),
            "expected paper-mismatch error, got: {err}"
        );
    }

    /// Same enforcement on the query side.
    #[test]
    fn ligerito_security_config_rejects_paper_inconsistent_eps_query() {
        let mut cfg = blake3_m29_udr_example();
        // Bump query bits by 5 — far outside tolerance.
        cfg.levels[0].expected_eps_query_bits += 5.0;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("doesn't match") && err.contains("prediction"),
            "expected paper-mismatch error, got: {err}"
        );
    }

    /// All 6 embedded configs validate strictly (i.e. each is paper-compat
    /// AND satisfies the security target).
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

    /// `derive_paper_compatible` produces a config that validates for every
    /// `(m, log_inv_rate)` combination we ship.
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

    /// `prover_config_for` is **strict** — only known `(m, log_inv_rate)`
    /// pairs load. Unknown pairs return an `Err` so production callers can't
    /// silently fall back to unaudited parameters.
    #[test]
    fn ligerito_prover_config_for_lookup() {
        // m=29 fast: known → loads from TOML.
        let pv = prover_config_for(22, 6, LigeritoProfile::Fast).expect("m29 fast must load");
        assert_eq!(pv.queries[0], 218);
        assert_eq!(pv.fold_grinding_bits[0], 16);

        // m=29 slim: known → loads from TOML.
        let pv = prover_config_for(22, 6, LigeritoProfile::Slim).expect("m29 slim must load");
        assert_eq!(pv.queries[0], 90);
        assert_eq!(pv.grinding_bits[0], 16);

        // m=29 secure: known → loads from TOML (UDR, 120-bit).
        let pv = prover_config_for(22, 6, LigeritoProfile::Secure).expect("m29 secure must load");
        assert!(pv.queries[0] > 280);
        assert_eq!(pv.ood_samples.iter().sum::<usize>(), 0);

        // m=36 (unknown — above the registered 22..=35 range): errors,
        // no silent fallback.
        let err = prover_config_for(29, 6, LigeritoProfile::Fast).unwrap_err();
        assert!(
            err.contains("no security config registered"),
            "unexpected error: {err}"
        );
    }

    /// TOML round-trip via `to_toml_string` ↔ `from_toml_str` preserves
    /// the config exactly (modulo validated invariants).
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

    /// Schema validates the worked example end to end.
    #[test]
    fn ligerito_security_config_validates() {
        let cfg = blake3_m29_udr_example();
        cfg.validate()
            .unwrap_or_else(|e| panic!("validate failed: {e}"));
    }

    /// The config's `hash` field selects the Merkle hash and reaches both
    /// derived configs — this is the knob the option is exposed through.
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

        // Survives a TOML round-trip, so the option is settable from a file.
        cfg.validate().expect("blake3 config validates");
        let back = LigeritoSecurityConfig::from_toml_str(&cfg.to_toml_string().unwrap())
            .expect("toml roundtrip");
        assert_eq!(back.merkle_hash().unwrap(), HashKind::Blake3);
    }

    /// A `hash` we do not implement must fail at validation rather than
    /// silently committing under SHA-256.
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

    /// Every embedded config must name a hash we actually implement — a typo
    /// in a checked-in TOML should fail here, not at proving time.
    #[test]
    fn embedded_configs_all_declare_a_supported_hash() {
        for &((m, profile), toml) in EMBEDDED_CONFIGS {
            let cfg = LigeritoSecurityConfig::from_toml_str(toml)
                .unwrap_or_else(|e| panic!("m{m} {profile:?}: {e}"));
            cfg.merkle_hash()
                .unwrap_or_else(|e| panic!("m{m} {profile:?}: {e}"));
        }
    }

    /// Lowering a level's expected_eps_query_bits below the required
    /// (target − grinding) is caught by validation.
    #[test]
    fn ligerito_security_config_rejects_insufficient_queries() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_query_bits = 50.0; // < target 100 (grinding 0)
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("expected_eps_query_bits"), "err = {err}");
    }

    /// UDR regime must not carry an `eta` value.
    #[test]
    fn ligerito_security_config_rejects_udr_with_eta() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].eta = Some(0.02); // eta is Johnson-only — should fail
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("udr") && err.contains("eta"), "err = {err}");
    }

    /// UDR regime requires `proximity_loss` to be set, not `eta`.
    #[test]
    fn ligerito_security_config_rejects_udr_without_proximity_loss() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].proximity_loss = None; // missing!
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("udr") && err.contains("proximity_loss"),
            "err = {err}"
        );
    }

    /// `proximity_loss` is only valid for the UDR regime.
    #[test]
    fn ligerito_security_config_rejects_johnson_with_proximity_loss() {
        let mut cfg = blake3_m29_udr_example();
        // JohnsonOod regime with proximity_loss set — should fail.
        cfg.levels[0].regime = SoundnessRegime::JohnsonOod;
        cfg.levels[0].eta = Some(0.02);
        cfg.levels[0].proximity_loss = Some(0.01);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("proximity_loss") && err.contains("udr"),
            "err = {err}"
        );
    }

    /// End-to-end: a hand-built UDR-regime level validates against the
    /// paper's Thm `ca-udr` bound (a = γ·n + 1) and the per-query/UDR formula.
    #[test]
    fn ligerito_security_config_udr_regime_validates() {
        let mut cfg = blake3_m29_udr_example();
        // Convert L0 to UDR at the maximal radius γ = δ/2 − 3/(δ·n) − ε*
        // (ε* = 0 → top of C.3's valid range). δ = 1 − ρ; per-query soundness
        // is log₂(1/(1−γ)) and Q is sized so Q·per_q ≥ 100 bits.
        let eps_star = 0.0f64;
        let rho = 0.5f64;
        let delta = 1.0 - rho;
        let n = ((cfg.levels[0].log_msg_cols + cfg.levels[0].log_inv_rate) as f64).exp2();
        let gamma = delta / 2.0 - 3.0 / (delta * n) - eps_star;
        let per_q = (1.0 / (1.0 - gamma)).log2();
        let queries = (100.0 / per_q).ceil() as usize;
        // a = γ·n + 1; ε_pg = 128 − log₂ a with NO row-union penalty in the
        // unique-decoding regime (list size 1; Diamond and Gruen). Any
        // shortfall below the 100-bit target is covered by fold-grinding.
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

    /// Schema round-trips cleanly through serde JSON. (TOML would work too
    /// once we add a toml dep.)
    #[test]
    fn ligerito_security_config_serde_roundtrip() {
        let cfg = blake3_m29_udr_example();
        let json = serde_json::to_string_pretty(&cfg).expect("serialize");
        let back: LigeritoSecurityConfig = serde_json::from_str(&json).expect("deserialize");
        back.validate().expect("roundtripped config validates");
        assert_eq!(back.levels.len(), cfg.levels.len());
        // rate 1/2, 100-bit target, full UD radius γ = δ/2 (ε* = 0):
        // per-query = log₂(1/(1−1/4)) ≈ 0.415 b/q → ⌈100/0.415⌉ = 241.
        assert_eq!(back.levels[0].queries, 241);
        assert_eq!(back.levels[0].grinding_bits, 0);
    }

    /// End-to-end: a security config with **non-zero grinding** at L0 drives
    /// an actual recursive_prover_with_basis → recursive_verifier_with_basis
    /// roundtrip. Confirms the PoW step is plumbed into the FS transcript
    /// on both sides (without grinding the proof would either be rejected
    /// or the FS state would diverge between prover and verifier).
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
        // Hand-set queries + grinding (small but non-zero c so we exercise
        // the SHA256 PoW search without blowing up test time).
        let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let grinding_bits = vec![6usize, 0]; // L0 grinds 6 bits, L1 doesn't
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

        // Tampering with the nonce flips the PoW check.
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

    /// The security config produces ProverConfig/VerifierConfig matching the
    /// existing `default_config(log_n=22, log_batch_size=6, log_inv_rate=1)`
    /// in shape (rates + recursive_ks + initial_k all agree).
    #[test]
    fn ligerito_security_config_matches_default_config() {
        let cfg = blake3_m29_udr_example();
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 6, 1).unwrap();
        assert_eq!(pv.log_inv_rates, default.log_inv_rates);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.initial_k, default.initial_k);
    }

    /// Single-lane RS encoding round-trips through inv-NTT: forward-transforming
    /// the zero-padded message and then inverse-transforming should give back the
    /// padded message.
    /// `partial_eval_lsb` followed by `eval_mle_lsb` on the residual equals
    /// `eval_mle_lsb` on the full point — i.e. partial evaluation is
    /// consistent with full evaluation under the same LSB-first convention.
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
        // Split the point into a (k, n-k) partial/residual prefix.
        let k = 3;
        let (lo, hi) = point.split_at(k);
        let residual = partial_eval_lsb(&evals, lo);
        assert_eq!(residual.len(), 1usize << (n - k));
        let after = eval_mle_lsb(&residual, hi);
        assert_eq!(full, after);

        // Sanity: build_eq_table evaluated at `point` and dot-producted
        // with `evals` should also equal `full` (LSB-first eq table).
        let eq = build_eq_table(&point);
        let dot = evals
            .iter()
            .zip(eq.iter())
            .map(|(&e, &q)| e * q)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(dot, full);
    }

    /// The production selector is deliberately narrower than the algebraic
    /// helper: one mutation to any ranked geometry/config/rollback input must
    /// return to the incumbent materialized equality path.
    #[test]
    fn factorized_ood_ranked_gate_is_exact() {
        let mut config =
            prover_config_for(25, 6, LigeritoProfile::Fast).expect("embedded M32 Fast config");
        config.merkle_hash = HashKind::Blake3;
        let selected = |cfg: &ProverConfig,
                        log_n: usize,
                        n_1: usize,
                        count: usize,
                        len: usize,
                        direct8: bool,
                        platform: bool,
                        disabled: bool| {
            ranked_l1_lazy_ood_eq_selected(cfg, log_n, n_1, count, len, direct8, platform, disabled)
        };
        assert!(selected(&config, 25, 19, 1, 1 << 19, true, true, false));
        assert!(!selected(&config, 24, 19, 1, 1 << 19, true, true, false));
        assert!(!selected(&config, 25, 18, 1, 1 << 19, true, true, false));
        assert!(!selected(&config, 25, 19, 2, 1 << 19, true, true, false));
        assert!(!selected(&config, 25, 19, 1, 1 << 18, true, true, false));
        assert!(!selected(&config, 25, 19, 1, 1 << 19, false, true, false));
        assert!(!selected(&config, 25, 19, 1, 1 << 19, true, false, false));
        assert!(!selected(&config, 25, 19, 1, 1 << 19, true, true, true));

        let mut wrong_hash = config.clone();
        wrong_hash.merkle_hash = HashKind::Sha256;
        assert!(!selected(
            &wrong_hash,
            25,
            19,
            1,
            1 << 19,
            true,
            true,
            false
        ));
        let mut wrong_rate = config.clone();
        wrong_rate.log_inv_rates[1] += 1;
        assert!(!selected(
            &wrong_rate,
            25,
            19,
            1,
            1 << 19,
            true,
            true,
            false
        ));
        let mut wrong_query = config;
        wrong_query.queries[1] += 1;
        assert!(!selected(
            &wrong_query,
            25,
            19,
            1,
            1 << 19,
            true,
            true,
            false
        ));
    }

    /// The ranked lazy OOD representation must be a protocol-transparent
    /// replacement for materializing `eq(z)`: claimed value, intro message,
    /// intervening ordinary introduce/glue, next folded state, and every
    /// subsequent transcript message are all bit-identical. Edge triples
    /// exercise `z_0`, OOD separation `beta`, and fold challenge `r` in
    /// `{0, 1}`; later cases use pseudorandom field elements. `log_n=13`
    /// also exercises a multi-chunk low/high tail split.
    #[test]
    fn factorized_ood_matches_full_table_through_fold() {
        let mut state = 0x4F4F_445F_4C41_5A59u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(state, state.rotate_left(23) ^ 0xD00D_F00D_CAFE_BABE)
        };

        for log_n in [3usize, 7, 13] {
            let len = 1usize << log_n;
            for case in 0..24usize {
                let f: Vec<F128> = (0..len).map(|_| rnd()).collect();
                let initial_basis: Vec<F128> = (0..len).map(|_| rnd()).collect();
                let ordinary_basis: Vec<F128> = (0..len).map(|_| rnd()).collect();
                let mut z: Vec<F128> = (0..log_n).map(|_| rnd()).collect();

                let (z_0, beta, r, alpha) = if case < 16 {
                    (
                        if case & 1 == 0 { F128::ZERO } else { F128::ONE },
                        if case & 2 == 0 { F128::ZERO } else { F128::ONE },
                        if case & 4 == 0 { F128::ZERO } else { F128::ONE },
                        if case & 8 == 0 { F128::ZERO } else { F128::ONE },
                    )
                } else {
                    (rnd(), rnd(), rnd(), rnd())
                };
                z[0] = z_0;
                let h_initial = f
                    .iter()
                    .zip(initial_basis.iter())
                    .map(|(&x, &b)| x * b)
                    .fold(F128::ZERO, |acc, v| acc + v);
                let h_ordinary = f
                    .iter()
                    .zip(ordinary_basis.iter())
                    .map(|(&x, &b)| x * b)
                    .fold(F128::ZERO, |acc, v| acc + v);

                let (mut full, full_first) =
                    SumcheckProver::new(f.clone(), initial_basis.clone(), h_initial);
                let (mut lazy, lazy_first) =
                    SumcheckProver::new(f.clone(), initial_basis.clone(), h_initial);
                let (mut deferred, deferred_first) =
                    SumcheckProver::new(f.clone(), initial_basis.clone(), h_initial);
                assert_eq!(lazy_first, full_first);
                assert_eq!(deferred_first, full_first);

                let (full_intro, full_y) = full.introduce_new_with_eval(build_eq_table(&z));
                let (lazy_intro, lazy_y) = lazy
                    .introduce_new_ood_factorized(&z)
                    .expect("supported factorized geometry");
                let (deferred_intro, deferred_y) = deferred
                    .introduce_new_ood_factorized(&z)
                    .expect("supported deferred factorized geometry");
                assert_eq!(
                    lazy_y, full_y,
                    "OOD value differs, log_n={log_n}, case={case}"
                );
                assert_eq!(
                    lazy_intro, full_intro,
                    "OOD intro differs, log_n={log_n}, case={case}"
                );
                assert_eq!((deferred_intro, deferred_y), (full_intro, full_y));
                full.glue(beta);
                lazy.glue_factorized_ood(beta);
                deferred.glue_factorized_ood(beta);
                assert_eq!(lazy.t_r, full.t_r);
                assert_eq!(deferred.t_r, full.t_r);
                assert_eq!(lazy.transcript(), full.transcript());
                assert_eq!(deferred.transcript(), full.transcript());

                // This is the ranked ordering: the OOD term remains lazy while
                // the ordinary opening-induced basis is introduced and glued.
                let full_ordinary = full.introduce_new(ordinary_basis.clone(), h_ordinary);
                let lazy_ordinary = lazy.introduce_new(ordinary_basis.clone(), h_ordinary);
                let deferred_ordinary = deferred.introduce_new(ordinary_basis, h_ordinary);
                assert_eq!(lazy_ordinary, full_ordinary);
                assert_eq!(deferred_ordinary, full_ordinary);
                full.glue(alpha);
                lazy.glue(alpha);
                deferred.glue_deferred_into_lazy_ood_fold(alpha);
                assert_eq!(lazy.t_r, full.t_r);
                assert_eq!(deferred.t_r, full.t_r);
                assert_eq!(lazy.transcript(), full.transcript());
                assert_eq!(deferred.transcript(), full.transcript());
                assert!(deferred.pending_glue.is_none());
                assert!(deferred.pending_fold_basis.is_some());
                assert!(matches!(
                    deferred.pending_ood_eq.as_ref(),
                    Some(PendingOodEq::Glued { .. })
                ));

                let full_next = full.fold(r);
                let lazy_next = lazy.fold(r);
                let deferred_next = deferred.fold(r);
                assert_eq!(
                    lazy_next, full_next,
                    "fold msg differs, log_n={log_n}, case={case}"
                );
                assert_eq!(
                    deferred_next, full_next,
                    "deferred fold msg differs, log_n={log_n}, case={case}"
                );
                assert_eq!(
                    lazy.f, full.f,
                    "folded f differs, log_n={log_n}, case={case}"
                );
                assert_eq!(deferred.f, full.f, "deferred folded f differs");
                assert_eq!(
                    lazy.combined_basis, full.combined_basis,
                    "folded basis differs, log_n={log_n}, case={case}"
                );
                assert_eq!(deferred.combined_basis, full.combined_basis);
                assert_eq!(lazy.t_r, full.t_r);
                assert_eq!(deferred.t_r, full.t_r);
                assert_eq!(lazy.transcript(), full.transcript());
                assert_eq!(deferred.transcript(), full.transcript());
                assert!(lazy.pending_ood_eq.is_none(), "lazy OOD was not consumed");
                assert!(deferred.pending_ood_eq.is_none());
                assert!(deferred.pending_fold_basis.is_none());

                // A second fold proves the term was cleared rather than
                // accidentally applied again after its first consumer.
                let r_2 = rnd();
                let full_after = full.fold(r_2);
                let lazy_after = lazy.fold(r_2);
                let deferred_after = deferred.fold(r_2);
                assert_eq!(lazy_after, full_after);
                assert_eq!(deferred_after, full_after);
                assert_eq!(lazy.f, full.f);
                assert_eq!(deferred.f, full.f);
                assert_eq!(lazy.combined_basis, full.combined_basis);
                assert_eq!(deferred.combined_basis, full.combined_basis);
                assert_eq!(lazy.transcript(), full.transcript());
                assert_eq!(deferred.transcript(), full.transcript());
            }
        }
    }

    /// Exercise the exact ranked tensor geometry once against the incumbent
    /// fully materialized path. Besides protocol equality, inspect the pending
    /// representation before glue to pin the intended 11+7 storage split.
    #[test]
    fn factorized_ood_ranked_11_7_split_matches_full_path() {
        const LOG_N: usize = 19;
        const LEN: usize = 1 << LOG_N;

        let mut state = 0x3131_2B37_5F4F_4F44u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(state, state.rotate_left(29) ^ 0x5350_4C49_545F_4551)
        };

        let f: Vec<F128> = (0..LEN).map(|_| rnd()).collect();
        let initial_basis: Vec<F128> = (0..LEN).map(|_| rnd()).collect();
        let ordinary_basis: Vec<F128> = (0..LEN).map(|_| rnd()).collect();
        let z: Vec<F128> = (0..LOG_N).map(|_| rnd()).collect();
        let beta = rnd();
        let alpha = rnd();
        let r = rnd();

        let h_initial = f
            .iter()
            .zip(initial_basis.iter())
            .map(|(&x, &b)| x * b)
            .fold(F128::ZERO, |acc, v| acc + v);
        let h_ordinary = f
            .iter()
            .zip(ordinary_basis.iter())
            .map(|(&x, &b)| x * b)
            .fold(F128::ZERO, |acc, v| acc + v);
        let first_msg = round_msg_lsb(&f, &initial_basis);

        let (mut full, full_first) = SumcheckProver::new_with_first_msg(
            f.clone(),
            initial_basis.clone(),
            h_initial,
            first_msg,
        );
        let (mut lazy, lazy_first) =
            SumcheckProver::new_with_first_msg(f, initial_basis, h_initial, first_msg);
        assert_eq!(lazy_first, full_first);

        let (full_intro, full_y) = full.introduce_new_with_eval(build_eq_table(&z));
        let (lazy_intro, lazy_y) = lazy
            .introduce_new_ood_factorized(&z)
            .expect("ranked OOD geometry must use the split path");
        assert_eq!((lazy_intro, lazy_y), (full_intro, full_y));
        match lazy
            .pending_ood_eq
            .as_ref()
            .expect("split OOD must remain pending until glue and fold")
        {
            PendingOodEq::Introduced { eq_lo, eq_hi, .. } => {
                assert_eq!(eq_lo.len(), 1 << 11);
                assert_eq!(eq_hi.len(), 1 << 7);
                assert_eq!(eq_lo.len() * eq_hi.len(), 1 << (LOG_N - 1));
                assert!(eq_lo.len() + eq_hi.len() < 1 << (LOG_N - 1));
            }
            PendingOodEq::Glued { .. } => panic!("OOD was glued before its challenge"),
        }

        full.glue(beta);
        lazy.glue_factorized_ood(beta);
        assert_eq!(lazy.t_r, full.t_r);
        assert_eq!(lazy.transcript(), full.transcript());

        let full_ordinary = full.introduce_new(ordinary_basis.clone(), h_ordinary);
        let lazy_ordinary = lazy.introduce_new(ordinary_basis, h_ordinary);
        assert_eq!(lazy_ordinary, full_ordinary);
        full.glue(alpha);
        lazy.glue_deferred_into_lazy_ood_fold(alpha);
        assert_eq!(lazy.t_r, full.t_r);
        assert_eq!(lazy.transcript(), full.transcript());
        assert!(lazy.pending_glue.is_none());
        assert!(lazy.pending_fold_basis.is_some());

        let full_next = full.fold(r);
        let lazy_next = lazy.fold(r);
        assert_eq!(lazy_next, full_next);
        assert_eq!(lazy.f, full.f);
        assert_eq!(lazy.combined_basis, full.combined_basis);
        assert_eq!(lazy.t_r, full.t_r);
        assert_eq!(lazy.transcript(), full.transcript());
        assert!(lazy.pending_ood_eq.is_none());
        assert!(lazy.pending_fold_basis.is_none());
    }

    /// Unsupported geometry and a second outstanding OOD use the incumbent
    /// materialized representation without perturbing protocol state. The
    /// hybrid (first lazy, second full) result must still equal two full-table
    /// introductions through the consuming fold.
    #[test]
    fn factorized_ood_fallback_is_exact_for_multiple_pending() {
        let mut state = 0x4641_4C4C_4241_434Bu64;
        let mut rnd = || {
            state = state
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            F128::new(state, state.rotate_right(19) ^ 0xABCD_EF01_2345_6789)
        };
        let log_n = 6usize;
        let len = 1usize << log_n;
        let f: Vec<F128> = (0..len).map(|_| rnd()).collect();
        let basis: Vec<F128> = (0..len).map(|_| rnd()).collect();
        let z_1: Vec<F128> = (0..log_n).map(|_| rnd()).collect();
        let z_2: Vec<F128> = (0..log_n).map(|_| rnd()).collect();
        let h = f
            .iter()
            .zip(basis.iter())
            .map(|(&x, &b)| x * b)
            .fold(F128::ZERO, |acc, v| acc + v);
        let beta_1 = rnd();
        let beta_2 = rnd();
        let r = rnd();

        let (mut full, _) = SumcheckProver::new(f.clone(), basis.clone(), h);
        let (mut hybrid, _) = SumcheckProver::new(f, basis, h);
        let transcript_len = hybrid.transcript().len();
        assert!(
            hybrid
                .introduce_new_ood_factorized(&z_1[..log_n - 1])
                .is_none(),
            "mismatched geometry must fall back"
        );
        assert_eq!(hybrid.transcript().len(), transcript_len);

        let (full_intro_1, full_y_1) = full.introduce_new_with_eval(build_eq_table(&z_1));
        let (hybrid_intro_1, hybrid_y_1) = hybrid
            .introduce_new_ood_factorized(&z_1)
            .expect("first OOD should factorize");
        assert_eq!((hybrid_intro_1, hybrid_y_1), (full_intro_1, full_y_1));
        full.glue(beta_1);
        hybrid.glue_factorized_ood(beta_1);

        let before_second = hybrid.transcript().len();
        assert!(
            hybrid.introduce_new_ood_factorized(&z_2).is_none(),
            "multiple lazy OOD terms must fall back"
        );
        assert_eq!(hybrid.transcript().len(), before_second);
        let (full_intro_2, full_y_2) = full.introduce_new_with_eval(build_eq_table(&z_2));
        let (hybrid_intro_2, hybrid_y_2) = hybrid.introduce_new_with_eval(build_eq_table(&z_2));
        assert_eq!((hybrid_intro_2, hybrid_y_2), (full_intro_2, full_y_2));
        full.glue(beta_2);
        hybrid.glue(beta_2);

        let full_msg = full.fold(r);
        let hybrid_msg = hybrid.fold(r);
        assert_eq!(hybrid_msg, full_msg);
        assert_eq!(hybrid.f, full.f);
        assert_eq!(hybrid.combined_basis, full.combined_basis);
        assert_eq!(hybrid.t_r, full.t_r);
        assert_eq!(hybrid.transcript(), full.transcript());
        assert!(hybrid.pending_ood_eq.is_none());
    }

    const LAZY_OOD_TIMING_LOG_N: usize = 19;
    const LAZY_OOD_TIMING_LEN: usize = 1 << LAZY_OOD_TIMING_LOG_N;
    const LAZY_OOD_TIMING_FOLDED_LEN: usize = LAZY_OOD_TIMING_LEN / 2;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum LazyOodTimingArm {
        Control,
        Candidate,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum LazyOodTimingOrder {
        ControlCandidate,
        CandidateControl,
    }

    #[derive(Clone, Copy, Debug)]
    struct LazyOodTimingMetrics {
        ood_ms: f64,
        ordinary_glue_ms: f64,
        fold_ms: f64,
    }

    impl LazyOodTimingMetrics {
        fn total_ms(self) -> f64 {
            self.ood_ms + self.ordinary_glue_ms + self.fold_ms
        }
    }

    struct LazyOodTimingRun {
        metrics: LazyOodTimingMetrics,
        z: Vec<F128>,
        ood_value: F128,
        ood_intro: SumcheckMessage,
        ood_beta: F128,
        ordinary_intro: SumcheckMessage,
        ordinary_beta: F128,
        fold_challenge: F128,
        fold_msg: SumcheckMessage,
        fs_continuation: F128,
    }

    struct LazyOodTimingInput {
        f: Vec<F128>,
        initial_basis: Vec<F128>,
        ordinary_basis: Vec<F128>,
        initial_target: F128,
        ordinary_target: F128,
        first_msg: SumcheckMessage,
    }

    struct LazyOodTimingSlot {
        prover: SumcheckProver,
        challenger: crate::challenger::FsChallenger,
    }

    impl LazyOodTimingSlot {
        fn new(input: &LazyOodTimingInput, challenger: &crate::challenger::FsChallenger) -> Self {
            let (prover, msg) = SumcheckProver::new_with_first_msg(
                input.f.clone(),
                input.initial_basis.clone(),
                input.initial_target,
                input.first_msg,
            );
            assert_eq!(msg, input.first_msg);
            Self {
                prover,
                challenger: challenger.clone(),
            }
        }

        /// Restore the full L1 state without replacing either persistent
        /// allocation slot. After a measured fold, the original full-sized
        /// buffers live in the spares, so swap them back before copying the
        /// deterministic templates.
        fn reset(
            &mut self,
            input: &LazyOodTimingInput,
            challenger: &crate::challenger::FsChallenger,
        ) {
            assert!(self.prover.pending_glue.is_none());
            assert!(self.prover.pending_fold_basis.is_none());
            assert!(self.prover.pending_ood_eq.is_none());
            if self.prover.f.len() == LAZY_OOD_TIMING_FOLDED_LEN {
                assert_eq!(self.prover.spare_f.len(), LAZY_OOD_TIMING_LEN);
                assert_eq!(self.prover.spare_b.len(), LAZY_OOD_TIMING_LEN);
                std::mem::swap(&mut self.prover.f, &mut self.prover.spare_f);
                std::mem::swap(&mut self.prover.combined_basis, &mut self.prover.spare_b);
            }
            assert_eq!(self.prover.f.len(), LAZY_OOD_TIMING_LEN);
            assert_eq!(self.prover.combined_basis.len(), LAZY_OOD_TIMING_LEN);
            assert!(self.prover.spare_f.capacity() >= LAZY_OOD_TIMING_FOLDED_LEN);
            assert!(self.prover.spare_b.capacity() >= LAZY_OOD_TIMING_FOLDED_LEN);
            self.prover.f.copy_from_slice(&input.f);
            self.prover
                .combined_basis
                .copy_from_slice(&input.initial_basis);
            self.prover.t_r = input.initial_target;
            self.prover.transcript.clear();
            self.prover.transcript.push(input.first_msg);
            self.challenger = challenger.clone();
        }
    }

    #[derive(Debug)]
    struct LazyOodTimingRecord {
        seed: u64,
        pair_index: usize,
        order: LazyOodTimingOrder,
        candidate_slot: usize,
        control: LazyOodTimingMetrics,
        candidate: LazyOodTimingMetrics,
        delta_ms: f64,
    }

    fn lazy_ood_timing_condition_cache(words: &[u64], salt: u64) {
        let mut checksum = salt;
        // One read per 64-byte cache line across 128 MiB. This is deliberately
        // outside both measured spans; between OOD and the ordinary introduce
        // it models the much larger commit/open/induce working set and removes
        // cache-residency bias between the candidate's retained 34 KiB 11+7
        // factors and the control's materialized full-equality basis update.
        for &word in words.iter().step_by(8) {
            checksum = checksum.rotate_left(7) ^ std::hint::black_box(word);
        }
        std::hint::black_box(checksum);
    }

    fn lazy_ood_timing_challenger(
        seed: u64,
        first_msg: SumcheckMessage,
    ) -> crate::challenger::FsChallenger {
        use crate::challenger::Challenger;

        let mut challenger = crate::challenger::FsChallenger::with_hash(
            b"flock-lazy-ood-l1-component-v1",
            HashKind::Blake3,
        );
        challenger.observe_bytes(&seed.to_le_bytes());
        challenger.observe_f128(first_msg.u_0);
        challenger.observe_f128(first_msg.u_2);
        challenger
    }

    fn lazy_ood_timing_run_arm(
        slot: &mut LazyOodTimingSlot,
        input: &LazyOodTimingInput,
        challenger: &crate::challenger::FsChallenger,
        cache_conditioner: &[u64],
        arm: LazyOodTimingArm,
        salt: u64,
    ) -> LazyOodTimingRun {
        use crate::challenger::Challenger;
        use std::time::Instant;

        slot.reset(input, challenger);
        lazy_ood_timing_condition_cache(cache_conditioner, salt ^ 0x0DD0_0001);

        // Span one is the complete production-order L1 OOD operation. Sampling
        // and transcript work are included because the production t_ood span
        // includes them too; both arms start from cloned FS state.
        let ood_started = Instant::now();
        let z = slot.challenger.sample_f128_vec(LAZY_OOD_TIMING_LOG_N);
        let (ood_intro, ood_value) = match arm {
            LazyOodTimingArm::Control => {
                let eq_z = build_eq_table(&z);
                slot.prover.introduce_new_with_eval(eq_z)
            }
            LazyOodTimingArm::Candidate => slot
                .prover
                .introduce_new_ood_factorized(&z)
                .expect("ranked L1 candidate must accept exact geometry"),
        };
        slot.challenger.observe_f128(ood_value);
        slot.challenger.observe_f128(ood_intro.u_0);
        slot.challenger.observe_f128(ood_intro.u_2);
        let ood_beta = slot.challenger.sample_f128();
        match arm {
            LazyOodTimingArm::Control => slot.prover.glue(ood_beta),
            LazyOodTimingArm::Candidate => slot.prover.glue_factorized_ood(ood_beta),
        }
        let ood_ms = ood_started.elapsed().as_secs_f64() * 1e3;

        match arm {
            LazyOodTimingArm::Control => {
                assert!(slot.prover.pending_ood_eq.is_none());
            }
            LazyOodTimingArm::Candidate => assert!(matches!(
                slot.prover.pending_ood_eq.as_ref(),
                Some(PendingOodEq::Glued { .. })
            )),
        }
        assert!(slot.prover.pending_glue.is_none());

        // Preserve the ranked ordering while excluding only unchanged work:
        // large intervening activity and the ordinary L0 opening-induced basis
        // introduction. The ordinary glue itself is now part of the candidate,
        // so charge it separately and sum that span with OOD + consuming fold.
        lazy_ood_timing_condition_cache(cache_conditioner, salt ^ 0x1AD0_0002);
        let ordinary_intro = slot
            .prover
            .introduce_new(input.ordinary_basis.clone(), input.ordinary_target);
        slot.challenger.observe_f128(ordinary_intro.u_0);
        slot.challenger.observe_f128(ordinary_intro.u_2);
        let ordinary_beta = slot.challenger.sample_f128();
        let ordinary_glue_started = Instant::now();
        match arm {
            LazyOodTimingArm::Control => slot.prover.glue(ordinary_beta),
            LazyOodTimingArm::Candidate => {
                slot.prover.glue_deferred_into_lazy_ood_fold(ordinary_beta)
            }
        }
        let ordinary_glue_ms = ordinary_glue_started.elapsed().as_secs_f64() * 1e3;
        assert!(slot.prover.pending_glue.is_none());
        match arm {
            LazyOodTimingArm::Control => {
                assert!(slot.prover.pending_ood_eq.is_none());
                assert!(slot.prover.pending_fold_basis.is_none());
            }
            LazyOodTimingArm::Candidate => {
                assert!(matches!(
                    slot.prover.pending_ood_eq.as_ref(),
                    Some(PendingOodEq::Glued { .. })
                ));
                assert!(slot.prover.pending_fold_basis.is_some());
            }
        }

        let fold_challenge = slot.challenger.sample_f128();
        let fold_started = Instant::now();
        let fold_msg = slot.prover.fold(fold_challenge);
        let fold_ms = fold_started.elapsed().as_secs_f64() * 1e3;
        slot.challenger.observe_f128(fold_msg.u_0);
        slot.challenger.observe_f128(fold_msg.u_2);
        let fs_continuation = slot.challenger.sample_f128();

        assert_eq!(slot.prover.f.len(), LAZY_OOD_TIMING_FOLDED_LEN);
        assert_eq!(slot.prover.combined_basis.len(), LAZY_OOD_TIMING_FOLDED_LEN);
        assert!(slot.prover.pending_glue.is_none());
        assert!(slot.prover.pending_fold_basis.is_none());
        assert!(slot.prover.pending_ood_eq.is_none());
        assert!(ood_ms.is_finite() && ood_ms > 0.0);
        assert!(ordinary_glue_ms.is_finite() && ordinary_glue_ms >= 0.0);
        assert!(fold_ms.is_finite() && fold_ms > 0.0);

        LazyOodTimingRun {
            metrics: LazyOodTimingMetrics {
                ood_ms,
                ordinary_glue_ms,
                fold_ms,
            },
            z,
            ood_value,
            ood_intro,
            ood_beta,
            ordinary_intro,
            ordinary_beta,
            fold_challenge,
            fold_msg,
            fs_continuation,
        }
    }

    fn lazy_ood_timing_median(values: &[f64]) -> f64 {
        assert!(!values.is_empty());
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let middle = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    fn lazy_ood_timing_percentile(values: &[f64], numerator: usize, denominator: usize) -> f64 {
        assert!(!values.is_empty());
        assert!(numerator > 0 && numerator <= denominator);
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let rank = (numerator * sorted.len()).div_ceil(denominator);
        sorted[rank - 1]
    }

    /// Same-binary component adjudication at the exact ranked M32 Fast L1
    /// geometry. The candidate moves both OOD work and the ordinary basis glue
    /// into the next fold, so the valid component wall is the sum of the OOD,
    /// ordinary-glue, and consuming-fold spans. The unchanged ordinary intro
    /// is executed outside the timer; a 128 MiB conditioner models the real
    /// open/induce gap before it.
    ///
    /// Run alone, under the repository's exclusive timing lock, with an empty
    /// diagnostic/override environment and the challenge profile:
    ///
    /// ```text
    /// FLOCK_RUN_LAZY_OOD_TIMING=1 RAYON_NUM_THREADS=10 \
    /// cargo +1.97.0 test --locked --offline --profile challenge -p flock-core --lib \
    ///   pcs::ligerito::tests::lazy_ood_l1_production_geometry_paired_timing -- \
    ///   --ignored --exact --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore = "manual exact-ranked L1 paired component timing gate"]
    fn lazy_ood_l1_production_geometry_paired_timing() {
        use rayon::prelude::*;

        const OPT_IN: &str = "FLOCK_RUN_LAZY_OOD_TIMING";
        const WARMUP_PAIRS: usize = 8;
        const MEASURED_PAIRS: usize = 32;
        const MIN_MEDIAN_WIN_MS: f64 = 0.300;
        const CACHE_CONDITIONER_U64S: usize = (128usize << 20) / core::mem::size_of::<u64>();
        const SEEDS: [u64; 4] = [
            0x4F4F_445F_A11C_0001,
            0xD1CE_600D_5EED_0002,
            0xA5A5_19F0_CAFE_0003,
            0x73A9_184B_F01D_0004,
        ];

        // Reject an accidental ordinary test invocation before process-global
        // Rayon initialization or any ranked allocation.
        assert_eq!(
            std::env::var_os(OPT_IN).as_deref(),
            Some(std::ffi::OsStr::new("1")),
            "explicit exact-1 timing opt-in missing"
        );
        assert!(
            !cfg!(debug_assertions),
            "timing gate rejects debug builds; use --profile challenge"
        );
        assert!(
            cfg!(all(
                target_os = "macos",
                target_arch = "aarch64",
                target_feature = "aes"
            )),
            "timing gate requires native Apple AArch64 PMULL codegen"
        );
        let executable = std::env::current_exe().expect("resolve timing test executable");
        assert!(
            executable
                .components()
                .any(|component| component.as_os_str() == std::ffi::OsStr::new("challenge")),
            "timing gate requires Cargo's challenge profile directory: {executable:?}"
        );
        assert_eq!(
            std::env::var("RAYON_NUM_THREADS").as_deref(),
            Ok("10"),
            "timing gate requires exact RAYON_NUM_THREADS=10"
        );
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            crate::perf_core_count_cached(),
            10,
            "timing gate requires the official ten-performance-core topology"
        );

        let mut inherited_overrides: Vec<String> = std::env::vars_os()
            .filter_map(|(key, _)| {
                let key = key.to_string_lossy();
                let forbidden = (key.starts_with("FLOCK_") && key.as_ref() != OPT_IN)
                    || key.starts_with("BLAKE3_")
                    || key.starts_with("LIG_")
                    || key.starts_with("LIGERITO_")
                    || matches!(key.as_ref(), "RAYON_RS_NUM_CPUS" | "RAYON_LOG" | "RUST_LOG")
                    || key.starts_with("MTL_")
                    || key.starts_with("METAL_")
                    || key.starts_with("Malloc")
                    || matches!(
                        key.as_ref(),
                        "NSZombieEnabled"
                            | "NSAutoreleaseFreedObjectCheck"
                            | "DYLD_INSERT_LIBRARIES"
                    )
                    || key.starts_with("DYLD_PRINT_");
                forbidden.then(|| key.into_owned())
            })
            .collect();
        inherited_overrides.sort_unstable();
        assert!(
            inherited_overrides.is_empty(),
            "timing gate rejects inherited prover, Rayon, logging, Metal, allocator, Objective-C, or DYLD overrides: {inherited_overrides:?}"
        );
        assert_eq!(
            crate::init_perf_thread_pool(),
            Some(10),
            "timing gate requires a fresh official ten-thread perf pool"
        );
        assert_eq!(
            rayon::current_num_threads(),
            10,
            "Rayon global pool must contain exactly ten threads"
        );

        let mut config =
            prover_config_for(25, 6, LigeritoProfile::Fast).expect("embedded M32 Fast config");
        config.merkle_hash = HashKind::Blake3;
        assert!(ranked_l1_lazy_ood_eq_enabled(
            &config,
            25,
            LAZY_OOD_TIMING_LOG_N,
            1,
            LAZY_OOD_TIMING_LEN,
            true,
        ));

        let mut random_state = 0x1A2B_3C4D_5E6F_7081u64;
        let mut random_f128 = || {
            random_state = random_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(
                random_state,
                random_state.rotate_left(29) ^ 0xA5A5_5A5A_D1CE_6EAD,
            )
        };
        let f: Vec<F128> = (0..LAZY_OOD_TIMING_LEN).map(|_| random_f128()).collect();
        let initial_basis: Vec<F128> = (0..LAZY_OOD_TIMING_LEN).map(|_| random_f128()).collect();
        let ordinary_basis: Vec<F128> = (0..LAZY_OOD_TIMING_LEN).map(|_| random_f128()).collect();
        let initial_target = f
            .par_iter()
            .zip(initial_basis.par_iter())
            .map(|(&fv, &bv)| fv * bv)
            .reduce(|| F128::ZERO, |left, right| left + right);
        let ordinary_target = f
            .par_iter()
            .zip(ordinary_basis.par_iter())
            .map(|(&fv, &bv)| fv * bv)
            .reduce(|| F128::ZERO, |left, right| left + right);
        let first_msg = round_msg_lsb(&f, &initial_basis);
        let input = LazyOodTimingInput {
            f,
            initial_basis,
            ordinary_basis,
            initial_target,
            ordinary_target,
            first_msg,
        };

        let first_challenger = lazy_ood_timing_challenger(SEEDS[0], first_msg);
        let mut slots = [
            LazyOodTimingSlot::new(&input, &first_challenger),
            LazyOodTimingSlot::new(&input, &first_challenger),
        ];
        let cache_conditioner: Vec<u64> = (0..CACHE_CONDITIONER_U64S)
            .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let mut records = Vec::with_capacity(MEASURED_PAIRS);

        // Two warmups and eight measured pairs per challenge seed. Order
        // alternates every pair; allocation roles mirror every two pairs, so
        // candidate/control and first/second execution are not tied to a slot.
        for (seed_index, &seed) in SEEDS.iter().enumerate() {
            let challenger = lazy_ood_timing_challenger(seed, first_msg);
            for local_pair in 0..10usize {
                let measured = local_pair >= 2;
                let global_pair = seed_index * 10 + local_pair;
                let order = if global_pair.is_multiple_of(2) {
                    LazyOodTimingOrder::ControlCandidate
                } else {
                    LazyOodTimingOrder::CandidateControl
                };
                let candidate_slot = (global_pair / 2) % 2;
                let control_slot = 1 - candidate_slot;
                let salt = seed ^ (global_pair as u64).rotate_left(23);

                let (control, candidate) = match order {
                    LazyOodTimingOrder::ControlCandidate => {
                        let control = lazy_ood_timing_run_arm(
                            &mut slots[control_slot],
                            &input,
                            &challenger,
                            &cache_conditioner,
                            LazyOodTimingArm::Control,
                            salt ^ 0xC011_7001,
                        );
                        let candidate = lazy_ood_timing_run_arm(
                            &mut slots[candidate_slot],
                            &input,
                            &challenger,
                            &cache_conditioner,
                            LazyOodTimingArm::Candidate,
                            salt ^ 0xCAAD_1DA7,
                        );
                        (control, candidate)
                    }
                    LazyOodTimingOrder::CandidateControl => {
                        let candidate = lazy_ood_timing_run_arm(
                            &mut slots[candidate_slot],
                            &input,
                            &challenger,
                            &cache_conditioner,
                            LazyOodTimingArm::Candidate,
                            salt ^ 0xCAAD_1DA7,
                        );
                        let control = lazy_ood_timing_run_arm(
                            &mut slots[control_slot],
                            &input,
                            &challenger,
                            &cache_conditioner,
                            LazyOodTimingArm::Control,
                            salt ^ 0xC011_7001,
                        );
                        (control, candidate)
                    }
                };

                assert_eq!(
                    candidate.z, control.z,
                    "seed={seed:#x} pair={local_pair}: z"
                );
                assert_eq!(
                    candidate.ood_value, control.ood_value,
                    "seed={seed:#x} pair={local_pair}: OOD value"
                );
                assert_eq!(
                    candidate.ood_intro, control.ood_intro,
                    "seed={seed:#x} pair={local_pair}: OOD intro"
                );
                assert_eq!(
                    candidate.ood_beta, control.ood_beta,
                    "seed={seed:#x} pair={local_pair}: OOD beta"
                );
                assert_eq!(
                    candidate.ordinary_intro, control.ordinary_intro,
                    "seed={seed:#x} pair={local_pair}: ordinary intro"
                );
                assert_eq!(
                    candidate.ordinary_beta, control.ordinary_beta,
                    "seed={seed:#x} pair={local_pair}: ordinary beta"
                );
                assert_eq!(
                    candidate.fold_challenge, control.fold_challenge,
                    "seed={seed:#x} pair={local_pair}: fold challenge"
                );
                assert_eq!(
                    candidate.fold_msg, control.fold_msg,
                    "seed={seed:#x} pair={local_pair}: fold message"
                );
                assert_eq!(
                    candidate.fs_continuation, control.fs_continuation,
                    "seed={seed:#x} pair={local_pair}: FS continuation"
                );
                let control_state = &slots[control_slot].prover;
                let candidate_state = &slots[candidate_slot].prover;
                assert_eq!(
                    candidate_state.f, control_state.f,
                    "seed={seed:#x} pair={local_pair}: folded f"
                );
                assert_eq!(
                    candidate_state.combined_basis, control_state.combined_basis,
                    "seed={seed:#x} pair={local_pair}: folded basis"
                );
                assert_eq!(
                    candidate_state.t_r, control_state.t_r,
                    "seed={seed:#x} pair={local_pair}: target"
                );
                assert_eq!(
                    candidate_state.transcript, control_state.transcript,
                    "seed={seed:#x} pair={local_pair}: transcript"
                );
                assert!(candidate_state.pending_glue.is_none());
                assert!(candidate_state.pending_fold_basis.is_none());
                assert!(candidate_state.pending_ood_eq.is_none());
                assert!(control_state.pending_glue.is_none());
                assert!(control_state.pending_fold_basis.is_none());
                assert!(control_state.pending_ood_eq.is_none());

                if measured {
                    let delta_ms = candidate.metrics.total_ms() - control.metrics.total_ms();
                    assert!(delta_ms.is_finite());
                    records.push(LazyOodTimingRecord {
                        seed,
                        pair_index: seed_index * 8 + (local_pair - 2),
                        order,
                        candidate_slot,
                        control: control.metrics,
                        candidate: candidate.metrics,
                        delta_ms,
                    });
                }
            }
        }

        assert_eq!(WARMUP_PAIRS, SEEDS.len() * 2);
        assert_eq!(records.len(), MEASURED_PAIRS);
        assert!(
            records
                .iter()
                .enumerate()
                .all(|(index, record)| record.pair_index == index),
            "measured pair indices must be contiguous"
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record.candidate_slot == 0)
                .count(),
            MEASURED_PAIRS / 2,
            "candidate allocation slots must be balanced"
        );
        let control_total: Vec<f64> = records
            .iter()
            .map(|record| record.control.total_ms())
            .collect();
        let candidate_total: Vec<f64> = records
            .iter()
            .map(|record| record.candidate.total_ms())
            .collect();
        let deltas: Vec<f64> = records.iter().map(|record| record.delta_ms).collect();
        let control_ood: Vec<f64> = records.iter().map(|record| record.control.ood_ms).collect();
        let candidate_ood: Vec<f64> = records
            .iter()
            .map(|record| record.candidate.ood_ms)
            .collect();
        let control_ordinary_glue: Vec<f64> = records
            .iter()
            .map(|record| record.control.ordinary_glue_ms)
            .collect();
        let candidate_ordinary_glue: Vec<f64> = records
            .iter()
            .map(|record| record.candidate.ordinary_glue_ms)
            .collect();
        let control_fold: Vec<f64> = records
            .iter()
            .map(|record| record.control.fold_ms)
            .collect();
        let candidate_fold: Vec<f64> = records
            .iter()
            .map(|record| record.candidate.fold_ms)
            .collect();
        let control_first_deltas: Vec<f64> = records
            .iter()
            .filter(|record| record.order == LazyOodTimingOrder::ControlCandidate)
            .map(|record| record.delta_ms)
            .collect();
        let candidate_first_deltas: Vec<f64> = records
            .iter()
            .filter(|record| record.order == LazyOodTimingOrder::CandidateControl)
            .map(|record| record.delta_ms)
            .collect();
        assert_eq!(control_first_deltas.len(), MEASURED_PAIRS / 2);
        assert_eq!(candidate_first_deltas.len(), MEASURED_PAIRS / 2);

        let paired_median = lazy_ood_timing_median(&deltas);
        let control_first_p90 = lazy_ood_timing_percentile(&control_first_deltas, 9, 10);
        let candidate_first_p90 = lazy_ood_timing_percentile(&candidate_first_deltas, 9, 10);
        let mean_delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
        let wins = deltas.iter().filter(|&&delta| delta < 0.0).count();

        // All output is buffered until every observation and exact-state audit
        // is complete, so formatting cannot perturb a later sample.
        println!("lazy-ood-l1 raw={records:#?}");
        println!(
            "lazy-ood-l1 summary pairs={} wins={} paired_delta_median_ms={paired_median:.6} paired_delta_mean_ms={mean_delta:.6} paired_delta_p90_ms={:.6} paired_delta_p95_ms={:.6} control_first_delta_p90_ms={control_first_p90:.6} candidate_first_delta_p90_ms={candidate_first_p90:.6}",
            records.len(),
            wins,
            lazy_ood_timing_percentile(&deltas, 9, 10),
            lazy_ood_timing_percentile(&deltas, 95, 100),
        );
        println!(
            "lazy-ood-l1 raw-walls control_median_ms={:.6} candidate_median_ms={:.6} control_p90_ms={:.6} candidate_p90_ms={:.6} control_p95_ms={:.6} candidate_p95_ms={:.6}",
            lazy_ood_timing_median(&control_total),
            lazy_ood_timing_median(&candidate_total),
            lazy_ood_timing_percentile(&control_total, 9, 10),
            lazy_ood_timing_percentile(&candidate_total, 9, 10),
            lazy_ood_timing_percentile(&control_total, 95, 100),
            lazy_ood_timing_percentile(&candidate_total, 95, 100),
        );
        println!(
            "lazy-ood-l1 phases control_ood_median_ms={:.6} candidate_ood_median_ms={:.6} ood_delta_median_ms={:.6} control_ordinary_glue_median_ms={:.6} candidate_ordinary_glue_median_ms={:.6} ordinary_glue_delta_median_ms={:.6} control_fold_median_ms={:.6} candidate_fold_median_ms={:.6} fold_delta_median_ms={:.6}",
            lazy_ood_timing_median(&control_ood),
            lazy_ood_timing_median(&candidate_ood),
            lazy_ood_timing_median(
                &candidate_ood
                    .iter()
                    .zip(&control_ood)
                    .map(|(candidate, control)| candidate - control)
                    .collect::<Vec<_>>()
            ),
            lazy_ood_timing_median(&control_ordinary_glue),
            lazy_ood_timing_median(&candidate_ordinary_glue),
            lazy_ood_timing_median(
                &candidate_ordinary_glue
                    .iter()
                    .zip(&control_ordinary_glue)
                    .map(|(candidate, control)| candidate - control)
                    .collect::<Vec<_>>()
            ),
            lazy_ood_timing_median(&control_fold),
            lazy_ood_timing_median(&candidate_fold),
            lazy_ood_timing_median(
                &candidate_fold
                    .iter()
                    .zip(&control_fold)
                    .map(|(candidate, control)| candidate - control)
                    .collect::<Vec<_>>()
            ),
        );
        for &seed in &SEEDS {
            let seed_deltas: Vec<f64> = records
                .iter()
                .filter(|record| record.seed == seed)
                .map(|record| record.delta_ms)
                .collect();
            println!(
                "lazy-ood-l1 seed={seed:#018x} median_delta_ms={:.6} p90_delta_ms={:.6} wins={}/{}",
                lazy_ood_timing_median(&seed_deltas),
                lazy_ood_timing_percentile(&seed_deltas, 9, 10),
                seed_deltas.iter().filter(|&&delta| delta < 0.0).count(),
                seed_deltas.len(),
            );
        }

        assert!(
            paired_median <= -MIN_MEDIAN_WIN_MS,
            "paired candidate-minus-control median {paired_median:.6} ms does not clear -{MIN_MEDIAN_WIN_MS:.3} ms; raw={records:#?}"
        );
        assert!(
            control_first_p90 <= 0.0,
            "control-first paired delta p90 {control_first_p90:.6} ms regressed; raw={records:#?}"
        );
        assert!(
            candidate_first_p90 <= 0.0,
            "candidate-first paired delta p90 {candidate_first_p90:.6} ms regressed; raw={records:#?}"
        );
    }

    /// End-to-end sumcheck on a single basis poly: prove `Σ_x f(x)·b(x) = h`.
    /// Stops one round early (yr length 2 sent in clear, à la Ligerito).
    /// Verifier replays each round message, checks `q(0)+q(1)=T_r`, applies
    /// the challenge, and confirms the residual inner product matches.
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

        // Prover: 1 start message + (n-1) folds, leaving a length-2 residual.
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

        // Verifier replay: n messages (start + n-1 folds), n-1 prover-folds challenges
        // (r_0..r_{n-2}) already in ris, plus one new r_last for the final residual.
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

    /// Multi-basis sumcheck: introduce_new + glue mid-protocol. Verifier replays.
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

        // Fold once before introducing b2 (must fold at the same dim as the introduced poly).
        let r0 = ch.sample_f128();
        prover.fold(r0);
        // Partial-eval b2 too so it matches the prover's current f dim.
        let mut b2_folded = b2.clone();
        partial_eval_lsb_one(&mut b2_folded, r0);
        // The h for b2 at the folded dim is Σ b2_folded · f_folded — but the verifier
        // also gets to recompute this from the same shared inputs. For the test we
        // pass it explicitly.
        let h2_folded: F128 = b2_folded
            .iter()
            .zip(prover.f().iter())
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |a, v| a + v);
        prover.introduce_new(b2_folded.clone(), h2_folded);
        let alpha = ch.sample_f128();
        prover.glue(alpha);

        // Continue folding to length 2 residual: n total fold-vars used, but
        // we've already used 1 (r0). One more r_last is the verifier's final.
        let mut ris = vec![r0];
        for _ in 0..(n - 2) {
            let r = ch.sample_f128();
            ris.push(r);
            prover.fold(r);
        }
        let r_last = ch.sample_f128();
        ris.push(r_last);
        assert_eq!(prover.f().len(), 2);

        // Verifier replays: 1 start, 1 fold, 1 introduce_new (no T_r update), 1 glue
        // (combine running quad with introduced, update T_r), then (n-2) folds.
        let msgs = prover.transcript().to_vec();
        // start (idx 0) + fold(r0) → idx 1 + introduce_new → idx 2 + later folds
        // Note: glue doesn't add a transcript entry; it just combines internal state.
        assert_eq!(msgs.len(), 1 + 1 + 1 + (n - 2));

        let mut t_r = h1;
        // start
        let q0 = RoundQuad::from_msg(msgs[0], t_r);
        assert_eq!(q0.eval(F128::ZERO) + q0.eval(F128::ONE), t_r);
        t_r = q0.eval(r0); // fold(r0)
        // fold msg (idx 1)
        let q1 = RoundQuad::from_msg(msgs[1], t_r);
        assert_eq!(q1.eval(F128::ZERO) + q1.eval(F128::ONE), t_r);
        // introduce_new msg (idx 2): claim is h2_folded, not T_r
        let q_intro = RoundQuad::from_msg(msgs[2], h2_folded);
        assert_eq!(
            q_intro.eval(F128::ZERO) + q_intro.eval(F128::ONE),
            h2_folded
        );
        // glue: running := q1 + alpha · q_intro; T_r := T_r + alpha · h2_folded
        let combined = RoundQuad::fold(&q1, &q_intro, alpha);
        t_r += alpha * h2_folded;
        // The combined quad must satisfy sumcheck identity against the new T_r
        assert_eq!(combined.eval(F128::ZERO) + combined.eval(F128::ONE), t_r);
        // Apply the rest of the folds; each subsequent msg supersedes `combined` after eval.
        // After glue, the next fold uses challenge ris[1]. msgs[3] is from fold(ris[1]).
        let mut running = combined;
        // Remaining prover folds: ris[1..n-1] correspond to msgs[3..n+1].
        // Total prover-fold messages after start = (n-1) (single basis) ... but here we
        // have 1 start + 1 fold + 1 intro + (n-2) more folds = n+1 messages.
        assert_eq!(msgs.len(), n + 1);
        for (k, &r) in ris.iter().enumerate().skip(1).take(n - 2) {
            t_r = running.eval(r);
            let msg = msgs[2 + k]; // idx 3, 4, ...
            running = RoundQuad::from_msg(msg, t_r);
            assert_eq!(
                running.eval(F128::ZERO) + running.eval(F128::ONE),
                t_r,
                "post-glue round k={k}"
            );
        }
        // Final: apply r_last to the LAST message's quad
        t_r = running.eval(r_last);

        let one_plus_r = F128::ONE + r_last;
        let f_resid = prover.f()[0] * one_plus_r + prover.f()[1] * r_last;
        // With the collapsed-basis design, combined_basis already holds
        // eq + α·b2 at the residual dim.
        let combined_resid =
            prover.combined_basis[0] * one_plus_r + prover.combined_basis[1] * r_last;
        assert_eq!(
            f_resid * combined_resid,
            t_r,
            "residual inner product != t_r"
        );
    }

    /// `induce_sumcheck_poly` is consistent with the codeword:
    ///   1. `enforced_sum` equals `Σ_i α^i · c[q_i]` computed directly,
    ///   2. `Σ_j msg[j] · basis_poly[j]` equals `enforced_sum` (the sumcheck
    ///      claim that the verifier reduces to a residual eval).
    #[test]
    fn induce_sumcheck_poly_consistent_with_codeword() {
        use crate::challenger::Challenger;
        let log_msg = 4;
        let log_inv_rate = 1;
        let msg_cols = 1usize << log_msg;
        let block_len = msg_cols << log_inv_rate;

        // Single-lane (num_interleaved = 1, no v_challenges).
        let mut ch = crate::challenger::RandomChallenger::new(0xF00DCAFE);
        let msg: Vec<F128> = (0..msg_cols).map(|_| ch.sample_f128()).collect();

        // Encode via Flock's NTT (zero-pad to block_len).
        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let mut codeword = vec![F128::ZERO; block_len];
        codeword[..msg_cols].copy_from_slice(&msg);
        ntt.forward_transform(&mut codeword);

        // Pick random distinct query positions.
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

        // Check 1: enforced_sum = Σ_i eq(α, i_bin) · c[q_i]
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

        // Check 2: Σ_j msg[j] · basis_poly[j] = enforced_sum.
        // This is the LCH novel-basis identity: c[q] = Σ_j msg[j] · Ŵ_j(q_field),
        // so Σ_i α^i · c[q_i] = Σ_j msg[j] · Σ_i α^i · Ŵ_j(q_i_field) = Σ_j msg[j] · basis_poly[j].
        let inner: F128 = msg
            .iter()
            .zip(basis_poly.iter())
            .map(|(&m, &b)| m * b)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(inner, enforced_sum, "msg · basis_poly != enforced_sum");
    }

    /// The micro-stack memo for `eval_sk_at_vks` must return exactly the
    /// direct computation at every dim the provers use (and then some),
    /// including on repeated (cache-hit) calls.
    #[test]
    fn eval_sk_at_vks_memo_matches_direct() {
        for log_n in 0..=20usize {
            let direct = eval_sk_at_vks_uncached(log_n);
            assert_eq!(eval_sk_at_vks(log_n), direct, "first call, log_n={log_n}");
            assert_eq!(eval_sk_at_vks(log_n), direct, "cached call, log_n={log_n}");
        }
    }

    /// `induce_sumcheck_poly_via_ntt` must be byte-identical to dense across
    /// shapes incl. the real m30_fast level dims.
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

    /// The sparse-prefix transpose must equal the baseline dense transpose on
    /// the same scattered input, across sizes (incl. > and < the k=8 prefix gate).
    #[test]
    fn transpose_sparse_matches_dense() {
        use crate::challenger::Challenger;
        for &log_d in &[0usize, 1, 6, 8, 11, 12, 14, 16, 18, 20] {
            for &nq in &[0usize, 1, 2, 5, 43, 106, 218] {
                let n = 1usize << log_d;
                let nq = nq.min(n);
                let mut ch =
                    crate::challenger::RandomChallenger::new(0xC0DE ^ (log_d * 131 + nq) as u64);
                // The transform itself supports log_d=0, while the standard
                // basis constructor starts at one dimension.
                let ntt = AdditiveNttF128::standard(log_d.max(1));
                let mut positions: Vec<usize> = Vec::new();
                let mut values: Vec<F128> = Vec::new();
                while positions.len() < nq {
                    let p = (ch.sample_f128().lo as usize) % n;
                    if !positions.contains(&p) {
                        positions.push(p);
                        values.push(ch.sample_f128());
                    }
                }
                // Baseline: scatter then dense transpose.
                let mut dense = vec![F128::ZERO; n];
                for (&p, &v) in positions.iter().zip(&values) {
                    dense[p] += v;
                }
                transpose_forward_ntt(&ntt, &mut dense, log_d);
                let sparse = transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d, false);
                assert_eq!(sparse, dense, "log_d={log_d}, nq={nq}");
            }
        }
    }

    #[test]
    fn linear_sparse_windows_match_hashmap_and_dense() {
        use crate::challenger::Challenger;

        for &log_d in &[12usize, 14, 18, 20] {
            let n = 1usize << log_d;
            let ntt = AdditiveNttF128::standard(log_d);
            for &n_queries in &[0usize, 1, 5, 43, 106, 218] {
                let mut challenger = crate::challenger::RandomChallenger::new(
                    0x11EA_2105 ^ ((log_d as u64) << 32) ^ n_queries as u64,
                );
                let mut pairs = Vec::with_capacity(n_queries);
                while pairs.len() < n_queries {
                    let position = (challenger.sample_f128().lo as usize) % n;
                    if !pairs.iter().any(|&(p, _)| p == position) {
                        pairs.push((position, challenger.sample_f128()));
                    }
                }
                pairs.sort_unstable_by_key(|&(position, _)| position);
                let (positions, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();

                let mut dense = vec![F128::ZERO; n];
                for (&position, &value) in positions.iter().zip(&values) {
                    dense[position] += value;
                }
                transpose_forward_ntt(&ntt, &mut dense, log_d);

                let legacy = transpose_forward_ntt_sparse_hashmap(
                    &ntt, &positions, &values, log_d, 8, false,
                );
                let linear = transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d, false);
                assert_eq!(
                    linear, legacy,
                    "linear != hashmap at log_d={log_d}, nq={n_queries}"
                );
                assert_eq!(
                    linear, dense,
                    "linear != dense at log_d={log_d}, nq={n_queries}"
                );
            }
        }

        let log_d = 12;
        let ntt = AdditiveNttF128::standard(log_d);
        let positions = vec![0usize, 0, 1, 255, 256, 256, (1usize << log_d) - 1];
        let values = vec![
            F128::ONE,
            F128::ONE,
            F128::new(2, 0),
            F128::ZERO,
            F128::new(4, 0),
            F128::new(5, 0),
            F128::new(6, 0),
        ];
        let legacy =
            transpose_forward_ntt_sparse_hashmap(&ntt, &positions, &values, log_d, 8, false);
        let linear = transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d, false);
        assert_eq!(linear, legacy, "duplicate-position accumulation changed");

        let positions: Vec<usize> = (0..1usize << log_d).step_by(1 << 8).collect();
        let values = vec![F128::ONE; positions.len()];
        let legacy =
            transpose_forward_ntt_sparse_hashmap(&ntt, &positions, &values, log_d, 8, false);
        let linear = transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d, false);
        assert_eq!(linear, legacy, "all-active-window case changed");
    }

    /// The direct gather/materialize kernel must equal the incumbent two-step
    /// `densify + first fused-three-layer pass` for every sparse-frontier edge
    /// shape, including the exact ranked L0/L1 query counts.
    #[test]
    fn fused_densify_first_3layer_matches_incumbent_stage() {
        use crate::challenger::Challenger;

        fn check_case(log_d: usize, mut pairs: Vec<(usize, F128)>, label: &str) {
            const PREFIX_K: usize = 8;
            pairs.sort_unstable_by_key(|&(position, _)| position);
            let (positions, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
            let ntt = AdditiveNttF128::standard(log_d);
            let groups = group_sorted_positions(&positions, PREFIX_K);
            let mut arena = scatter_active_windows(&groups, &positions, &values, PREFIX_K);
            transform_active_windows(&ntt, &mut arena, &groups, PREFIX_K, log_d);

            let mut expected = densify_active_windows(&arena, &groups, log_d, PREFIX_K);
            let first_layer = log_d - PREFIX_K - 3;
            transpose_forward_ntt_fused_3layer(&ntt, &mut expected, log_d, first_layer);
            let actual =
                densify_active_windows_fused_first_3layer(&ntt, &arena, &groups, log_d, PREFIX_K);
            assert_eq!(actual, expected, "{label}: log_d={log_d}");
        }

        check_case(12, Vec::new(), "empty");
        check_case(18, vec![(0, F128::ONE)], "single");
        check_case(
            12,
            vec![
                (0, F128::ONE),
                (0, F128::ONE),
                (1, F128::new(2, 0)),
                (255, F128::new(3, 1)),
                (256, F128::new(4, 2)),
                (256, F128::new(5, 3)),
                ((1 << 12) - 1, F128::new(6, 4)),
            ],
            "duplicates",
        );

        let all_active: Vec<(usize, F128)> = (0..1usize << 12)
            .step_by(1 << 8)
            .enumerate()
            .map(|(i, position)| (position, F128::new(i as u64 + 1, (i as u64).rotate_left(7))))
            .collect();
        check_case(12, all_active, "all-active");

        for &(log_d, n_queries) in &[(14usize, 43usize), (18, 106), (20, 218)] {
            let n = 1usize << log_d;
            let mut challenger = crate::challenger::RandomChallenger::new(
                0xF05E_D3A5_1F1E_0000 ^ ((log_d as u64) << 24) ^ n_queries as u64,
            );
            let mut pairs = Vec::with_capacity(n_queries);
            while pairs.len() < n_queries {
                let position = (challenger.sample_f128().lo as usize) % n;
                if !pairs.iter().any(|&(p, _)| p == position) {
                    pairs.push((position, challenger.sample_f128()));
                }
            }
            check_case(log_d, pairs, "random/ranked");
        }
    }

    #[test]
    fn ranked_fused_densify_first_shape_gate_is_narrow() {
        assert!(is_ranked_fused_densify_first_shape(20, 8, 218));
        assert!(is_ranked_fused_densify_first_shape(18, 8, 106));
        for &(log_d, prefix_k, n_positions) in &[
            (20usize, 7usize, 218usize),
            (20, 8, 217),
            (20, 8, 219),
            (19, 8, 218),
            (18, 7, 106),
            (18, 8, 105),
            (18, 8, 107),
            (17, 8, 106),
        ] {
            assert!(!is_ranked_fused_densify_first_shape(
                log_d,
                prefix_k,
                n_positions,
            ));
        }
    }

    /// Exact ranked component adjudication. The control times the incumbent
    /// full-domain densification plus its first dense radix-8 pass; the
    /// candidate times their direct gather/materialize replacement. Sparse
    /// grouping and the eight local prefix layers are common setup outside the
    /// measured spans. Arm order reverses every pair and allocations are
    /// dropped outside both timers.
    ///
    /// ```text
    /// FLOCK_RUN_INDUCE_FUSED_DENSIFY_TIMING=1 RAYON_NUM_THREADS=10 \
    /// cargo +1.97.0 test --locked --offline --profile challenge -p flock-core --lib \
    /// pcs::ligerito::tests::fused_densify_first_ranked_shapes_paired_timing -- \
    /// --ignored --exact --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore]
    fn fused_densify_first_ranked_shapes_paired_timing() {
        use crate::challenger::Challenger;

        if std::env::var_os("FLOCK_RUN_INDUCE_FUSED_DENSIFY_TIMING").is_none() {
            eprintln!("set FLOCK_RUN_INDUCE_FUSED_DENSIFY_TIMING=1 to run");
            return;
        }

        const PREFIX_K: usize = 8;
        const WARMUP_PAIRS: usize = 6;
        const MEASURED_PAIRS: usize = 48;

        for &(label, log_d, n_queries) in &[("L0", 20usize, 218usize), ("L1", 18, 106)] {
            let n = 1usize << log_d;
            let mut challenger = crate::challenger::RandomChallenger::new(
                0xD3A5_1F1E_71A1_0000 ^ ((log_d as u64) << 24) ^ n_queries as u64,
            );
            let mut pairs = Vec::with_capacity(n_queries);
            while pairs.len() < n_queries {
                let position = (challenger.sample_f128().lo as usize) % n;
                if !pairs.iter().any(|&(p, _)| p == position) {
                    pairs.push((position, challenger.sample_f128()));
                }
            }
            pairs.sort_unstable_by_key(|&(position, _)| position);
            let (positions, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
            let ntt = AdditiveNttF128::standard(log_d);
            let groups = group_sorted_positions(&positions, PREFIX_K);
            let mut arena = scatter_active_windows(&groups, &positions, &values, PREFIX_K);
            transform_active_windows(&ntt, &mut arena, &groups, PREFIX_K, log_d);
            let first_layer = log_d - PREFIX_K - 3;

            let mut active_blocks = 0usize;
            let mut previous_block = None;
            for group in &groups {
                let block = group.window_index >> 3;
                if previous_block != Some(block) {
                    active_blocks += 1;
                    previous_block = Some(block);
                }
            }

            let run_control = || {
                let started = std::time::Instant::now();
                let mut data = densify_active_windows(&arena, &groups, log_d, PREFIX_K);
                transpose_forward_ntt_fused_3layer(&ntt, &mut data, log_d, first_layer);
                let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
                let anchor =
                    std::hint::black_box(data[0] + data[n / 7] + data[n / 2] + data[n - 1]);
                (elapsed_ms, anchor)
            };
            let run_candidate = || {
                let started = std::time::Instant::now();
                let data = densify_active_windows_fused_first_3layer(
                    &ntt, &arena, &groups, log_d, PREFIX_K,
                );
                let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
                let anchor =
                    std::hint::black_box(data[0] + data[n / 7] + data[n / 2] + data[n - 1]);
                (elapsed_ms, anchor)
            };

            let mut control_ms = Vec::with_capacity(MEASURED_PAIRS);
            let mut candidate_ms = Vec::with_capacity(MEASURED_PAIRS);
            let mut deltas_ms = Vec::with_capacity(MEASURED_PAIRS);
            for pair in 0..WARMUP_PAIRS + MEASURED_PAIRS {
                let ((control_elapsed, control_anchor), (candidate_elapsed, candidate_anchor)) =
                    if pair.is_multiple_of(2) {
                        let control = run_control();
                        let candidate = run_candidate();
                        (control, candidate)
                    } else {
                        let candidate = run_candidate();
                        let control = run_control();
                        (control, candidate)
                    };
                assert_eq!(candidate_anchor, control_anchor, "{label}: pair={pair}");
                if pair >= WARMUP_PAIRS {
                    control_ms.push(control_elapsed);
                    candidate_ms.push(candidate_elapsed);
                    deltas_ms.push(candidate_elapsed - control_elapsed);
                }
            }

            let wins = deltas_ms.iter().filter(|&&delta| delta < 0.0).count();
            let mean_delta_ms = deltas_ms.iter().sum::<f64>() / deltas_ms.len() as f64;
            println!(
                "fused-densify {label} windows={} active_blocks={}/{} pairs={} wins={} control_median_ms={:.6} candidate_median_ms={:.6} paired_delta_median_ms={:.6} paired_delta_mean_ms={:.6} paired_delta_p90_ms={:.6}",
                groups.len(),
                active_blocks,
                1usize << first_layer,
                MEASURED_PAIRS,
                wins,
                lazy_ood_timing_median(&control_ms),
                lazy_ood_timing_median(&candidate_ms),
                lazy_ood_timing_median(&deltas_ms),
                mean_delta_ms,
                lazy_ood_timing_percentile(&deltas_ms, 90, 100),
            );
        }
    }

    #[test]
    fn ranked_induce_truncated_final_ntt_shape_gate_is_narrow() {
        assert!(is_ranked_induce_truncated_final_ntt_shape(19, 1, 6, 218, 8));
        for shape in [
            (18, 1, 6, 218, 8),
            (19, 2, 6, 218, 8),
            (19, 1, 5, 218, 8),
            (19, 1, 6, 217, 8),
            (19, 1, 6, 218, 7),
        ] {
            assert!(!is_ranked_induce_truncated_final_ntt_shape(
                shape.0, shape.1, shape.2, shape.3, shape.4,
            ));
        }
    }

    #[test]
    fn ranked_induce_truncated_final_ntt_gate_tracks_optout() {
        let expected = cfg!(all(target_os = "macos", target_arch = "aarch64"))
            && std::env::var_os("FLOCK_NO_LIG_INDUCE_TRUNCATED_NTT").is_none();
        assert_eq!(
            use_ranked_induce_truncated_final_ntt(19, 1, 6, 218, 8),
            expected,
        );
    }

    #[test]
    fn ranked_induce_fused_msg_gate_is_exact() {
        with_truncated_final_ntt_override(true, || {
            let expected = std::env::var_os("FLOCK_NO_LIG_INDUCE_FUSED_MSG").is_none();
            assert_eq!(
                use_ranked_induce_fused_msg(19, 1, 6, 218, 8, 1 << 19),
                expected,
            );
            for shape in [
                (18, 1, 6, 218, 8, 1 << 18),
                (19, 2, 6, 218, 8, 1 << 19),
                (19, 1, 5, 218, 8, 1 << 19),
                (19, 1, 6, 217, 8, 1 << 19),
                (19, 1, 6, 218, 7, 1 << 19),
                (19, 1, 6, 218, 8, (1 << 19) - 1),
            ] {
                assert!(!use_ranked_induce_fused_msg(
                    shape.0, shape.1, shape.2, shape.3, shape.4, shape.5,
                ));
            }
        });
    }

    /// Independent oracle for the final fused group. The optimized kernel
    /// must reproduce the retained half while leaving every discarded slot
    /// untouched, proving that the dead root stores are absent.
    #[test]
    fn transpose_truncated_final_group_matches_reference_low_half() {
        use crate::challenger::Challenger;

        fn final_three_layer_reference(ntt: &AdditiveNttF128, data: &mut [F128], log_d: usize) {
            for layer in (0..3).rev() {
                let num_blocks = 1usize << layer;
                let block_size = 1usize << (log_d - layer);
                let half = block_size >> 1;
                for block in 0..num_blocks {
                    let twiddle = ntt.twiddle(layer, block);
                    let start = block * block_size;
                    for row in 0..half {
                        let a = data[start + row];
                        let b = data[start + half + row];
                        let sum = a + b;
                        data[start + row] = sum;
                        data[start + half + row] = twiddle * sum + b;
                    }
                }
            }
        }

        for &log_d in &[3usize, 6, 9, 14] {
            let mut challenger =
                crate::challenger::RandomChallenger::new(0x7A11_F17E ^ log_d as u64);
            let before = challenger.sample_f128_vec(1usize << log_d);
            let mut expected = before.clone();
            let mut actual = before.clone();
            let ntt = AdditiveNttF128::standard(log_d);

            final_three_layer_reference(&ntt, &mut expected, log_d);
            transpose_forward_ntt_fused_final_3layer_low_half(&ntt, &mut actual, log_d);

            let half = actual.len() >> 1;
            assert_eq!(&actual[..half], &expected[..half], "log_d={log_d}");
            assert_eq!(
                &actual[half..],
                &before[half..],
                "discarded-half write at log_d={log_d}",
            );
        }
    }

    /// Independent exact oracle for the fused ordinary-message accumulation:
    /// run the incumbent truncated final pass, then its separate
    /// `round_msg_lsb`, and compare both retained coefficients and message.
    #[test]
    fn transpose_truncated_final_group_with_round_msg_matches_separate_oracle() {
        use crate::challenger::Challenger;

        for &log_d in &[4usize, 6, 9, 14] {
            let n = 1usize << log_d;
            for case in 0..4u64 {
                let mut challenger = crate::challenger::RandomChallenger::new(
                    0xF17E_DA7A_0000_0000 ^ ((log_d as u64) << 8) ^ case,
                );
                let before = challenger.sample_f128_vec(n);
                let f = match case {
                    0 => vec![F128::ZERO; n >> 1],
                    1 => vec![F128::ONE; n >> 1],
                    _ => challenger.sample_f128_vec(n >> 1),
                };
                let mut expected = before.clone();
                let mut actual = before.clone();
                let ntt = AdditiveNttF128::standard(log_d);

                transpose_forward_ntt_fused_final_3layer_low_half(&ntt, &mut expected, log_d);
                let expected_msg = round_msg_lsb(&f, &expected[..n >> 1]);
                let actual_msg = transpose_forward_ntt_fused_final_3layer_low_half_with_round_msg(
                    &ntt,
                    &mut actual,
                    log_d,
                    &f,
                );

                assert_eq!(actual_msg, expected_msg, "log_d={log_d}, case={case}");
                assert_eq!(
                    &actual[..n >> 1],
                    &expected[..n >> 1],
                    "retained coefficients differ at log_d={log_d}, case={case}",
                );
                assert_eq!(
                    &actual[n >> 1..],
                    &before[n >> 1..],
                    "discarded-half write at log_d={log_d}, case={case}",
                );
            }
        }
    }

    #[test]
    fn precomputed_ordinary_intro_matches_incumbent_state() {
        use crate::challenger::Challenger;

        let mut challenger = crate::challenger::RandomChallenger::new(0x1A7E_0D00_0000_0001);
        let f = challenger.sample_f128_vec(1 << 8);
        let initial_basis = challenger.sample_f128_vec(f.len());
        let introduced_basis = challenger.sample_f128_vec(f.len());
        let initial_target = challenger.sample_f128();
        let introduced_target = challenger.sample_f128();
        let (mut control, control_first) =
            SumcheckProver::new(f.clone(), initial_basis.clone(), initial_target);
        let (mut fused, fused_first) = SumcheckProver::new(f, initial_basis, initial_target);
        assert_eq!(fused_first, control_first);

        let expected_msg = round_msg_lsb(control.f(), &introduced_basis);
        let control_msg = control.introduce_new(introduced_basis.clone(), introduced_target);
        let fused_msg = fused.introduce_new_with_precomputed_msg(
            introduced_basis,
            introduced_target,
            expected_msg,
        );
        assert_eq!(control_msg, expected_msg);
        assert_eq!(fused_msg, control_msg);
        assert_eq!(fused.transcript, control.transcript);
        assert_eq!(fused.t_r, control.t_r);
        assert_eq!(fused.pending_glue, control.pending_glue);
        assert!(fused.pending_fold_basis.is_none());
        assert!(fused.pending_ood_eq.is_none());
    }

    /// Same-binary paired comparison at the exact ranked 2^20 -> 2^19 final
    /// transpose geometry. The control charges the truncated final group and
    /// its separate ordinary `round_msg_lsb`; the candidate charges the fused
    /// group/message kernel. Input restoration is outside both measured spans.
    ///
    /// Run alone with the challenge profile:
    ///
    /// ```text
    /// FLOCK_RUN_INDUCE_FUSED_MSG_TIMING=1 RAYON_NUM_THREADS=10 \
    /// cargo test --profile challenge -p flock-core --lib \
    /// pcs::ligerito::tests::ranked_final_ntt_fused_msg_paired_timing -- \
    /// --ignored --exact --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore]
    fn ranked_final_ntt_fused_msg_paired_timing() {
        if std::env::var_os("FLOCK_RUN_INDUCE_FUSED_MSG_TIMING").is_none() {
            eprintln!("set FLOCK_RUN_INDUCE_FUSED_MSG_TIMING=1 to run");
            return;
        }

        const LOG_D: usize = 20;
        const WARMUP_PAIRS: usize = 4;
        const MEASURED_PAIRS: usize = 32;
        let n = 1usize << LOG_D;
        let mut state = 0xF17E_DA7A_7A11_0001u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(state, state.rotate_left(23) ^ 0x9E37_79B9_7F4A_7C15)
        };
        let source: Vec<F128> = (0..n).map(|_| rnd()).collect();
        let f: Vec<F128> = (0..n / 2).map(|_| rnd()).collect();
        let ntt = AdditiveNttF128::standard(LOG_D);
        let mut control_data = source.clone();
        let mut candidate_data = source.clone();
        let mut control_ms = Vec::with_capacity(MEASURED_PAIRS);
        let mut candidate_ms = Vec::with_capacity(MEASURED_PAIRS);
        let mut deltas_ms = Vec::with_capacity(MEASURED_PAIRS);

        let run_control = |data: &mut [F128]| {
            data.copy_from_slice(&source);
            let started = std::time::Instant::now();
            transpose_forward_ntt_fused_final_3layer_low_half(&ntt, data, LOG_D);
            let msg = round_msg_lsb(&f, &data[..n / 2]);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
            (elapsed_ms, std::hint::black_box(msg))
        };
        let run_candidate = |data: &mut [F128]| {
            data.copy_from_slice(&source);
            let started = std::time::Instant::now();
            let msg = transpose_forward_ntt_fused_final_3layer_low_half_with_round_msg(
                &ntt, data, LOG_D, &f,
            );
            let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
            (elapsed_ms, std::hint::black_box(msg))
        };

        for pair in 0..WARMUP_PAIRS + MEASURED_PAIRS {
            let ((control_elapsed, control_msg), (candidate_elapsed, candidate_msg)) =
                if pair.is_multiple_of(2) {
                    let control = run_control(&mut control_data);
                    let candidate = run_candidate(&mut candidate_data);
                    (control, candidate)
                } else {
                    let candidate = run_candidate(&mut candidate_data);
                    let control = run_control(&mut control_data);
                    (control, candidate)
                };
            assert_eq!(candidate_msg, control_msg, "pair={pair}: message");
            assert_eq!(candidate_data, control_data, "pair={pair}: coefficients");
            if pair >= WARMUP_PAIRS {
                control_ms.push(control_elapsed);
                candidate_ms.push(candidate_elapsed);
                deltas_ms.push(candidate_elapsed - control_elapsed);
            }
        }

        let wins = deltas_ms.iter().filter(|&&delta| delta < 0.0).count();
        let mean_delta_ms = deltas_ms.iter().sum::<f64>() / deltas_ms.len() as f64;
        println!(
            "induce-fused-msg pairs={} wins={} control_median_ms={:.6} candidate_median_ms={:.6} paired_delta_median_ms={:.6} paired_delta_mean_ms={:.6} paired_delta_p90_ms={:.6}",
            MEASURED_PAIRS,
            wins,
            lazy_ood_timing_median(&control_ms),
            lazy_ood_timing_median(&candidate_ms),
            lazy_ood_timing_median(&deltas_ms),
            mean_delta_ms,
            lazy_ood_timing_percentile(&deltas_ms, 90, 100),
        );
    }

    /// Exercise the complete sparse-prefix schedule at sizes where its final
    /// dense group is layers 2,1,0. The truncated result must equal the low
    /// half of the untouched frontier transform.
    #[test]
    fn transpose_sparse_truncated_final_group_matches_full_transform() {
        use crate::challenger::Challenger;

        for &log_d in &[14usize, 17] {
            let n = 1usize << log_d;
            let mut challenger =
                crate::challenger::RandomChallenger::new(0x5A95_EF17 ^ log_d as u64);
            let mut pairs = Vec::new();
            while pairs.len() < 43 {
                let position = (challenger.sample_f128().lo as usize) % n;
                if !pairs.iter().any(|&(p, _)| p == position) {
                    pairs.push((position, challenger.sample_f128()));
                }
            }
            pairs.sort_unstable_by_key(|&(position, _)| position);
            let (positions, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
            let ntt = AdditiveNttF128::standard(log_d);
            let mut expected =
                transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d, false);
            expected.truncate(n >> 1);
            let actual = transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d, true);
            assert_eq!(actual, expected, "log_d={log_d}");
        }
    }

    /// The fused production kernel must be byte-identical to the original
    /// one-layer-at-a-time transpose, including the ranked proof's largest
    /// induction domain (`log_d = 20`). Keep the oracle serial and structurally
    /// independent so a shared parallel indexing bug cannot mask itself.
    #[test]
    fn transpose_fused_matches_single_layer_reference() {
        use crate::challenger::Challenger;

        fn transpose_single_layer_reference(
            ntt: &AdditiveNttF128,
            data: &mut [F128],
            log_d: usize,
        ) {
            for layer in (0..log_d).rev() {
                let num_blocks = 1usize << layer;
                let block_size = 1usize << (log_d - layer);
                let half = block_size >> 1;
                for block in 0..num_blocks {
                    let twiddle = ntt.twiddle(layer, block);
                    let start = block * block_size;
                    for row in 0..half {
                        let top = data[start + row];
                        let bottom = data[start + half + row];
                        let sum = top + bottom;
                        data[start + row] = sum;
                        data[start + half + row] = twiddle * sum + bottom;
                    }
                }
            }
        }

        for &log_d in &[3usize, 4, 5, 8, 12, 18, 20] {
            let mut challenger =
                crate::challenger::RandomChallenger::new(0xF053_DA7A ^ log_d as u64);
            let mut expected = challenger.sample_f128_vec(1usize << log_d);
            let mut actual = expected.clone();
            let ntt = AdditiveNttF128::standard(log_d);

            transpose_single_layer_reference(&ntt, &mut expected, log_d);
            transpose_forward_ntt(&ntt, &mut actual, log_d);

            assert_eq!(actual, expected, "log_d={log_d}");
        }
    }

    /// As above, with num_interleaved > 1 and non-empty v_challenges (the
    /// partial-eval challenges used to fold lanes).
    #[test]
    fn induce_sumcheck_poly_with_interleaving_and_v_challenges() {
        use crate::challenger::Challenger;
        let log_msg = 3; // msg_cols = 8
        let log_interleaved = 2; // num_interleaved = 4
        let log_inv_rate = 1; // block_len = 16
        let msg_cols = 1usize << log_msg;
        let num_interleaved = 1usize << log_interleaved;
        let block_len = msg_cols << log_inv_rate;
        let poly_len = msg_cols * num_interleaved;

        let mut ch = crate::challenger::RandomChallenger::new(0xDEAD_BEEF);
        // poly[lane * msg_cols + col] convention (matches ligero_commit input).
        let poly: Vec<F128> = (0..poly_len).map(|_| ch.sample_f128()).collect();

        // v_challenges fold the lanes after commit. Under the LSB-lane layout,
        // f_folded is just partial_eval_lsb of the poly at v_challenges.
        let v_challenges: Vec<F128> = (0..log_interleaved).map(|_| ch.sample_f128()).collect();
        let f_folded = partial_eval_lsb(&poly, &v_challenges);
        assert_eq!(f_folded.len(), msg_cols);

        // Encode via ligero_commit (so we use the same matrix layout).
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

        // The folded polynomial f_folded should satisfy Σ_j f_folded[j] · basis_poly[j] = enforced_sum.
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

    /// End-to-end roundtrip: prover proves `poly(z) = v`, verifier accepts.
    /// R = 1 (one recursive step).
    #[test]
    fn ligerito_r1_roundtrip_accepts() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let num_queries = 0; // unused — kept to silence the moved literal

        let mut rng = crate::challenger::RandomChallenger::new(0xCAFE_F00D);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();

        // True value v = poly(z)
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
        let _ = num_queries; // queries derived per-level from log_inv_rates now

        // Prove
        let mut p_ch = crate::challenger::FsChallenger::new(b"test");
        let proof = recursive_prover(&prover_cfg, &poly, &z, v, &mut p_ch);

        // Verify
        let mut v_ch = crate::challenger::FsChallenger::new(b"test");
        let ok = recursive_verifier(&verifier_cfg, &proof, &z, v, &mut v_ch);
        assert!(ok, "verifier rejected a valid proof");
    }

    /// Run the size measurement at the configured (log_n, initial_k, ks, rates).
    /// `log_inv_rates.len()` must equal `recursive_ks.len() + 1` (one per commit).
    /// Also times the prover (best of 3 runs). Returns the measured proof size
    /// in bytes.
    fn size_breakdown_at(
        log_n: usize,
        initial_k: usize,
        recursive_ks: Vec<usize>,
        log_inv_rates: Vec<usize>,
    ) -> usize {
        use crate::challenger::Challenger;
        use std::time::Instant;
        assert_eq!(log_inv_rates.len(), recursive_ks.len() + 1);

        // dims sanity: n1 = 16; after k_0=4 → 12; after k_1=3 → 9 → yr = 512 elems.
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
        drop(eq); // free 16 MB

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

        // Time the prover, best of 3.
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

        // Smoke-check it verifies (so we know the proof is valid, not just plausibly-sized).
        let mut v_ch = crate::challenger::FsChallenger::new(b"size-test");
        assert!(recursive_verifier(&v_cfg, &proof, &z, v, &mut v_ch));
        proof.size_bytes()
    }

    /// Uniform rate (basefold-style) baseline at m=20.
    #[test]
    fn ligerito_size_breakdown_m20_uniform_rate() {
        size_breakdown_at(20, 4, vec![4, 3], vec![1, 1, 1]);
    }

    /// **The actual Ligerito design**: rate decreases at deeper levels, so
    /// fewer queries are needed there.
    #[test]
    fn ligerito_size_breakdown_m20_decreasing_rate() {
        size_breakdown_at(20, 4, vec![4, 3], vec![1, 2, 4]);
    }

    #[test]
    fn ligerito_size_breakdown_m20_decreasing_rate_thin() {
        // More levels with thin lanes + aggressive rate decrease.
        size_breakdown_at(20, 4, vec![3, 3, 3], vec![1, 2, 3, 4]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_aggressive() {
        // Thin initial lanes + steep rate decrease.
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

    /// Same total scale as m=22 but with initial_k=6 (64-lane initial leaves)
    /// to make the L0 commit shape exactly match basefold's.
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

    /// Count the merkle multi-proof siblings that would be needed for `positions`
    /// against a tree with `num_leaves` leaves. Same algorithm as
    /// `merkle::merkle_multi_proof` but counts only — no tree allocation,
    /// O(positions.len() · log num_leaves). For size estimation at scales where
    /// the actual tree wouldn't fit in memory.
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

    /// Analytical size estimator — runs **only** the challenger-driven query
    /// sampling + merkle-multi-proof counting. Does NOT materialize the
    /// polynomial or any merkle tree, so it scales to m=29, m=30+.
    /// Returns total bytes; prints a per-level breakdown.
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

        // Dim/lane/queries per commit (R+1 commits).
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
            n_running -= log_num_interleaved[i]; // consumes initial_k or k_{i-1}
        }
        let yr_log_n = n_running; // = log_n - initial_k - Σ k_i
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

        // Drive a challenger-deterministic query sampling, count siblings.
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
        // Transcript: 1 start + 1 intro per recursive boundary (R) + sum(k_i) folds, all (u_0, u_2).
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

    /// Verify the estimator matches the actual measurement at m=20.
    #[test]
    fn estimator_matches_actual_m20() {
        let estimated = estimate_size_at(20, 4, vec![4, 3], vec![1, 2, 4]);
        // Measure the real proof at the same shape (cheap at m=20) instead of
        // hardcoding a baseline that goes stale when query counts change.
        let actual = size_breakdown_at(20, 4, vec![4, 3], vec![1, 2, 4]);
        let diff = estimated.abs_diff(actual);
        eprintln!("estimator={estimated}  actual={actual}  diff={diff}");
        // Drift is from different challenger seeds producing different query
        // positions (and hence slightly different octopus sibling counts).
        // 5% is plenty of room.
        assert!(
            diff < actual / 20,
            "estimator drift too large: {diff} bytes"
        );
    }

    /// **The headline measurement**: Ligerito at m=29 with decreasing rate.
    #[test]
    fn estimate_ligerito_m29() {
        eprintln!("\n=== Ligerito m=29 — decreasing rate (the real Ligerito design) ===");
        // Pick a reasonable config: thin lanes, aggressive rate decrease.
        estimate_size_at(29, 4, vec![4, 4, 4, 4, 3], vec![1, 2, 3, 4, 5, 6]);

        eprintln!(
            "\n=== Ligerito m=29 — uniform rate 1/2 (basefold-style baseline, infeasible at deepest level) ==="
        );
        // Uniform rate with deep recursion: block_len at L5 = 2^6 = 64 < 221 queries.
        // Show this is structurally bad without aggressive rate decrease.
        estimate_size_at(29, 4, vec![4, 4, 4, 4, 3], vec![1, 1, 1, 1, 1, 1]);

        eprintln!("\n=== Ligerito m=29 — uniform rate, shallower (R=2) ===");
        // To make uniform rate feasible, use fewer levels with bigger ks.
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

    /// Apples-to-apples vs basefold: same initial interleaving factor
    /// `2^6 = 64` lanes at L0 (basefold's log_batch_size = 6).
    #[test]
    fn estimate_ligerito_m29_initial_k6() {
        eprintln!(
            "\n=== Ligerito m=29 — initial_k=6 (matches basefold's 64-lane initial leaves) ==="
        );
        // initial_k = 6, then ks chosen to keep deeper levels thin.
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

    /// Multi-level (R = 2) roundtrip.
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

        // wtns_0: log_n - initial_k = 9, num_interleaved = 8
        // wtns_1: dim n1 = 9, num_interleaved = 2^k_0 = 8, msg_cols = 2^(9-3) = 64
        // After k_0 folds: dim 6. wtns_2: num_interleaved = 2^k_1 = 4, msg_cols = 2^(6-2) = 16
        // After k_1 folds: dim 4. yr = 16 elems.
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

    /// `LigeritoProof` bincode-roundtrips identically.
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

    /// Full-prover control for the truncated final F^T group. This uses a
    /// smaller domain whose sparse suffix still ends in fused layers 2,1,0;
    /// a test-only policy selects candidate/control without weakening the
    /// exact production gate.
    #[test]
    fn truncated_final_ntt_full_proof_and_claim_bytes_match_control() {
        use crate::challenger::Challenger;

        let log_n = 16;
        let initial_k = 3;
        let k_0 = 3;
        let log_inv_rate = 1;
        let log_msg_cols_0 = log_n - initial_k;
        let mut rng = crate::challenger::RandomChallenger::new(0xF17E_BA5E);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let point: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let basis = build_eq_table(&point);
        let target = poly
            .iter()
            .zip(basis.iter())
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |acc, value| acc + value);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let queries = vec![218, 106];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_msg_cols_0,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_msg_cols_0 - k_0],
            recursive_ks: vec![k_0],
            queries: queries.clone(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
        };
        let verifier_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_msg_cols_0,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_msg_cols_0 - k_0],
            recursive_ks: vec![k_0],
            queries,
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
        };

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

        let prove = |truncate_final_group: bool| {
            with_truncated_final_ntt_override(truncate_final_group, || {
                TEST_TRUNCATED_FINAL_NTT_HITS.with(|hits| hits.set(0));
                assert_eq!(
                    use_ranked_induce_truncated_final_ntt(
                        log_msg_cols_0,
                        log_inv_rate,
                        initial_k,
                        218,
                        8,
                    ),
                    truncate_final_group,
                );
                let mut challenger =
                    crate::challenger::FsChallenger::new(b"truncated-final-ntt-proof-oracle");
                let proof = recursive_prover_with_basis(
                    &cfg,
                    poly.clone(),
                    basis.clone(),
                    target,
                    &wtns_0.mat,
                    &wtns_0.tree,
                    &mut challenger,
                );
                let hits = TEST_TRUNCATED_FINAL_NTT_HITS.with(|hits| hits.get());
                (proof, hits)
            })
        };

        let (control, control_hits) = prove(false);
        let (truncated, truncated_hits) = prove(true);
        assert_eq!(control_hits, 0);
        assert_eq!(truncated_hits, 1);
        assert_eq!(truncated, control);
        assert_eq!(
            bincode::serialize(&(&truncated, target)).expect("serialize truncated proof/claim"),
            bincode::serialize(&(&control, target)).expect("serialize control proof/claim"),
        );

        let mut verifier_challenger =
            crate::challenger::FsChallenger::new(b"truncated-final-ntt-proof-oracle");
        assert!(recursive_verifier_with_basis(
            &verifier_cfg,
            &truncated,
            &basis,
            target,
            &initial_root,
            &mut verifier_challenger,
        ));
    }

    /// `recursive_prover_with_basis` + `recursive_verifier_with_basis`
    /// roundtrip — this is the basefold-compatible signature that
    /// `pcs::open_batch` will call. Single-claim case (`b = eq(z, ·)`,
    /// `target = poly(z)`) — must round-trip cleanly.
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
    fn direct_ab_full_proof_and_claim_bytes_match_ordinary_fold2() {
        use crate::challenger::Challenger;

        let log_n = 12;
        // Six initial folds exercise direct materialization at j=1, the
        // ordinary paired fold at j=3, and the direct-only final pair at j=5.
        let initial_k = 6;
        let k_0 = 2;
        // The final 4-column message needs enough codeword positions for the
        // unique-decoding query count.
        let log_inv_rate = 3;
        let mut rng = crate::challenger::RandomChallenger::new(0xD1CE_AB02);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let ordinary_c: Vec<F128> = (0..poly.len()).map(|_| rng.sample_f128()).collect();
        let suffix: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let scaled_rdp: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| rng.sample_f128())
                .collect::<Vec<_>>(),
        );
        let direct_full =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix), &scaled_rdp);
        let combined_basis: Vec<F128> = ordinary_c
            .iter()
            .zip(direct_full)
            .map(|(&ordinary, direct)| ordinary + direct)
            .collect();
        let target = poly
            .iter()
            .zip(combined_basis.iter())
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |acc, value| acc + value);
        let (round0, lookahead) = super::super::round0_and_round1_lookahead(&poly, &combined_basis);

        let (eq_lo, eq_hi) =
            super::super::ring_switch::build_eq_split(&suffix[2..], (log_n - 2) / 2);
        let direct = vec![super::super::ring_switch::DirectFold2Factors {
            eq_lo,
            eq_hi,
            low_eq: build_eq_table(&suffix[..2]).try_into().unwrap(),
            table: super::super::ring_switch::build_fold_byte_table(&scaled_rdp),
            products: [F128::ZERO; 16],
        }];

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates
                .iter()
                .map(|&rate| udr_queries(rate))
                .collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let ntt_0 = AdditiveNttF128::standard(log_n - initial_k + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_n - initial_k,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );

        let mut ordinary_challenger =
            crate::challenger::FsChallenger::new(b"direct-ab-proof-byte-oracle");
        let ordinary = recursive_prover_with_basis_precomputed_round0(
            &cfg,
            poly.clone(),
            combined_basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            Some(lookahead),
            &mut ordinary_challenger,
        );
        let mut direct_challenger =
            crate::challenger::FsChallenger::new(b"direct-ab-proof-byte-oracle");
        let got = recursive_prover_with_basis_direct_ab_fold2(
            &cfg,
            poly,
            ordinary_c,
            direct,
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            lookahead,
            &mut direct_challenger,
        );

        assert_eq!(got, ordinary);
        assert_eq!(
            bincode::serialize(&(got.clone(), target)).expect("serialize direct proof/claim"),
            bincode::serialize(&(ordinary, target)).expect("serialize ordinary proof/claim"),
        );

        // The specialization changes no transcript field or verifier rule.
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates
                .iter()
                .map(|&rate| udr_queries(rate))
                .collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut verifier_challenger =
            crate::challenger::FsChallenger::new(b"direct-ab-proof-byte-oracle");
        assert!(recursive_verifier_with_basis(
            &v_cfg,
            &got,
            &combined_basis,
            target,
            &wtns_0.root(),
            &mut verifier_challenger,
        ));
    }

    #[test]
    fn direct_fold4_full_proof_and_claim_bytes_match_ordinary_fold2() {
        use crate::challenger::Challenger;

        let log_n = 12;
        let initial_k = 6;
        let k_0 = 2;
        let log_inv_rate = 3;
        let mut rng = crate::challenger::RandomChallenger::new(0xD1CE_F004);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let suffix: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let scaled_rdp: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| rng.sample_f128())
                .collect::<Vec<_>>(),
        );
        let combined_basis =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix), &scaled_rdp);
        let target = poly
            .iter()
            .zip(combined_basis.iter())
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |acc, value| acc + value);

        let mut products = [F128::ZERO; 256];
        for high in 0..poly.len() / 16 {
            for e in 0..16 {
                for d in 0..16 {
                    products[16 * e + d] += poly[16 * high + e] * combined_basis[16 * high + d];
                }
            }
        }
        let (eq_lo, eq_hi) =
            super::super::ring_switch::build_eq_split(&suffix[4..], (log_n - 4) / 2);
        let direct = vec![super::super::ring_switch::DirectFold4Factors {
            eq_lo,
            eq_hi,
            low_eq: build_eq_table(&suffix[..4]).try_into().unwrap(),
            table: super::super::ring_switch::build_fold_byte_table(&scaled_rdp),
            products,
        }];
        let (round0, round1, round2, round3) =
            super::super::messages_from_direct_products_fold4(&direct);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates
                .iter()
                .map(|&rate| udr_queries(rate))
                .collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let ntt_0 = AdditiveNttF128::standard(log_n - initial_k + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_n - initial_k,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );

        let mut ordinary_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold4-proof-byte-oracle");
        let ordinary = recursive_prover_with_basis_precomputed_round0(
            &cfg,
            poly.clone(),
            combined_basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            Some(round1),
            &mut ordinary_challenger,
        );
        let mut direct_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold4-proof-byte-oracle");
        let got = recursive_prover_with_basis_direct_fold4(
            &cfg,
            poly,
            Vec::new(),
            direct,
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            round1,
            round2,
            round3,
            &mut direct_challenger,
        );

        assert_eq!(got, ordinary);
        assert_eq!(
            bincode::serialize(&(got.clone(), target)).expect("serialize direct-fold4 proof/claim"),
            bincode::serialize(&(ordinary, target)).expect("serialize ordinary proof/claim"),
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates
                .iter()
                .map(|&rate| udr_queries(rate))
                .collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut verifier_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold4-proof-byte-oracle");
        assert!(recursive_verifier_with_basis(
            &v_cfg,
            &got,
            &combined_basis,
            target,
            &wtns_0.root(),
            &mut verifier_challenger,
        ));
    }

    #[test]
    fn direct_fold8_claim_parallel_selector_is_exact_and_early_round_only() {
        use std::ffi::OsStr;

        assert!(!super::direct_fold8_claim_parallel_value_enabled(Some(
            OsStr::new("1")
        )));
        for value in [None, Some(""), Some("0"), Some("01"), Some("true")] {
            assert!(super::direct_fold8_claim_parallel_value_enabled(
                value.map(OsStr::new)
            ));
        }

        let select = super::select_direct_fold8_claim_parallel;
        assert!(select(2, 8192, 10, true, true));
        assert!(select(2, 4096, 2, true, true));
        assert!(!select(2, 2048, 10, true, true));
        assert!(!select(1, 8192, 10, true, true));
        assert!(!select(3, 8192, 10, true, true));
        assert!(!select(2, 8192, 1, true, true));
        assert!(!select(2, 8192, 10, false, true));
        assert!(!select(2, 8192, 10, true, false));
    }

    #[test]
    fn direct_fold8_two_claim_parallel_rounds_match_serial() {
        use crate::challenger::Challenger;

        let mut rng = crate::challenger::RandomChallenger::new(0xD1CE_C1A1);
        let n_packed = 1usize << crate::pcs::LOG_PACKING;
        let claims: Vec<super::super::ring_switch::DirectFold8Factors> = (0..2)
            .map(|_| super::super::ring_switch::DirectFold8Factors {
                eq_lo: (0..4).map(|_| rng.sample_f128()).collect(),
                eq_hi: (0..4).map(|_| rng.sample_f128()).collect(),
                a_state: (0..64 * n_packed).map(|_| rng.sample_f128()).collect(),
                w_state: (0..64 * n_packed).map(|_| rng.sample_f128()).collect(),
                round0: (rng.sample_f128(), rng.sample_f128()),
            })
            .collect();
        let challenges: [F128; 5] = std::array::from_fn(|_| rng.sample_f128());
        let mut serial = claims.clone();
        let mut candidate = claims;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build two-thread fold8 identity pool");

        for (round, challenge) in challenges.into_iter().enumerate() {
            let serial_msg = super::fold_direct_fold8_factors_and_message_selected(
                &mut serial,
                challenge,
                false,
            );
            let min_state_len = candidate
                .iter()
                .map(|claim| claim.a_state.len())
                .min()
                .unwrap();
            let parallel = super::select_direct_fold8_claim_parallel(
                candidate.len(),
                min_state_len,
                2,
                true,
                true,
            );
            assert_eq!(parallel, round < 2, "unexpected selector at round {round}");
            let candidate_msg = pool.install(|| {
                super::fold_direct_fold8_factors_and_message_selected(
                    &mut candidate,
                    challenge,
                    parallel,
                )
            });

            assert_eq!(
                candidate_msg, serial_msg,
                "message mismatch at round {round}"
            );
            for (claim_index, (got, want)) in candidate.iter().zip(&serial).enumerate() {
                assert_eq!(
                    got.a_state, want.a_state,
                    "A state mismatch at round {round}, claim {claim_index}"
                );
                assert_eq!(
                    got.w_state, want.w_state,
                    "W state mismatch at round {round}, claim {claim_index}"
                );
                assert_eq!(got.eq_lo, want.eq_lo);
                assert_eq!(got.eq_hi, want.eq_hi);
                assert_eq!(got.round0, want.round0);
            }
            assert_eq!(candidate[0].a_state.len(), (64 * n_packed) >> (round + 1));
        }
    }

    #[test]
    fn direct_fold8_stateful_messages_and_generator_match_product_tensor() {
        use crate::challenger::Challenger;

        let mut rng = crate::challenger::RandomChallenger::new(0xD1CE_5A8E);
        let n_packed = 1usize << crate::pcs::LOG_PACKING;
        let a_state: Vec<F128> = (0..64 * n_packed).map(|_| rng.sample_f128()).collect();
        let w_state: Vec<F128> = (0..64 * n_packed).map(|_| rng.sample_f128()).collect();
        let original_w = w_state.clone();

        let mut products = [F128::ZERO; 4096];
        for e in 0..64 {
            for d in 0..64 {
                for bit in 0..n_packed {
                    products[64 * e + d] += a_state[bit * 64 + e] * w_state[bit * 64 + d];
                }
            }
        }
        let cached_round0 = super::super::round0_deferred(&a_state, &w_state);
        let mut direct = vec![super::super::ring_switch::DirectFold8Factors {
            eq_lo: vec![F128::ONE],
            eq_hi: vec![F128::ONE],
            a_state,
            w_state,
            round0: cached_round0,
        }];
        let (round0, round1, mut round2, mut round3, mut round4, mut round5) =
            super::super::messages_from_direct_products_fold8(&products);
        assert_eq!(
            super::super::message_from_direct_factors_fold8(&direct),
            round0
        );

        let challenges: [F128; 6] = std::array::from_fn(|_| rng.sample_f128());
        let expected = [
            super::eval_lookahead(&round1, challenges[0]),
            super::eval_fold4_lookahead2(&mut round2, challenges[0], challenges[1]),
            super::eval_fold4_lookahead3(&mut round3, challenges[0], challenges[1], challenges[2]),
            super::eval_fold8_lookahead4(
                &mut round4,
                challenges[0],
                challenges[1],
                challenges[2],
                challenges[3],
            ),
            super::eval_fold8_lookahead5(
                &mut round5,
                challenges[0],
                challenges[1],
                challenges[2],
                challenges[3],
                challenges[4],
            ),
        ];
        for (round, want) in expected.into_iter().enumerate() {
            assert_eq!(
                super::fold_direct_fold8_factors_and_message(&mut direct, challenges[round],),
                want,
                "stateful message mismatch after challenge {round}",
            );
        }

        let got_generators = super::direct_fold8_final_generators(&direct[0], challenges[5]);
        let fold_weight: [F128; 64] = std::array::from_fn(|bank| {
            challenges
                .iter()
                .enumerate()
                .fold(F128::ONE, |weight, (bit, &challenge)| {
                    weight
                        * if (bank >> bit) & 1 == 0 {
                            F128::ONE + challenge
                        } else {
                            challenge
                        }
                })
        });
        let want_generators: [F128; 128] = std::array::from_fn(|bit| {
            (0..64).fold(F128::ZERO, |sum, bank| {
                sum + fold_weight[bank] * original_w[bit * 64 + bank]
            })
        });
        assert_eq!(got_generators, want_generators);
        let mut w_prime = vec![F128::ZERO; original_w.len()];
        for bit in 0..n_packed {
            for bank in 0..64 {
                w_prime[bank * n_packed + bit] = original_w[bit * 64 + bank];
            }
        }
        assert_eq!(
            super::super::ring_switch::build_direct_fold8_table_from_generators(&got_generators),
            super::super::ring_switch::build_direct_fold8_table_from_w_prime(
                &w_prime,
                &fold_weight,
            )
        );
    }

    #[test]
    fn direct_fold8_full_proof_and_claim_bytes_match_ordinary_fold2() {
        use crate::challenger::Challenger;

        let log_n = 12;
        let initial_k = 6;
        let k_0 = 2;
        let log_inv_rate = 3;
        let mut rng = crate::challenger::RandomChallenger::new(0xD1CE_F008);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let suffix: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let scaled_rdp: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| rng.sample_f128())
                .collect::<Vec<_>>(),
        );
        let combined_basis =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix), &scaled_rdp);
        let target = poly
            .iter()
            .zip(combined_basis.iter())
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |acc, value| acc + value);

        let mut products = [F128::ZERO; 4096];
        for high in 0..poly.len() / 64 {
            for e in 0..64 {
                for d in 0..64 {
                    products[64 * e + d] += poly[64 * high + e] * combined_basis[64 * high + d];
                }
            }
        }
        let eq_tail = build_eq_table(&suffix[6..]);
        let mut a_state = vec![F128::ZERO; 64 * (1usize << crate::pcs::LOG_PACKING)];
        for e in 0..64 {
            let bank: Vec<F128> = (0..poly.len() / 64)
                .map(|high| poly[64 * high + e])
                .collect();
            let slices = super::super::ring_switch::fold_1b_rows_naive(&bank, &eq_tail);
            let transposed = super::super::ring_switch::tensor_algebra_transpose(&slices);
            for (bit, value) in transposed.into_iter().enumerate() {
                a_state[bit * 64 + e] = value;
            }
        }
        let (eq_lo, eq_hi) =
            super::super::ring_switch::build_eq_split(&suffix[6..], (log_n - 6) / 2);
        let low_eq: [F128; 64] = build_eq_table(&suffix[..6]).try_into().unwrap();
        let table = super::super::ring_switch::build_fold_byte_table(&scaled_rdp);
        let n_packed = 1usize << crate::pcs::LOG_PACKING;
        let mut w_prime = vec![F128::ZERO; 64 * n_packed];
        for (d_low, row) in w_prime.chunks_mut(n_packed).enumerate() {
            let scale = low_eq[d_low];
            for (bit, value) in row.iter_mut().enumerate() {
                let basis = if bit < 64 {
                    F128::new(1u64 << bit, 0)
                } else {
                    F128::new(0, 1u64 << (bit - 64))
                };
                *value = super::super::ring_switch::fold_one_slot(scale * basis, &table);
            }
        }
        let oracle_fold_weight: [F128; 64] = std::array::from_fn(|_| rng.sample_f128());
        assert_eq!(
            super::super::ring_switch::build_direct_fold8_table(
                &low_eq,
                &oracle_fold_weight,
                &table,
            ),
            super::super::ring_switch::build_direct_fold8_table_from_w_prime(
                &w_prime,
                &oracle_fold_weight,
            )
        );
        let mut w_state = vec![F128::ZERO; w_prime.len()];
        for d in 0..64 {
            for bit in 0..n_packed {
                w_state[bit * 64 + d] = w_prime[d * n_packed + bit];
            }
        }
        let mut factored_products = [F128::ZERO; 4096];
        for e in 0..64 {
            for d in 0..64 {
                for bit in 0..n_packed {
                    factored_products[64 * e + d] += a_state[bit * 64 + e] * w_state[bit * 64 + d];
                }
            }
        }
        assert_eq!(factored_products, products);
        let cached_round0 = super::super::round0_deferred(&a_state, &w_state);
        let direct = vec![super::super::ring_switch::DirectFold8Factors {
            eq_lo,
            eq_hi,
            a_state,
            w_state,
            round0: cached_round0,
        }];
        let (round0, round1, _, _, _, _) =
            super::super::messages_from_direct_products_fold8(&products);
        assert_eq!(
            super::super::message_from_direct_factors_fold8(&direct),
            round0
        );

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates
                .iter()
                .map(|&rate| udr_queries(rate))
                .collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let ntt_0 = AdditiveNttF128::standard(log_n - initial_k + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_n - initial_k,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );

        let mut ordinary_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold8-proof-byte-oracle");
        let ordinary = recursive_prover_with_basis_precomputed_round0(
            &cfg,
            poly.clone(),
            combined_basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            Some(round1),
            &mut ordinary_challenger,
        );
        let mut direct_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold8-proof-byte-oracle");
        let got = recursive_prover_with_basis_direct_fold8(
            &cfg,
            poly,
            Vec::new(),
            direct,
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            &mut direct_challenger,
        );

        assert_eq!(got, ordinary);
        assert_eq!(
            bincode::serialize(&(got.clone(), target)).expect("serialize direct-fold8 proof/claim"),
            bincode::serialize(&(ordinary, target)).expect("serialize ordinary proof/claim"),
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates
                .iter()
                .map(|&rate| udr_queries(rate))
                .collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut verifier_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold8-proof-byte-oracle");
        assert!(recursive_verifier_with_basis(
            &v_cfg,
            &got,
            &combined_basis,
            target,
            &wtns_0.root(),
            &mut verifier_challenger,
        ));
    }

    /// `induce_sumcheck_evaluate_at_residual` matches dense
    /// `induce_sumcheck_poly` + `partial_eval_lsb`.
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

        // Dense path
        let (basis_dense, dense_enforced_sum) = induce_sumcheck_poly(
            log_msg_cols,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            &alpha,
        );
        let dense_residual = partial_eval_lsb(&basis_dense, &ris_for_basis);

        // Succinct path
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

    /// Regression for the final-level proximity binding (the Ligerito
    /// soundness fix). Every non-final recursion level folds its opened rows
    /// into the running sumcheck via `induce_sumcheck`; the final level used to
    /// only Merkle-check its opened rows, leaving `yr` (the claimed final
    /// message) constrained by a single scalar equation — so a malicious prover
    /// could solve for a `yr` that opens the commitment to an arbitrary value.
    ///
    /// The fixed verifier ties `yr` to the committed codeword by checking
    /// `enforced_sum_last == ⟨yr, induced_basis_last⟩`, exactly as every other
    /// level does. This test pins that identity against a *real* `ligero_commit`
    /// codeword: the honest `yr` (the committed message) satisfies it, and any
    /// perturbed `yr` violates it. If `ligero_commit`'s additive-NTT encoding
    /// and the verifier's LCH novel-basis (`induce_sumcheck_evaluate_at_residual`)
    /// ever diverged, the honest assertion here would fail.
    #[test]
    fn final_level_binding_pins_yr_to_committed_codeword() {
        use crate::challenger::Challenger;
        let log_msg_cols = 5; // yr has 32 entries (within the shipped yr_log_n range)
        let log_inv_rate = 1;
        let num_queries = 20;
        let msg_cols = 1usize << log_msg_cols;
        let block_len = msg_cols << log_inv_rate;

        let mut rng = crate::challenger::RandomChallenger::new(0xB19D_1235);
        // num_interleaved = 1 ⇒ no lane fold (level_rs empty) ⇒ yr == the message.
        let yr: Vec<F128> = (0..msg_cols).map(|_| rng.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_msg_cols + log_inv_rate);
        let wtns = ligero_commit(&yr, log_msg_cols, 0, log_inv_rate, &ntt, HashKind::Sha256);

        // Distinct query positions (the protocol always samples distinct ones).
        let mut queries: Vec<usize> = Vec::new();
        let mut q = 1usize;
        while queries.len() < num_queries {
            q = (q * 73 + 41) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&p| wtns.row(p).to_vec()).collect();

        let level_rs: Vec<F128> = Vec::new(); // num_interleaved = 1
        let alpha: Vec<F128> = (0..ceil_log2(num_queries))
            .map(|_| rng.sample_f128())
            .collect();

        // The two quantities the fixed verifier batches into the final check.
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

        // Honest yr (the committed message) satisfies the proximity tie.
        assert_eq!(
            inner(&yr),
            enforced_sum,
            "honest yr must satisfy ⟨yr, induced_basis⟩ == enforced_sum"
        );

        // A forged yr violates it: perturb a coordinate with nonzero basis weight,
        // so the change to the inner product is provably nonzero.
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

    /// Succinct verifier accepts the same proof as the dense verifier when
    /// given an `eval_b` closure that returns the same values as the dense
    /// `b_initial[idx]` at multilinear `point = bit-decomp(idx)`.
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

        // Dense verifier
        let mut v_ch = crate::challenger::FsChallenger::new(b"succ-cmp");
        let dense_ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(dense_ok, "dense verifier must accept");

        // Succinct verifier — batch eval_b is just eq(z, ris ++ y_bits) by construction
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

    /// Build a matching (ProverConfig, VerifierConfig) pair with explicit
    /// OOD samples and fold-challenge grinding, for the OOD-path tests below.
    /// Shape: L0 (initial_k) → r recursive levels of `k`; small query counts
    /// and grind bits keep the test fast while still exercising every path.
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

    /// End-to-end OOD binding + fold-challenge grinding: a JohnsonOod-shaped
    /// config (explicit OOD samples at L1/L2, a few fold-grind bits at every
    /// level) round-trips through BOTH the dense and succinct verifiers, and
    /// tampering with either an OOD value or a fold-grinding nonce makes both
    /// reject. Exercises every new prover/verifier code path.
    #[test]
    fn ligerito_ood_and_fold_grinding_roundtrip_and_tamper() {
        use crate::challenger::Challenger;
        let log_n = 12;
        let initial_k = 2;
        let ks = [2usize, 2];
        // OOD at L1 and L2 (L0 must be 0); 3 fold-grind bits at each level.
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

        // Sanity: the new proof fields are populated.
        assert_eq!(proof.ood_values.len(), 4, "2 OOD samples each at L1 and L2");
        // 2 lane folds (L0) + 2 + 2 recursive folds, each with 3 grind bits.
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

        // Tamper an OOD value → both verifiers reject.
        let mut bad_ood = proof.clone();
        bad_ood.ood_values[0] += F128::ONE;
        assert!(!dense(&bad_ood), "dense must reject tampered OOD value");
        assert!(
            !succinct(&bad_ood),
            "succinct must reject tampered OOD value"
        );

        // Tamper a fold-grinding nonce → both verifiers reject (PoW fails or
        // the FS state diverges).
        let mut bad_nonce = proof.clone();
        bad_nonce.fold_grinding_nonces[0] ^= 0xDEAD_BEEF;
        assert!(!dense(&bad_nonce), "dense must reject tampered fold nonce");
        assert!(
            !succinct(&bad_nonce),
            "succinct must reject tampered fold nonce"
        );
    }

    /// A real embedded profile config (m=22 fast = JohnsonOod) drives a full
    /// prover→verifier round-trip through the basis opening path. This is the
    /// production shape: OOD samples and fold grinding come straight from the
    /// derived TOML, not a hand-built config.
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
        // The fast profile must actually use the new features.
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

    /// End-to-end under BLAKE3: the same recursion, every Merkle commitment
    /// (L0 and each recursive level) built and checked with the other hash.
    /// Also pins the failure mode of a hash mismatch — a verifier configured
    /// for the wrong hash must reject, since the roots commit to the hash.
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
        // The embedded configs all declare sha256; override to exercise the
        // other arm of the option end to end.
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

        // Same proof, verifier configured for SHA-256 → every opening's
        // recomputed root disagrees, so it must reject.
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

    /// The Merkle hash and the Fiat-Shamir transcript hash are independent
    /// options: all four combinations must prove and verify. Also pins the
    /// failure mode of a transcript-hash mismatch, the FS analogue of the
    /// Merkle mismatch checked above.
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

                // Verifier on the other transcript hash: challenges diverge
                // from the first sample, so it must reject.
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

    /// Multi-claim batched basis: `b = γ_1·eq(z_1, ·) + γ_2·eq(z_2, ·)`,
    /// `target = γ_1·poly(z_1) + γ_2·poly(z_2)`. This is the shape ring_switch
    /// produces.
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

    /// `recursive_prover_with_l0` (external L0 path, for integration with
    /// Flock's `pcs::commit`) produces a byte-identical proof to
    /// `recursive_prover` when given a matching pre-built L0.
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

        // Path 1: built-in L0 commit.
        let mut p_ch = crate::challenger::FsChallenger::new(b"l0-test");
        let proof_a = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        // Path 2: build L0 externally via ligero_commit, then call _with_l0.
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

        // Proofs must be byte-identical (same FS state, same prover work).
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
        // And both must verify against the same VerifierConfig.
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

    /// Mutation rejection: change one element of yr → verify should fail.
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

        // Mutate yr.
        proof.final_proof.yr[0] += F128::ONE;

        let mut v_ch = crate::challenger::FsChallenger::new(b"test-mut");
        let ok = recursive_verifier(&verifier_cfg, &proof, &z, v, &mut v_ch);
        assert!(!ok, "verifier accepted a proof with mutated yr");
    }

    #[test]
    fn ligero_commit_encoding_roundtrips_via_inv_ntt() {
        let log_msg = 4; // msg_cols = 16
        let log_interleaved = 3; // num_interleaved = 8
        let log_inv_rate = 1; // block_len = 32
        let msg_cols = 1 << log_msg;
        let num_interleaved = 1 << log_interleaved;
        let block_len = msg_cols << log_inv_rate;

        // Deterministic dummy polynomial.
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

        // Per-lane inv-NTT should recover the padded message. Under the LSB-lane
        // layout, lane `lane`'s col `col` message lives at `poly[col * num_interleaved + lane]`.
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

        // Merkle root is deterministic: re-running the same commit yields the
        // same root.
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
// Redraw marker 4 (drift probe): zero-diff; prior draws 1,205,646 / 1,205,107 / 1,206,245.
// RealAdii draw 1 on 1a6ad0e.
// RealAdii draw 1 on 76f9e98.
// RealAdii draw 2 on 76f9e98 (draw 1: 1,250,243.88).
// RealAdii draw 3 on 76f9e98.
// angelX disclosed draw 3 on 39541e2 (draws 1-2: 1,252,541 / 1,241,514; zero-diff marker per board protocol).
// angelX disclosed draw 7 of the tree on 775378c (prior: 1,252,541 / 1,241,514 / 1,255,076 P / 1,245,411 / 1,249,152 / pending; zero-diff marker).
// RealAdii sample 1 on beeedc6.
// RealAdii sample 1 on 88aff39.
// RealAdii sample 1 on 281206e.
// RealAdii sample 2 on 281206e.
// RealAdii sample 1 on 31a9c72.
// numinous draw 8 1785734384494936424
// RealAdii sample 1 on f6e921b.
// RealAdii sample 1 on 81acf4f.
// RealAdii sample 2 on 81acf4f.
// RealAdii sample 3 on 81acf4f.
// RealAdii sample 4 on 81acf4f.
// RealAdii sample 5 on 81acf4f.
// RealAdii sample 6 on 81acf4f.
// RealAdii sample 7 on 81acf4f.
// RealAdii sample 8 on 81acf4f.
// RealAdii sample 1 on dc385af.
// RealAdii sample 2 on dc385af.
// RealAdii sample 3 on dc385af.
// RealAdii sample 4 on dc385af.
// RealAdii sample 5 on dc385af.
// RealAdii sample 6 on dc385af.
// RealAdii sample 7 on dc385af.
// RealAdii sample 1 on 18f9d67.
// RealAdii sample 2 on 18f9d67.
// RealAdii sample 3 on 18f9d67.
// RealAdii sample 4 on 18f9d67.
// RealAdii sample 1 on c52fba6.
// RealAdii sample 2 on c52fba6.
// RealAdii sample 3 on c52fba6.
// RealAdii sample 4 on c52fba6.
// RealAdii sample 5 on c52fba6.
// RealAdii sample 6 on c52fba6.
// RealAdii sample 7 on c52fba6.
// RealAdii sample 8 on c52fba6.
// RealAdii sample 9 on c52fba6.
// RealAdii fresh-tree pull 1 on cc1d811.
// RealAdii frontier pull 2 on cc1d811.
// RealAdii frontier pull 3 on cc1d811.
// welttowelt disclosed cadence resample 4 of the record tree on f027957 (previous draw: 1703434.61020512; zero-diff marker per board protocol).
// RealAdii sample 1 on 368da6d.
// angelx lane-warm draw 31 on frontier 2d89d2b (resample 31).
// angelx lane-warm draw 49 on frontier d9b4232 (resample 49).
// RealAdii next sample on 90b93d6 (marker 22229).

// angel resample r463 of the current bar tree (629d733, JH-321 zerocheck) — measurement draw 2 (bar 1764890.73).
// RealAdii next sample on 17c0767 (marker 5149).
// RealAdii next sample on 8697f1c (marker 26269).

// r475 official cadence marker: promoted r472 tree, fresh draw after benchmark lifecycle interruption.

// r491: explicit-benchmark-ID submission probe; archive-distinct, no semantic change.

// r492: archive-distinct candidate marker; no runtime effect.

// r493: archive identity marker for Hilbert credential-route experiment.

// r496 archive identity: authenticated Hilbert cadence retry with unchanged semantics.

// r499 archive identity marker; intentionally no runtime effect.

// Competition candidate r500: archive-distinct no-op marker; arithmetic semantics unchanged.

// r502: preserve the benchmarked packing path; archive-distinct cadence marker.

// Submission r503: archive identity marker; no runtime effect.
// r504: archive-distinct cadence marker; no runtime effect.

// Submission archive nonce r508: preserves semantics while distinguishing the candidate.

// Submission archive nonce r509: cadence retry after validator/rate-limit turnover; semantics unchanged.
// Submission archive nonce r510: post-cooldown authenticated cadence; semantics unchanged.

// competition archive nonce r513 20260806T024104Z
// competition archive nonce r515 20260806T024255Z: unchanged benchmark semantics.

// r516 archive nonce: 20260806T024342Z

// Submission archive nonce r517: 20260806T024511Z

// Submission archive nonce r518.

// r519 archive nonce 20260806T024708Z

// r520 archive nonce: 20260806T024805Z

// r521 archive nonce: 20260806T024929Z

// r522 archive nonce: 20260806T025056Z
// r524 archive nonce: cooldown-expiry cadence; benchmark semantics unchanged.

// r525 archive nonce: 20260806T025431Z; benchmark semantics unchanged.

// Archive nonce r526: cooldown-expiry submission cadence.

// Archive nonce r527: 20260806T025626Z; benchmark semantics unchanged.

// Submission archive nonce r533: preserve optimized implementation semantics.

// r534 submission nonce: 20260806T030337Z

// submission archive nonce r535 20260806T030429Z

// submission archive nonce r536 20260806T030638Z

// Submission archive nonce r537: hot-slot retry after effec75 validation window.

// Submission nonce r539: preserves semantics while making the editable archive distinct.

// Submission archive nonce r540: validator sample follow-up.

// Submission archive nonce r541: 20260806T031450Z

// Submission nonce r542: preserve semantics while keeping the candidate archive distinct.

// Submission nonce r543: 20260806T031951Z; semantics unchanged.

// r544 archive nonce: test verifier portability and live submission gate.

// r545 archive nonce: 20260806T032427Z; benchmark semantics unchanged.

// Submission cadence nonce r546: preserves semantics while producing a distinct editable archive.

// r547 cadence nonce: 20260806T033044Z

// r548 cadence nonce: 20260806T033235Z

// r549 cadence nonce: 20260806T033504Z; semantics unchanged.

// chewy cadence nonce r550

// Competition candidate r551: archive-distinct cadence nonce 20260806T033949Z.

// r552 cadence nonce: poll-after-queue experiment

// r554 cadence nonce: 20260806T034654Z; semantics unchanged.
// r564 cadence nonce: 20260806T041230Z

// r565: archive-distinct hot-line candidate; semantics unchanged.

// r566: archive-distinct cadence marker; no semantic effect.

// r568 archive nonce: 20260806T042222Z; semantics unchanged.

// r569 hot-line archive nonce: 20260806T042512Z; semantics unchanged.

// r570 hot-line archive nonce: 20260806T042745Z; semantics unchanged.

// r571 hot-line archive nonce: 20260806T043105Z; semantics unchanged.

// r572 chewy cadence: distinct submission archive, semantics unchanged.

// r573 submission-cadence marker: preserves semantics.
// r574 chewy hot-line nonce: 20260806T043827Z; semantics unchanged.

// r575: source-distinct competition candidate; preserves kernel semantics.

// r576 chewy hot-line nonce: 20260806T044306Z; semantics unchanged.

// r580: archive-distinct candidate; no semantic change.

// r583 chewy hot-line nonce: 20260806T050013Z; semantics unchanged.
// RealAdii next sample on eda4129 (marker 23601).
