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

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::build_eq_table;
use crate::merkle::{self, Hash, HashKind};
use crate::ntt::additive_ntt_f128::AdditiveNttF128;
use serde::{Deserialize, Serialize};

/// `FLOCK_OPEN_TIMING`: per-level open-phase instrumentation — recursive
/// commit shapes with their NTT-encode/Merkle split, plus the section totals
/// the `LIG_PROVE_TRACE` breakdown already prints. Read once per process
/// (diagnostics only; the ranked worker's cleared env never sets it).
pub(crate) fn open_timing() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_OPEN_TIMING").is_some());
    *ON
}

/// `FLOCK_NO_OPEN_FILL=1` restores the incumbent open-phase scheduling at every
/// site where the recursive open left cores idle for want of tasks (measured
/// per sub-phase on a quiet 16-vCPU c7i with `CLOCK_PROCESS_CPUTIME_ID` deltas
/// across in-process phase marks):
///
///   * `transpose_forward_ntt_dense_layers_blocked` pass (b): the tile count is
///     `seg / tile` with `tile` sized purely for cache residency. At the L1
///     induce (`log_d = 18`) that is FOUR tiles on sixteen threads; the floor
///     added here caps the tile so at least `n_threads` tiles exist. The ranked
///     L0 induce (`log_d = 20`) already sits exactly at the floor, so its tuned
///     tile is untouched.
///   * `induce_sumcheck_poly`'s cross-thread reduce: `chunk` was floored at
///     `REDUCE_PAR_FLOOR = 2^12`, which is TWO chunks at the L2 induce's
///     `n = 2^13`. The floor drops to `2^9` (still ≥ one 8 KiB slab per task).
///   * `build_eq_table_split`'s `SPLIT_MIN_LOG`: the L2/L3 OOD tables
///     (`d = 16`, `13`) fell below 17 and ran the serial doubling recurrence —
///     65 535 dependent GHASH multiplies on ONE core inside a phase whose other
///     two passes use all sixteen.
///   * the sparse-prefix NTT's window grouping: a serial `HashMap` pass that
///     allocated and ZEROED one `2^k` buffer per active window (213 × 4 KiB at
///     the ranked L0) before any thread saw them; now the buffers are built and
///     transposed in ONE parallel region, one task per window.
///   * the opened-row `clone()`s feeding `RecursiveProof`: the rows are moved
///     into the proof after the induce reads them instead of copied before it.
///
/// Every arm is bit-identical: no arithmetic, operand order, or transcript
/// message changes — only how many tasks the same work is cut into (F128
/// addition is XOR, and each site's per-slot summation order is preserved).
/// Read once per process; default ON (the ranked worker clears its env).
pub(crate) fn open_fill_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_FILL").is_none());
    *ON
}

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

/// Parse and derive the immutable embedded configurations once. The mandatory
/// untimed worker proof pays initialization; ranked proofs only scan and clone.
#[allow(clippy::type_complexity)]
static PARSED_EMBEDDED_CONFIGS: std::sync::LazyLock<
    Vec<(
        (usize, LigeritoProfile),
        Result<(usize, ProverConfig, VerifierConfig), String>,
    )>,
> = std::sync::LazyLock::new(|| {
    EMBEDDED_CONFIGS
        .iter()
        .map(|&(key, toml)| {
            let parsed = LigeritoSecurityConfig::from_toml_str(toml).and_then(|sec| {
                let initial_k = sec.initial_k;
                sec.to_prover_verifier_configs()
                    .map(|(pv, vc)| (initial_k, pv, vc))
            });
            (key, parsed)
        })
        .collect()
});

fn parsed_config_for(
    m: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Option<Result<(ProverConfig, VerifierConfig), String>> {
    let (_, parsed) = PARSED_EMBEDDED_CONFIGS
        .iter()
        .find(|&&(key, _)| key == (m, profile))?;
    Some(match parsed {
        Err(e) => Err(e.clone()),
        Ok((initial_k, pv, vc)) => {
            if *initial_k != log_batch_size {
                Err(format!(
                    "embedded config for (m={m}, profile={}) has \
                     initial_k={initial_k} but caller requested log_batch_size={log_batch_size}",
                    profile.as_str()
                ))
            } else {
                Ok((pv.clone(), vc.clone()))
            }
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
    let m = log_n + crate::pcs::LOG_PACKING;
    let (pv, _) = parsed_config_for(m, log_batch_size, profile).ok_or_else(|| {
        format!(
            "no security config registered for (m={m}, profile={}). \
             Add a TOML at configs/ligerito/m{m}_{}.toml and register it in \
             EMBEDDED_CONFIGS, or call default_config explicitly for ad-hoc shapes.",
            profile.as_str(),
            profile.as_str(),
        )
    })??;
    Ok(pv)
}

/// Verifier-side counterpart to [`prover_config_for`]. Same strict lookup.
pub fn verifier_config_for(
    log_n: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Result<VerifierConfig, String> {
    let m = log_n + crate::pcs::LOG_PACKING;
    let (_, vc) = parsed_config_for(m, log_batch_size, profile).ok_or_else(|| {
        format!(
            "no security config registered for (m={m}, profile={})",
            profile.as_str()
        )
    })??;
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
    /// - `Fast`:   JohnsonOod, rate 1/2, η = 0.04, 100 bits per round.
    /// - `Slim`:   JohnsonOod, rate 1/4, η = 0.02, 16-bit query grinding at
    ///             every level, 100 bits per round.
    /// - `Secure`: Udr, rate 1/2, ε* = 1e-3, 120 bits per round.
    pub fn derive_profile(m: usize, profile: LigeritoProfile) -> Result<Self, String> {
        // Fast trades a few more queries for much cheaper proximity-gap
        // grinding. Slim retains its proof-size-oriented 0.02 slack.
        let johnson_eta = match profile {
            LigeritoProfile::Fast => 0.04,
            LigeritoProfile::Slim => 0.02,
            LigeritoProfile::Secure => 0.0, // unused in the UDR branch
        };
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
                    paper_per_query_bits(rate, johnson_eta)
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
                    paper_per_query_bits(rate, johnson_eta)
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
                    let eps_pg = ANALYSIS_LOG_Q - paper_johnson_log_a(rate, johnson_eta, cols, ilv);
                    let mu = cols + ilv;
                    let ood_samples = if i == 0 {
                        0 // bound by the opening's own evaluation claim
                    } else {
                        (1..=8usize)
                            .find(|&s| paper_ood_bits(rate, johnson_eta, mu, s) >= t)
                            .ok_or_else(|| {
                                format!("L{i}: no OOD sample count reaches {t:.1} bits")
                            })?
                    };
                    let eps_ood = paper_ood_bits(rate, johnson_eta, mu, ood_samples);
                    (
                        SoundnessRegime::JohnsonOod,
                        Some(johnson_eta),
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
    let mut len = cur.len();
    for &r in rs {
        let half = len / 2;
        // Char-2: even*(1+r)+odd*r = even + r*(even+odd). One mul per pair.
        // Fold in place: the write index i is always ≤ the read indices
        // 2i, 2i+1, so no round's reads are clobbered by its own or any
        // earlier round's writes. One allocation per call instead of one
        // fresh Vec per challenge; identical values, less allocator traffic.
        for i in 0..half {
            let e0 = cur[2 * i];
            let e1 = cur[2 * i + 1];
            cur[i] = e0 + r * (e0 + e1);
        }
        len = half;
    }
    cur.truncate(len);
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
/// Only depends on `log_n`, so callers cache.
///
/// Ranked default memoizes the table per `log_n` in a process-wide
/// `OnceLock` array: the result is a pure function of `log_n` and every
/// call site consumes it by shared borrow only, so this is bit-identical
/// to the previous fresh-Vec computation. The worker's untimed warm-up
/// proves populate the cache; the timed open/verify calls that follow
/// become table lookups with no per-call heap traffic and no recompute.
pub(crate) fn eval_sk_at_vks(log_n: usize) -> &'static [F128] {
    /// Reachable geometries never exceed `log_n = 20` (BLAKE3 batches up to
    /// 2^20 in this challenge); 33 slots cover every power-of-two case plus
    /// headroom. Anything larger falls through to a fresh computation.
    const CACHE_LEN: usize = 33;
    static CACHE: [std::sync::OnceLock<Vec<F128>>; CACHE_LEN] =
        [const { std::sync::OnceLock::new() }; CACHE_LEN];

    fn compute(log_n: usize) -> Vec<F128> {
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

    if log_n < CACHE_LEN {
        return CACHE[log_n].get_or_init(|| compute(log_n));
    }
    let boxed: Box<[F128]> = compute(log_n).into();
    Box::leak(boxed)
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
    let wide = open_basis_x4_enabled();
    for k in 0..log_n {
        let s_at_x = sks_at_x[k];
        let current_len = 1 << k;
        if wide {
            // Doubling step: the upper half is `s_at_x` times the lower half,
            // element by element, with the two halves disjoint. Same shape as
            // `eq_expand_block_x4`, which writes each product directly.
            let (current, rest) = basis[..2 * current_len].split_at_mut(current_len);
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            // SAFETY: target features are cfg-guaranteed and `split_at_mut`
            // hands back two disjoint halves of equal length.
            unsafe {
                eq_expand_block_x4(rest, current, s_at_x);
            }
            #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
            for (dst, &src) in rest.iter_mut().zip(current.iter()) {
                *dst = s_at_x * src;
            }
        } else {
            for i in 0..current_len {
                basis[i + current_len] = s_at_x * basis[i];
            }
        }
    }
}

/// Ranked default runs the [`evaluate_scaled_basis_inplace`] doubling step
/// through the four-lane `eq_expand_block_x4` leaf: every entry of the upper
/// half is an independent product of the same scalar with one lower-half
/// entry, so widening computes the identical set of canonical products.
/// `FLOCK_NO_OPEN_BASIS_X4=1` restores the per-element scalar doubling loop.
/// Read once per call, outside the doubling loop.
fn open_basis_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_BASIS_X4").is_none());
    *ON
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

    // Per-thread chunked accumulation: each thread accumulates a partial
    // basis_poly (length n) and a partial enforced_sum, then we reduce.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n_queries + n_threads - 1) / n_threads.max(1);

    let mut partials: Vec<(Vec<F128>, F128)> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(n_queries);
            if start >= end {
                return (vec![F128::ZERO; n], F128::ZERO);
            }
            let mut accum_basis = vec![F128::ZERO; n];
            // Per-thread scratch reused across this chunk's queries.
            let mut local_basis = vec![F128::ZERO; n];
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
        })
        .collect();

    // Reuse worker zero's complete partial as the output buffer. Starting from
    // partial zero and adding workers 1.. in order is byte-identical to
    // allocating/zeroing another `n`-element Vec and adding workers 0.., while
    // deleting one full allocation, zero-fill, read and add pass.
    debug_assert!(!partials.is_empty());
    let (mut basis_poly, mut enforced_sum) = partials.remove(0);
    for (_, ls) in partials.iter() {
        enforced_sum += *ls;
    }
    // Below the floor a rayon dispatch costs more than the reduce itself.
    const REDUCE_PAR_FLOOR: usize = 1 << 12;
    if induce_sched_enabled() && n >= REDUCE_PAR_FLOOR && partials.len() > 1 {
        // `chunk` floored at REDUCE_PAR_FLOOR is TWO tasks at the ranked L2
        // induce (n = 2^13, 16 threads) — fourteen cores idle through the whole
        // reduce. The lower floor keeps each task at ≥ 8 KiB of output while
        // letting `n / threads` actually reach the thread count. Per-slot
        // summation order (partial 0, 1, … in thread order) is unchanged, so
        // the result is bit-identical.
        let reduce_floor = if open_fill_enabled() {
            1usize << 9
        } else {
            REDUCE_PAR_FLOOR
        };
        let chunk = (n / rayon::current_num_threads().max(1)).max(reduce_floor);
        basis_poly
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let base = ci * chunk;
                let len = out.len();
                for (lb, _) in partials.iter() {
                    for (acc, &v) in out.iter_mut().zip(lb[base..base + len].iter()) {
                        *acc += v;
                    }
                }
            });
    } else {
        for (lb, _) in partials.iter() {
            for (acc, &v) in basis_poly.iter_mut().zip(lb.iter()) {
                *acc += v;
            }
        }
    }

    (basis_poly, enforced_sum)
}

/// 4-lane AVX-512 transpose butterfly: `s = a ⊕ b; a' = s; b' = t·s ⊕ b`.
///
/// Mirrors the forward-NTT x86 kernel shape: broadcast `t`, one `ghash_mul_x4`
/// per 4-lane group, XOR for the sum. Field-identical to the scalar loop
/// (`ghash_mul_x4` is the canonical mod-p product, cross-checked against
/// `ghash_mul_karatsuba_barrett` in the field tests).
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq` (cfg-gated at call site). `top` and
/// `bot` must have equal length.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn transpose_butterfly_avx512(top: &mut [F128], bot: &mut [F128], t: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    // SAFETY: caller carries the target features; slice bounds hold.
    unsafe {
        let tb = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
        let tb_x64 = ghash_shift64_x4(tb);
        let lanes = top.len() & !3;
        let mut i = 0;
        while i < lanes {
            let va = _mm512_loadu_si512(top.as_ptr().add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(bot.as_ptr().add(i) as *const __m512i);
            let vs = _mm512_xor_si512(va, vb);
            _mm512_storeu_si512(top.as_mut_ptr().add(i) as *mut __m512i, vs);
            let nb = _mm512_xor_si512(vb, ghash_mul_x4_split(vs, tb, tb_x64));
            _mm512_storeu_si512(bot.as_mut_ptr().add(i) as *mut __m512i, nb);
            i += 4;
        }
        while i < top.len() {
            // F128 addition IS XOR (GF(2^128)).
            let s = top[i] + bot[i];
            top[i] = s;
            bot[i] = t * s + bot[i];
            i += 1;
        }
    }
}

/// Scalar/vector transpose butterfly on two equal-length halves:
/// `s = a + b; a' = s; b' = t·s + b`. Thin cfg wrapper so the schedulers
/// below stay readable.
#[inline(always)]
fn transpose_butterfly(top_h: &mut [F128], bot: &mut [F128], t: F128) {
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    // SAFETY: target features cfg-guaranteed; halves are equal-length.
    unsafe {
        transpose_butterfly_avx512(top_h, bot, t)
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
    {
        for (a_ref, b_ref) in top_h.iter_mut().zip(bot.iter_mut()) {
            let a = *a_ref;
            let b = *b_ref;
            let s = a + b;
            *a_ref = s;
            *b_ref = t * s + b;
        }
    }
}

/// Additive-NTT twiddle `(layer, 0)` is the empty span sum, hence zero.
/// The first column group in every blocked transpose pass-B layer therefore
/// needs only `top ^= bot`; `bot` is unchanged.  Resolve the diagnostics gate
/// before entering the Rayon closure so the ordinary butterfly loop remains
/// branch-free.
#[inline]
fn tnt_pass_b_zero_selected(disabled: bool) -> bool {
    !disabled
}

fn tnt_pass_b_zero_enabled() -> bool {
    static DISABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_TNT_ZERO_TWIDDLE").is_some());
    tnt_pass_b_zero_selected(*DISABLED)
}

/// Peel pass B's zero-twiddle column group out of the generic butterfly
/// closure. Keeping this leaf out of line is deliberate: a global zero test
/// inside `transpose_butterfly` inflated both hot transpose closures enough
/// to outweigh the deleted products on Sapphire Rapids.
#[inline(never)]
fn transpose_pass_b_block_zero_xor(
    cols: &mut [&mut [F128]],
    stride: usize,
    _layer: usize,
    _ntt: &AdditiveNttF128,
) {
    for u in 0..stride {
        let (lo, hi) = cols.split_at_mut(u + stride);
        let top = &mut *lo[u];
        let bot = &*hi[0];
        debug_assert_eq!(top.len(), bot.len());
        #[cfg(target_feature = "avx512f")]
        {
            use core::arch::x86_64::*;
            // SAFETY: avx512f is cfg-guaranteed and both slices have equal
            // length. F128 addition is bitwise XOR.
            unsafe {
                let lanes = top.len() & !3;
                let mut i = 0;
                while i < lanes {
                    let a = _mm512_loadu_si512(top.as_ptr().add(i).cast::<__m512i>());
                    let b = _mm512_loadu_si512(bot.as_ptr().add(i).cast::<__m512i>());
                    _mm512_storeu_si512(
                        top.as_mut_ptr().add(i).cast::<__m512i>(),
                        _mm512_xor_si512(a, b),
                    );
                    i += 4;
                }
                while i < top.len() {
                    top[i] += bot[i];
                    i += 1;
                }
            }
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            for (a, &b) in top.iter_mut().zip(bot.iter()) {
                *a += b;
            }
        }
    }
}

/// Exact kill-switch arm for the peeled group. The twiddle is still obtained
/// from the table and the established general multiply is used; only the
/// loop's placement moved out of the hot closure.
#[inline(never)]
fn transpose_pass_b_block_zero_general(
    cols: &mut [&mut [F128]],
    stride: usize,
    layer: usize,
    ntt: &AdditiveNttF128,
) {
    let t = ntt.twiddle(layer, 0);
    for u in 0..stride {
        let (lo, hi) = cols.split_at_mut(u + stride);
        transpose_butterfly(lo[u], hi[0], t);
    }
}

/// `FLOCK_NO_OPEN_TNTT_BLOCK=1` restores the incumbent one-rayon-region-per-
/// layer transpose sweep ([`transpose_forward_ntt_dense_layers_per_layer`]).
/// Read once per process; default ON (the ranked worker clears its env).
fn tntt_block_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_TNTT_BLOCK").is_none());
    *ON
}

/// Dense transpose sweep over forward layers `0..top` (applied in reverse
/// layer order). Shared by [`transpose_forward_ntt`] and the sparse-prefix
/// tail; `data.len()` must be `2^log_d` with `top <= log_d`.
///
/// Two schedules, byte-identical outputs (see
/// `transpose_forward_ntt_blocked_matches_per_layer`):
///  * [`transpose_forward_ntt_dense_layers_blocked`] (default) — two fused
///    cache passes,
///  * [`transpose_forward_ntt_dense_layers_per_layer`] (kill switch) — the
///    incumbent one parallel sweep per layer.
fn transpose_forward_ntt_dense_layers(ntt: &AdditiveNttF128, data: &mut [F128], top: usize) {
    if tntt_block_enabled() {
        transpose_forward_ntt_dense_layers_blocked(ntt, data, top);
    } else {
        transpose_forward_ntt_dense_layers_per_layer(ntt, data, top);
    }
}

/// Incumbent schedule: one parallel region per layer (and, for layers with
/// fewer blocks than threads, one region PER BLOCK). Kept as the kill-switch
/// arm and as the oracle for the blocked schedule's equality test.
fn transpose_forward_ntt_dense_layers_per_layer(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    top: usize,
) {
    use rayon::prelude::*;
    let log_d = data.len().trailing_zeros() as usize;
    debug_assert!(top <= log_d);
    let n_threads = rayon::current_num_threads().max(1);
    for layer in (0..top).rev() {
        let num_blocks = 1usize << layer;
        let block_size = 1usize << (log_d - layer);
        let bsh = block_size >> 1;
        if num_blocks >= n_threads {
            data.par_chunks_mut(block_size)
                .enumerate()
                .for_each(|(block, chunk)| {
                    let t = ntt.twiddle(layer, block);
                    let (top_h, bot) = chunk.split_at_mut(bsh);
                    transpose_butterfly(top_h, bot, t);
                });
        } else {
            for block in 0..num_blocks {
                let t = ntt.twiddle(layer, block);
                let chunk = &mut data[block * block_size..(block + 1) * block_size];
                let (top_h, bot) = chunk.split_at_mut(bsh);
                // Few huge blocks: split each block's span into segments so
                // the vector kernel still fills the cores.
                const SEG: usize = 4096;
                top_h
                    .par_chunks_mut(SEG)
                    .zip(bot.par_chunks_mut(SEG))
                    .for_each(|(a, b)| transpose_butterfly(a, b, t));
            }
        }
    }
}

/// Cache-blocked schedule: the `top` layers are applied in TWO fused passes
/// instead of one pass per layer.
///
/// Why: the transposed sweep runs layers `top-1 … 0`, i.e. pairing distances
/// `2^(log_d-top) … 2^(log_d-1)` — *increasing*. Layers `≥ split` mix only
/// **inside** a `2^(log_d-split)`-element chunk; layers `< split` mix only
/// **across** those chunks at a fixed offset. So the whole sweep is
///   (a) `top-split` layers, chunk-local: one rayon region, one read+write
///       pass, each chunk sized to stay in L2 for all of its layers;
///   (b) `split` layers, cross-chunk: one rayon region over tiles of the
///       chunk offset, each tile holding its `2^split` strided columns in L2
///       for all of its layers.
/// The incumbent made `top` full DRAM/L3 passes and — for layers whose block
/// count fell below the thread count — re-entered the rayon pool once per
/// block. At the ranked m=32 open's L0 induce (`log_d=20`, `top=12`) that was
/// 12 passes over 16 MiB and 23 rayon regions; this is 2 and 2.
///
/// Arithmetic is unchanged: each butterfly sees exactly the operands, twiddle
/// and layer order of the incumbent (layers commute only within a group, and
/// the groups are applied in the same relative order), so the output is
/// bit-identical.
fn transpose_forward_ntt_dense_layers_blocked(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    top: usize,
) {
    use rayon::prelude::*;
    let log_d = data.len().trailing_zeros() as usize;
    debug_assert!(top <= log_d);
    if top == 0 {
        return;
    }
    let n_threads = rayon::current_num_threads().max(1);
    // Chunk target: 2^CHUNK_LOG F128 (= 1 MiB at CHUNK_LOG=16) so a chunk
    // stays in L2 across all of pass (a)'s layers; but never so few chunks that
    // the pool starves. Swept 13..17 at the ranked L0 shape (log_d=20, top=12,
    // 16 threads): Zen 5 wants the large end (17: 1.8 ms, 15: 2.6, 13: n/m) and
    // Zen 3 the small end (13: 2.5 ms, 15: 2.6, 17: 3.1); 15 was the AMD
    // compromise. Official SPR rejected CHUNK_LOG=17 (−0.62% tntt17-2).
    //
    // Ranked L0 + 16 threads: split = max(20-CHUNK_LOG, 4).min(12), so
    // CHUNK_LOG=16 and 17 share split=4 (pass (a) chunk = 1 MiB both). Only
    // pass (b) tile doubles: 16 → 1 MiB resident, 17 → 2 MiB (= full SPR L2).
    // 16 keeps the Zen-5 split without the 2 MiB tile that evicted L2 on SPR.
    const CHUNK_LOG: usize = 16;

    let split = log_d
        .saturating_sub(CHUNK_LOG)
        .max(ceil_log2(n_threads))
        .min(top);

    // ---- (a) chunk-local layers `split .. top`, applied high → low. ----
    if top > split {
        let chunk_len = 1usize << (log_d - split);
        data.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(c, chunk)| {
                for layer in (split..top).rev() {
                    let nb = 1usize << (layer - split); // blocks inside this chunk
                    let block_size = 1usize << (log_d - layer);
                    let bsh = block_size >> 1;
                    for jb in 0..nb {
                        let t = ntt.twiddle(layer, (c << (layer - split)) + jb);
                        let (top_h, bot) =
                            chunk[jb * block_size..(jb + 1) * block_size].split_at_mut(bsh);
                        transpose_butterfly(top_h, bot, t);
                    }
                }
            });
    }

    // ---- (b) cross-chunk layers `split-1 .. 0`, applied high → low. ----
    // Position `p = x + j·seg` with `x < seg = 2^(log_d-split)`, `j < 2^split`.
    // Layer `l < split` pairs `j` with `j ^ 2^(split-1-l)` at the same `x`, and
    // its block index is `j >> (split-l)`. Fixing a tile of `x` values and
    // holding that tile's `2^split` columns resident applies all `split`
    // layers in one pass.
    if split > 0 {
        let nseg = 1usize << split;
        let seg = 1usize << (log_d - split);
        // Tile so the resident set is ~2^CHUNK_LOG F128 across all columns.
        let mut tile = ((1usize << CHUNK_LOG) / nseg).max(1).min(seg);
        // Starvation floor: the tile loop is pass (b)'s ONLY parallel
        // decomposition, so a cache-sized tile that leaves fewer tiles than
        // threads simply idles cores. Cap the tile at `seg >> log2(threads)`
        // (a power of two, so it still divides `seg` exactly). At the ranked L0
        // induce (log_d=20, seg=2^16, 16 threads) the cap EQUALS the tuned tile
        // — L0's resident set is unchanged — while the L1 induce (log_d=18,
        // seg=2^14) goes from 4 tiles to 16.
        if open_fill_enabled() {
            let cap = seg >> ceil_log2(n_threads);
            // Only when the segment can actually supply thread-many tiles of a
            // sane size; tiny domains keep the incumbent single-tile schedule.
            if cap >= 64 {
                tile = tile.min(cap);
            }
        }
        let ntiles = seg / tile;
        let mut tiles: Vec<Vec<&mut [F128]>> =
            (0..ntiles).map(|_| Vec::with_capacity(nseg)).collect();
        for part in data.chunks_mut(seg) {
            for (ti, col) in part.chunks_mut(tile).enumerate() {
                tiles[ti].push(col);
            }
        }
        // One indirect call per layer selects the peeled block-0 leaf. The
        // production arm is XOR-only; the kill arm is the exact general
        // butterfly. The much hotter j>=2*stride loop below is shared and has
        // no zero test or duplicated body.
        type BlockZeroFn = fn(&mut [&mut [F128]], usize, usize, &AdditiveNttF128);
        let block_zero: BlockZeroFn = if tnt_pass_b_zero_enabled() {
            transpose_pass_b_block_zero_xor
        } else {
            transpose_pass_b_block_zero_general
        };
        tiles.into_par_iter().for_each(|mut cols| {
            for b in 0..split {
                let layer = split - 1 - b;
                let stride = 1usize << b;
                block_zero(&mut cols, stride, layer, ntt);
                let mut j = stride << 1;
                while j < nseg {
                    for u in j..j + stride {
                        let t = ntt.twiddle(layer, u >> (b + 1));
                        let (lo, hi) = cols.split_at_mut(u + stride);
                        transpose_butterfly(lo[u], hi[0], t);
                    }
                    j += stride << 1;
                }
            }
        });
    }
}

/// Transposed forward additive NTT, `Fᵀ`, in place over `2^log_d` coefficients.
/// Forward butterfly is `M=[[1,t],[1,t+1]]`; transpose `Mᵀ=[[1,1],[t,t+1]]` is
/// `s=a+b; top=s; bot=t·s+b`, applied in **reverse** layer order. (Baseline:
/// one parallel sweep per layer.)
fn transpose_forward_ntt(ntt: &AdditiveNttF128, data: &mut [F128], log_d: usize) {
    debug_assert_eq!(data.len(), 1usize << log_d);
    debug_assert!(log_d <= ntt.log_domain_size());
    transpose_forward_ntt_dense_layers(ntt, data, log_d);
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

    let mut enforced_sum = F128::ZERO;
    for i in 0..n_queries {
        let dot: F128 = opened_rows[i]
            .iter()
            .zip(eq.iter())
            .map(|(&r, &e)| r * e)
            .fold(F128::ZERO, |a, v| a + v);
        enforced_sum += dot * alpha_pows[i];
    }

    let mut coeffs = if log_block == 0 {
        let mut c = vec![F128::ZERO; block_len];
        for i in 0..n_queries {
            c[queries[i]] += alpha_pows[i];
        }
        c
    } else {
        let ntt = AdditiveNttF128::standard(log_block);
        transpose_forward_ntt_sparse(&ntt, queries, &alpha_pows, log_block)
    };
    coeffs.truncate(n);
    (coeffs, enforced_sum)
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
/// `FLOCK_NO_OPEN_EQ_SPLIT=1` restores the serial [`build_eq_table`] for the
/// open's OOD tables. Read once per process; default ON (the ranked worker
/// clears its env).
fn eq_split_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_EQ_SPLIT").is_none());
    *ON
}

/// Parallel `eq(point, ·)` table for the open's OOD binding.
///
/// [`build_eq_table`] is a serial doubling recurrence: the L1 OOD sample at the
/// ranked m=32 shape needs `2^19` entries, i.e. half a million dependent GHASH
/// multiplies on ONE core. But `eq` is a pure tensor product and F128
/// arithmetic is exact (`v·(1+r) = v + v·r` is an identity, not a rounding
/// choice), so the table factors: with `i = i_lo + 2^h · i_hi`,
///   `eq[i] = eq(point[..h])[i_lo] · eq(point[h..])[i_hi]`.
/// Build the two small factors serially and expand in ONE rayon region — the
/// products are the same field elements in the same slots, so the table is
/// bit-identical (see `eq_split_matches_serial`).
///
/// `h` is chosen so the expansion has ~4 blocks per thread; below
/// `SPLIT_MIN_LOG` the serial recurrence wins.
/// `FLOCK_NO_EQ_SPLIT_X4=1` restores the scalar expansion loop of
/// [`build_eq_table_split`] (exact same-binary A/B; `ghash_mul_x4` is the same
/// canonical mod-p product as the scalar `Mul`, cross-checked in the field
/// tests, so the table is bit-identical).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
fn eq_split_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_EQ_SPLIT_X4").is_none());
    *ON
}

/// Four-lane expansion block of [`build_eq_table_split`]: `out[i] = lo[i]·e`.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq` (cfg-gated at the call site). `out` and
/// `lo` must have equal length.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn eq_expand_block_x4(out: &mut [F128], lo: &[F128], e: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;
    debug_assert_eq!(out.len(), lo.len());
    // SAFETY: caller carries the target features; the slices are equal-length
    // and every offset below stays inside both.
    unsafe {
        let eb = _mm512_broadcast_i32x4(_mm_set_epi64x(e.hi as i64, e.lo as i64));
        let eb_x64 = ghash_shift64_x4(eb);
        let lanes = out.len() & !3;
        let mut i = 0usize;
        while i < lanes {
            let v = _mm512_loadu_si512(lo.as_ptr().add(i) as *const __m512i);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i) as *mut __m512i,
                ghash_mul_x4_split(v, eb, eb_x64),
            );
            i += 4;
        }
        while i < out.len() {
            out[i] = lo[i] * e;
            i += 1;
        }
    }
}

fn build_eq_table_split(point: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    // The incumbent floor (17) let the L2 and L3 OOD tables (d = 16, 13) fall
    // back to the serial doubling recurrence: 2^d − 1 dependent GHASH
    // multiplies on ONE core, in a phase whose other two passes (round message
    // + glue) already spread over sixteen. 2^13 still leaves ≥ 128 entries per
    // expansion block, well above rayon's dispatch break-even.
    const SPLIT_MIN_LOG: usize = 17;
    const SPLIT_MIN_LOG_FILL: usize = 13;
    let split_min = if open_fill_enabled() {
        SPLIT_MIN_LOG_FILL
    } else {
        SPLIT_MIN_LOG
    };
    let d = point.len();
    if !eq_split_enabled() || d < split_min {
        return build_eq_table(point);
    }
    let n_threads = rayon::current_num_threads().max(1);
    // Number of expansion blocks = 2^(d-h); want ≥ 4 per thread but keep each
    // block big enough to amortize the per-block loop.
    let log_blocks = (ceil_log2(n_threads) + 2).min(d - 1);
    let h = d - log_blocks;
    let lo = build_eq_table(&point[..h]);
    let hi = build_eq_table(&point[h..]);
    debug_assert_eq!(lo.len(), 1usize << h);
    debug_assert_eq!(hi.len(), 1usize << log_blocks);
    let mut out = crate::alloc_uninit_vec::<F128>(1usize << d);
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    let x4 = eq_split_x4_enabled();
    out.par_chunks_mut(1usize << h)
        .zip(hi.par_iter())
        .for_each(|(chunk, &e)| {
            // Every slot of `chunk` is written here — upholds
            // `alloc_uninit_vec`'s write-before-read contract.
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            if x4 {
                // SAFETY: target features cfg-guaranteed; `chunk` and `lo`
                // are both `2^h` long.
                unsafe { eq_expand_block_x4(chunk, &lo, e) };
                return;
            }
            for (o, &l) in chunk.iter_mut().zip(lo.iter()) {
                *o = l * e;
            }
        });
    out
}

/// `FLOCK_NO_OPEN_INDUCE_SCHED=1` restores the incumbent `induce_sumcheck_poly`
/// scheduling: the dense-vs-Fᵀ-NTT crossover constant `C = 4`, the hard-wired
/// dense arm at recursion levels ≥ 1, and the serial cross-thread reduce.
/// Read once per process; default ON (the ranked worker clears its env).
pub(crate) fn induce_sched_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_INDUCE_SCHED").is_none());
    *ON
}

/// Ranked default computes the dense arm's `eval_sk_at_vks` table INSIDE
/// [`induce_sumcheck_poly_auto`]'s dense arm instead of eagerly at the call
/// sites: the Fᵀ-NTT arm never reads it, and both ranked prover sites that
/// can take the NTT arm were paying the O(log²) serial GHASH chain (plus an
/// allocation) inside the Fiat–Shamir critical path for a table that was
/// then dropped unread. Bit-identical output: the same table is built from
/// the same `log_msg_cols` whenever the dense arm actually runs.
/// `FLOCK_NO_INDUCE_LAZY_SKS=1` restores the eager call-site computation.
/// Read once per process; default ON.
fn induce_lazy_sks_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_INDUCE_LAZY_SKS").is_none());
    *ON
}

/// Crossover constant `C` in the [`induce_sumcheck_poly_auto`] dispatch rule
/// `n_queries > C · 2^log_inv_rate · log_block`.
///
/// The incumbent `C = 4` priced the Fᵀ-NTT arm at ~2× the dense arm per op
/// ("memory-bound, multi-pass") — true when the transposed sweep made one
/// full DRAM pass per layer. [`transpose_forward_ntt_dense_layers_blocked`]
/// cut that to two passes total (measured 20.2 → 2.3 ms at the ranked L0
/// shape), so the NTT arm is now the cheaper one from roughly
/// `n_queries > 2^log_inv_rate · log_block` upward; `C = 1` keeps a margin.
fn induce_ntt_crossover_c() -> usize {
    if induce_sched_enabled() { 1 } else { 4 }
}

/// `sks_vks` feeds ONLY the dense arm (the Fᵀ-NTT arm never reads it); pass
/// `None` to have the dense arm build the table itself when — and only
/// when — it is actually taken. `Some` keeps the caller's precomputed table
/// (the incumbent shape, and the `FLOCK_NO_INDUCE_LAZY_SKS` rollback).
pub(crate) fn induce_sumcheck_poly_auto(
    log_msg_cols: usize,
    log_inv_rate: usize,
    sks_vks: Option<&[F128]>,
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    let log_block = log_msg_cols + log_inv_rate;
    let use_ntt = log_msg_cols >= 12
        && queries.len() > induce_ntt_crossover_c() * (1usize << log_inv_rate) * log_block.max(1);
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
        let computed;
        let sks_vks = match sks_vks {
            Some(table) => table,
            None => {
                computed = eval_sk_at_vks(log_msg_cols);
                &computed
            }
        };
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

/// `FLOCK_NO_INDUCE_FUSED_DENSIFY=1` restores the two-pass densify of the
/// sparse-prefix NTT: a serial `vec![F128::ZERO; 2^log_d]` zero-fill followed
/// by a serial scatter of the active windows. Exact same-binary A/B; resolved
/// once per process.
fn induce_fused_densify_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_INDUCE_FUSED_DENSIFY").is_none());
    *ON
}

/// Scatter the processed active windows into a dense `nwin`-slot table so the
/// densify pass can be indexed (and therefore parallelised) by window number.
/// `nwin` is `2^(log_d - k)`; every `w` in `processed` is `< nwin` and unique
/// (they are the keys of the grouping `HashMap`).
fn window_slots(nwin: usize, processed: Vec<(usize, Vec<F128>)>) -> Vec<Option<Vec<F128>>> {
    let mut slots: Vec<Option<Vec<F128>>> = (0..nwin).map(|_| None).collect();
    for (w, buf) in processed {
        debug_assert!(w < nwin);
        debug_assert!(slots[w].is_none(), "window {w} densified twice");
        slots[w] = Some(buf);
    }
    slots
}

/// Fused zero-fill + densify: ONE parallel pass over the `2^k`-aligned windows
/// writes each of the `n / 2^k` windows exactly once — an active window gets
/// its processed buffer, an inactive window gets zeros.
///
/// Byte-identical to the incumbent `vec![F128::ZERO; n]` + serial scatter:
/// the windows partition `0..n`, the active arm copies exactly the bytes the
/// scatter copied, and the inactive arm writes exactly the zeros the
/// allocation left in place. The buffer therefore starts UNINITIALIZED
/// ([`crate::alloc_uninit_vec`]'s write-before-read contract is discharged by
/// the partition), which deletes the serial 16 MiB zero pass and spreads the
/// first-touch page faults across the rayon pool instead of one thread.
fn densify_windows_fused(n: usize, k: usize, slots: Vec<Option<Vec<F128>>>) -> Vec<F128> {
    use rayon::prelude::*;
    debug_assert_eq!(slots.len(), n >> k);
    let mut data: Vec<F128> = crate::alloc_uninit_vec(n);
    data.par_chunks_mut(1usize << k)
        .zip(slots.into_par_iter())
        .for_each(|(dst, src)| match src {
            Some(buf) => dst.copy_from_slice(&buf),
            None => dst.fill(F128::ZERO),
        });
    data
}

/// Dense `2^k` transpose for one sparse-prefix window. This is the incumbent
/// path and remains the exact collision fallback for the singleton shortcut.
#[inline]
fn transpose_forward_ntt_window_dense(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    w: usize,
    mut buf: Vec<F128>,
) -> (usize, Vec<F128>) {
    for s in 0..k {
        let layer = log_d - 1 - s;
        let bsh = 1usize << s;
        let block_size = bsh << 1;
        let nblocks = (1usize << k) / block_size;
        for jb in 0..nblocks {
            let t = ntt.twiddle(layer, (w << (k - s - 1)) + jb);
            let base = jb * block_size;
            let (top_h, bot) = buf[base..base + block_size].split_at_mut(bsh);
            transpose_butterfly(top_h, bot, t);
        }
    }
    (w, buf)
}

/// Ranked default skips the singleton window buffers' zero-fill (every slot
/// is written exactly once before any read — see [`expand_singleton_into`]'s
/// write-completeness contract; [`take_singleton_buf`] is the allocator).
/// Identical output bytes: the zeroed slots were never read.
/// `FLOCK_NO_LIG_SINGLETON_UNINIT=1` restores the incumbent
/// `vec![F128::ZERO; _]` for exact same-binary A/B. Read once per process;
/// default ON.
fn singleton_uninit_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_SINGLETON_UNINIT").is_none());
    *ON
}

/// One `2^k`-slot window buffer for the singleton / multi-singleton shortcut.
/// [`expand_singleton_into`] is write-complete, so the buffer may start
/// uninitialized; `FLOCK_NO_LIG_SINGLETON_UNINIT=1` restores the incumbent
/// zero-fill. Output bytes are identical either way.
fn take_singleton_buf(k: usize) -> Vec<F128> {
    if singleton_uninit_enabled() {
        crate::alloc_uninit_vec::<F128>(1usize << k)
    } else {
        vec![F128::ZERO; 1usize << k]
    }
}

/// Expand one nonzero at local index `p` through the window's `k` transpose
/// layers, into `out` (length `2^k`). Before layer `s`, only one `2^s` half
/// can be live; either source orientation produces the same top half
/// (`current`) and a bottom half scaled by `twiddle + bit_s`. Thus the work
/// is `1+2+...+2^(k-1)` products, versus `k·2^(k-1)` for the zero-padded
/// dense window.
///
/// **Write-complete**: `out[0]` is stored first and step `s` writes
/// `out[2^s..2^(s+1)]` (both the AVX-512 and scalar arms store the product
/// without reading the destination), so over `s ∈ 0..k` every slot of `out`
/// is written exactly once. `out` may therefore be UNINITIALIZED
/// ([`crate::alloc_uninit_vec`]'s write-before-read contract).
fn expand_singleton_into(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    w: usize,
    p: usize,
    value: F128,
    out: &mut [F128],
) {
    assert!(k > 0 && k <= log_d);
    assert!(p < 1usize << k);
    assert_eq!(out.len(), 1usize << k);
    out[0] = value;
    let mut len = 1usize;
    for s in 0..k {
        let layer = log_d - 1 - s;
        let jb = p >> (s + 1);
        let mut scale = ntt.twiddle(layer, (w << (k - s - 1)) + jb);
        if (p >> s) & 1 != 0 {
            scale += F128::ONE;
        }
        let (current, rest) = out[..2 * len].split_at_mut(len);
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        // SAFETY: target features are cfg-guaranteed and the two halves have
        // equal length. The kernel writes `out[i] = lo[i]·e` without reading
        // the destination, so it is exactly the write-complete store this
        // expansion needs (the destination half may be uninitialized).
        unsafe {
            eq_expand_block_x4(&mut rest[..len], current, scale)
        }
        #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
        for (dst, &src) in rest[..len].iter_mut().zip(current.iter()) {
            *dst = src * scale;
        }
        len <<= 1;
    }
}

/// One-nonzero window as a `(w, buf)` task result. Test oracle / densify
/// scatter shape; production expands in place through [`expand_singleton_into`].
#[allow(dead_code)] // Retained singleton expansion oracle.
fn transpose_forward_ntt_window_singleton(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    w: usize,
    p: usize,
    value: F128,
) -> (usize, Vec<F128>) {
    let mut buf = take_singleton_buf(k);
    expand_singleton_into(ntt, log_d, k, w, p, value, &mut buf);
    (w, buf)
}

/// `FLOCK_NO_LIG_INDUCE_SINGLETON=1` disables the sparse-induce singleton /
/// multi-singleton window shortcut — and, through [`induce_window_k`], drops
/// the window width back to the dense-era `k = 8` — restoring the incumbent
/// dense-window path byte-for-byte. Wider `k` and `nnz ≤ k/2` are one
/// mechanism: either alone is a loss, so they ride one gate.
fn lig_induce_singleton_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_INDUCE_SINGLETON").is_none());
    *ON
}

/// Per-task window-buffer memory cap: `2^12` F128 = 64 KiB (× 2 with the
/// multi-singleton scratch = 128 KiB per task, × 16 threads = 2 MiB live).
const INDUCE_K_MAX_MEM: usize = 12;

/// Window width `k` for the sparse-prefix transpose.
///
/// Small domains (`log_d < 12`) and empty query sets take the scatter + full
/// dense transform (`k = 0`). Without the singleton shortcut every active
/// window pays the dense `k·2^(k-1)` products — the dense-window optimum
/// stays at the incumbent `k = 8`. With the shortcut the window cost is
/// `nnz·(2^k−1)`, so the optimum moves: pick the `k ∈ 8..=hi` minimizing
///   `Q·(2^k − 1) + (log_d − k)·2^(log_d−1)`
/// (window expansions, conservatively priced as if every query were alone,
/// plus the remaining dense sweep). `hi` clamps to [`INDUCE_K_MAX_MEM`] and
/// to the parallelism bound `k_par` that keeps `≥ 4` windows per thread for
/// the densify, and never drops below the incumbent 8.
/// Ranked optima: `(20, 218, 16t) → 12`, `(18, 106, 16t) → 11`.
fn induce_window_k(log_d: usize, n_queries: usize, threads: usize, singleton: bool) -> usize {
    if log_d < 12 || n_queries == 0 {
        return 0;
    }
    if !singleton {
        return 8usize.min(log_d);
    }
    let k_par = log_d.saturating_sub(ceil_log2(threads.max(1)) + 2);
    let hi = INDUCE_K_MAX_MEM.min(k_par).min(log_d).max(8);
    let cost = |k: usize| -> u128 {
        (n_queries as u128) * ((1u128 << k) - 1) + (((log_d - k) as u128) << (log_d - 1))
    };
    (8..=hi).min_by_key(|&k| cost(k)).unwrap_or(8)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InduceSingletonStats {
    /// Windows expanded through the one-nonzero shortcut.
    singleton_hits: usize,
    /// Windows with `2 ≤ nnz ≤ k/2`, expanded per-nonzero and XOR-summed.
    multi_hits: usize,
    /// Windows past the `nnz ≤ k/2` cost boundary — incumbent dense path.
    collision_fallbacks: usize,
}

/// Sparse-prefix variant of [`transpose_forward_ntt`]: exploits that the input
/// has only `positions.len()` nonzeros and that the first `k` transpose steps
/// (forward layers `log_d-1 .. log_d-k`, pairing distances `1 .. 2^(k-1)`) mix
/// only **within** `2^k`-aligned windows. We process just the windows that
/// contain a nonzero (a dense `2^k` transpose each), densify, then run the
/// remaining steps as full dense sweeps. Output is identical to
/// `transpose_forward_ntt` applied to the scattered input.
fn transpose_forward_ntt_sparse(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
) -> Vec<F128> {
    transpose_forward_ntt_sparse_inner(ntt, positions, values, log_d, None, None).0
}

fn transpose_forward_ntt_sparse_inner(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    singleton_override: Option<bool>,
    k_override: Option<usize>,
) -> (Vec<F128>, InduceSingletonStats) {
    use rayon::prelude::*;
    use std::collections::HashMap;
    let n = 1usize << log_d;
    assert_eq!(positions.len(), values.len());
    // The singleton/multi-singleton shortcut needs the fill-era runs grouping
    // (per-window nonzero counts up front); both are hoisted above the window
    // width choice because the optimum k depends on whether the shortcut is
    // live — dense windows at k = 12 are a net LOSS, so k must fall back to 8
    // whenever the shortcut is off (`FLOCK_NO_LIG_INDUCE_SINGLETON` is the
    // whole-mechanism switch).
    let fill = open_fill_enabled();
    let singleton_on = fill && singleton_override.unwrap_or_else(lig_induce_singleton_enabled);
    // No prefix for small domains or empty query sets — scatter + full dense
    // transform. `k_override` is a test clamp.
    let k = if log_d < 12 || positions.is_empty() {
        0
    } else {
        match k_override {
            Some(v) => v.clamp(1, log_d),
            None => induce_window_k(
                log_d,
                positions.len(),
                rayon::current_num_threads(),
                singleton_on,
            ),
        }
    };

    if k == 0 {
        let mut data = vec![F128::ZERO; n];
        for (&p, &v) in positions.iter().zip(values) {
            data[p] += v;
        }
        if log_d > 0 {
            transpose_forward_ntt(ntt, &mut data, log_d);
        }
        return (data, InduceSingletonStats::default());
    }

    let wmask = (1usize << k) - 1;
    // Group nonzeros into 2^k windows.
    //
    // The incumbent built the window buffers in a SERIAL `HashMap` pass:
    // one `vec![ZERO; 2^k]` allocation + zero-fill per active window (213 ×
    // 4 KiB at the ranked L0 induce, 103 × 4 KiB at L1) plus the hashing,
    // all on one core before any worker started. With `FLOCK_NO_OPEN_FILL`
    // unset we instead compute the window key per nonzero, group by a sort of
    // the (tiny) index list, and let each window's task allocate, fill AND
    // transpose its own buffer inside the existing parallel region — the
    // buffer is written by the same thread that immediately reads it.
    //
    // Bit-identical: `sample_distinct_queries` yields DISTINCT positions, so
    // every slot of every window buffer is written by exactly one nonzero;
    // and the run grouping is a stable sort, so even coincident positions
    // would accumulate in their original order. The window ORDER in the output
    // vector is irrelevant — `window_slots` scatters by window index (the
    // incumbent's `HashMap` iteration order was already nondeterministic).
    let mut win_vec: Vec<(usize, Vec<F128>)> = Vec::new();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut order: Vec<u32> = Vec::new();
    if fill {
        order.extend(0..positions.len() as u32);
        order.sort_by_key(|&i| positions[i as usize] >> k);
        let mut s = 0usize;
        while s < order.len() {
            let w = positions[order[s] as usize] >> k;
            let mut e = s + 1;
            while e < order.len() && positions[order[e] as usize] >> k == w {
                e += 1;
            }
            runs.push((s, e));
            s = e;
        }
    } else {
        let mut windows: HashMap<usize, Vec<F128>> = HashMap::new();
        for (&p, &v) in positions.iter().zip(values) {
            let buf = windows
                .entry(p >> k)
                .or_insert_with(|| vec![F128::ZERO; 1 << k]);
            buf[p & wmask] += v;
        }
        win_vec = windows.into_iter().collect();
    }

    // The `nnz ≤ k/2` threshold is the exact per-window cost boundary:
    // `nnz` expansions cost `nnz·(2^k − 1)` products vs the dense window's
    // `k·2^(k-1)`, and `nnz ≤ k·2^(k-1)/(2^k − 1)` reduces to the integer
    // test `nnz ≤ k/2` for every k. The shortcut is therefore never costlier
    // than the dense window it replaces.
    let multi_max = k / 2;
    let singleton_stats = if singleton_on && fill {
        InduceSingletonStats {
            singleton_hits: runs.iter().filter(|(s, e)| e - s == 1).count(),
            multi_hits: runs
                .iter()
                .filter(|(s, e)| e - s > 1 && e - s <= multi_max)
                .count(),
            collision_fallbacks: runs.iter().filter(|(s, e)| e - s > multi_max).count(),
        }
    } else {
        InduceSingletonStats::default()
    };

    // Steps s = 0..k-1 within each active window, in parallel (windows disjoint).
    let nwins = if fill { runs.len() } else { win_vec.len() };
    let _tw = std::time::Instant::now();
    let processed: Vec<(usize, Vec<F128>)> = if fill {
        runs.par_iter()
            .map_init(Vec::<F128>::new, |scratch, &(rs, re)| {
                let first = order[rs] as usize;
                let w = positions[first] >> k;
                let nnz = re - rs;
                if singleton_on && nnz <= multi_max {
                    // Every nonzero expands separately; the sums XOR
                    // together. F2-linearity: the transpose layers are
                    // linear over F2, so T(Σ vᵢ·δ_pᵢ) = Σ T(vᵢ·δ_pᵢ),
                    // and F128 addition is XOR (order irrelevant).
                    // Separate buffers are required — step s of an
                    // expansion reads its own prefix.
                    let mut buf = take_singleton_buf(k);
                    expand_singleton_into(
                        ntt,
                        log_d,
                        k,
                        w,
                        positions[first] & wmask,
                        values[first],
                        &mut buf,
                    );
                    if nnz > 1 {
                        if scratch.len() < 1usize << k {
                            *scratch = take_singleton_buf(k);
                        }
                        let scratch = &mut scratch[..1usize << k];
                        for &i in &order[rs + 1..re] {
                            let i = i as usize;
                            expand_singleton_into(
                                ntt,
                                log_d,
                                k,
                                w,
                                positions[i] & wmask,
                                values[i],
                                scratch,
                            );
                            for (d, &s) in buf.iter_mut().zip(scratch.iter()) {
                                *d += s;
                            }
                        }
                    }
                    return (w, buf);
                }
                // Past the cost boundary: incumbent dense window
                // (needs the zeroed allocation — only the nonzero slots
                // are scattered).
                let mut buf = vec![F128::ZERO; 1 << k];
                for &i in &order[rs..re] {
                    let i = i as usize;
                    buf[positions[i] & wmask] += values[i];
                }
                transpose_forward_ntt_window_dense(ntt, log_d, k, w, buf)
            })
            .collect()
    } else {
        win_vec
            .into_par_iter()
            .map(|(w, buf)| transpose_forward_ntt_window_dense(ntt, log_d, k, w, buf))
            .collect()
    };
    let win_ms = _tw.elapsed().as_secs_f64() * 1e3;

    // Densify (active windows only; the rest stay zero, which is the correct
    // post-step-(k-1) state for an all-zero window).
    let ot = open_timing();
    let _ta = std::time::Instant::now();
    let mf0 = if ot { minor_faults() } else { 0 };
    let (mut data, alloc_ms, dens_ms) = if induce_fused_densify_enabled() {
        // FUSED: one parallel pass writes every window exactly once, from an
        // UNINITIALIZED buffer. See `densify_windows_fused`.
        let slots = window_slots(n >> k, processed);
        let alloc_ms = _ta.elapsed().as_secs_f64() * 1e3;
        let _td = std::time::Instant::now();
        let data = densify_windows_fused(n, k, slots);
        (data, alloc_ms, _td.elapsed().as_secs_f64() * 1e3)
    } else {
        let mut data = vec![F128::ZERO; n];
        let alloc_ms = _ta.elapsed().as_secs_f64() * 1e3;
        let _td = std::time::Instant::now();
        for (w, buf) in processed {
            data[(w << k)..((w + 1) << k)].copy_from_slice(&buf);
        }
        (data, alloc_ms, _td.elapsed().as_secs_f64() * 1e3)
    };

    // Remaining steps s = k..log_d-1 = forward layers (log_d-1-k) .. 0, dense.
    let _ts = std::time::Instant::now();
    transpose_forward_ntt_dense_layers(ntt, &mut data, log_d - k);
    if ot {
        eprintln!(
            "      [sparse-ntt] log_d={log_d} k={k} wins={nwin} win-phase {win_ms:.2} ms  alloc(zeroed 2^{log_d}) {alloc_ms:.2} ms  densify {dens_ms:.2} ms  dense({dl} layers) {ds:.2} ms  minflt +{mf}",
            nwin = nwins,
            win_ms = win_ms,
            dl = log_d - k,
            ds = _ts.elapsed().as_secs_f64() * 1e3,
            mf = minor_faults() - mf0,
        );
    }
    (data, singleton_stats)
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

// Recycle the codeword matrix through the F128 scratch pool and the Merkle
// tree through TREE_POOL when a level's witness is replaced/dropped. Ranked
// L1 is 16 MiB (2^18 leaves x 32 B x ~2); without give_tree the extra-warmup
// prove munmaps it and the timed path mmap/faults a fresh one.
impl Drop for LigeroWitness {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.mat));
        crate::pcs::commit::give_tree(std::mem::take(&mut self.tree));
    }
}

// SumcheckProver owns the two witness-sized polynomials of the open (the
// packed witness `f` and the γ-combined basis) — recycle owned heap buffers
// on drop. Arena-carved buffers are views into `fold_arena`, which drops
// (joins its prefault thread + frees the one allocation) right after.
impl Drop for SumcheckProver {
    fn drop(&mut self) {
        if let FoldBuf::Owned(v) = std::mem::take(&mut self.f) {
            crate::scratch::give_f128(v);
        }
        if let FoldBuf::Owned(v) = std::mem::take(&mut self.combined_basis) {
            crate::scratch::give_f128(v);
        }
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
/// Diagnostics: this process's minor page-fault count (`ru_minflt`), for the
/// `FLOCK_OPEN_TIMING` commit lines. 0 on non-unix.
#[cfg(unix)]
fn minor_faults() -> u64 {
    #[repr(C)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i64,
    }
    #[repr(C)]
    struct Rusage {
        ru_utime: Timeval,
        ru_stime: Timeval,
        ru_maxrss: i64,
        ru_ixrss: i64,
        ru_idrss: i64,
        ru_isrss: i64,
        ru_minflt: i64,
        ru_majflt: i64,
        ru_nswap: i64,
        ru_inblock: i64,
        ru_oublock: i64,
        ru_msgsnd: i64,
        ru_msgrcv: i64,
        ru_nsignals: i64,
        ru_nvcsw: i64,
        ru_nivcsw: i64,
    }
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut ru = std::mem::MaybeUninit::<Rusage>::uninit();
    // SAFETY: RUSAGE_SELF = 0; the struct matches glibc's layout on x86_64/aarch64 Linux.
    let rc = unsafe { getrusage(0, ru.as_mut_ptr()) };
    if rc != 0 {
        return 0;
    }
    // SAFETY: getrusage succeeded and initialized the struct.
    unsafe { ru.assume_init() }.ru_minflt as u64
}
#[cfg(not(unix))]
fn minor_faults() -> u64 {
    0
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

    // LSB-lane input already matches the position-major SoA codeword layout.
    // The semantic encoder owns zero-padding shortcuts and target-specific
    // fusion while overwriting every slot of the recycled matrix.
    let codeword_len = block_len * num_interleaved;
    let ot = open_timing();
    let t_alloc = std::time::Instant::now();
    let mut mat = crate::scratch::take_f128(codeword_len);
    let mat_alloc_ms = t_alloc.elapsed().as_secs_f64() * 1e3;
    // Merkle over rows. One leaf = `num_interleaved` consecutive F128 = 16·num_interleaved bytes.
    let leaf_size_bytes = num_interleaved * core::mem::size_of::<F128>();

    // Fused recursive commit (x86 production route): the NTT deep pass hands
    // each finalized sub-group to the worker that produced it, which hashes
    // those leaves serially while they are cache-resident and folds the
    // sub-group's own Merkle subtree — no separate cold leaf pass over the
    // codeword and no per-level rayon barrier for the wide levels. Same shape
    // the L0 commit uses; bit-identical to encode-then-fill_merkle_tree.
    // Skipped when a GPU Merkle session could take the leaves instead.
    if crate::pcs::commit::lig_fused_commit_enabled() && !crate::gpu::merkle::available() {
        let faults_before = if ot { minor_faults() } else { 0 };
        let mat_cap = mat.capacity();
        let t_encode = std::time::Instant::now();
        let mut tree = crate::pcs::commit::take_tree(2 * block_len - 1);
        let folded = crate::pcs::commit::fused_encode_leaves_subtree(
            ntt,
            poly,
            &mut mat,
            num_interleaved,
            &mut tree,
            block_len,
            leaf_size_bytes,
            kind,
        );
        let fused_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_upper = std::time::Instant::now();
        crate::pcs::commit::build_upper_levels(&mut tree, block_len, block_len >> folded, kind);
        if ot {
            eprintln!(
                "[open-timing] ligero_commit: leaves=2^{log_block_len} leaf={leaf_size_bytes}B \
                 ({:.1} MiB) mat-alloc {mat_alloc_ms:.2} ms (cap {} F128, tree cap {}) FUSED \
                 encode+leaves+subtree({folded}) {fused_ms:.2} ms upper {:.2} ms minflt +{}",
                (codeword_len * 16) as f64 / (1024.0 * 1024.0),
                mat_cap,
                tree.capacity(),
                t_upper.elapsed().as_secs_f64() * 1e3,
                minor_faults().saturating_sub(faults_before),
            );
        }
        return LigeroWitness {
            mat,
            tree,
            block_len,
            num_interleaved,
        };
    }

    let t_encode = std::time::Instant::now();
    ntt.rs_encode_interleaved(poly, &mut mat, num_interleaved);
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
    let data_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            mat.as_ptr() as *const u8,
            mat.len() * core::mem::size_of::<F128>(),
        )
    };
    debug_assert_eq!(data_bytes.len(), block_len * leaf_size_bytes);
    let t_merkle = std::time::Instant::now();
    let mut gpu_busy_ms = 0.0f64;
    let gpu_tree = if kind == HashKind::Blake3
        && block_len >= gpu_open_merkle_min_leaves()
        && merkle::blake3_leaf_size_is_batchable(leaf_size_bytes)
        && crate::gpu::merkle::available()
    {
        gpu_merkle_tree_for_open(data_bytes, block_len, leaf_size_bytes, &mut gpu_busy_ms)
    } else {
        None
    };
    let on_gpu = gpu_tree.is_some();
    let mut tree_alloc_ms = 0.0f64;
    let mut leaves_ms = 0.0f64;
    let tree = gpu_tree.unwrap_or_else(|| {
        // Same write-before-read contract as merkle_tree(); take from TREE_POOL
        // so the ranked L1 16 MiB tree is already resident after untimed warmup
        // (LigeroWitness::drop parks it). Public merkle_tree() stays unpooled
        // so tests/oracles cannot steal the L0 64 MiB slot.
        let t_ta = std::time::Instant::now();
        let mut tree = crate::pcs::commit::take_tree(2 * block_len - 1);
        tree_alloc_ms = t_ta.elapsed().as_secs_f64() * 1e3;
        if ot {
            let t_l = std::time::Instant::now();
            merkle::hash_leaves(data_bytes, leaf_size_bytes, &mut tree[..block_len], kind);
            leaves_ms = t_l.elapsed().as_secs_f64() * 1e3;
            crate::pcs::commit::build_upper_levels(&mut tree, block_len, block_len, kind);
        } else {
            merkle::fill_merkle_tree(&mut tree, data_bytes, block_len, kind);
        }
        tree
    });
    if ot {
        eprintln!(
            "[open-timing] ligero_commit: leaves=2^{log_block_len} leaf={leaf_size_bytes}B \
             ({:.1} MiB) mat-alloc {mat_alloc_ms:.2} ms encode {encode_ms:.2} ms merkle({}) {:.2} ms \
             [tree-alloc {tree_alloc_ms:.2} leaves {leaves_ms:.2}] (gpu busy {gpu_busy_ms:.2} ms)",
            (codeword_len * 16) as f64 / (1024.0 * 1024.0),
            if on_gpu { "gpu" } else { "cpu" },
            t_merkle.elapsed().as_secs_f64() * 1e3,
        );
    }

    LigeroWitness {
        mat,
        tree,
        block_len,
        num_interleaved,
    }
}

/// Leaf-count floor for routing a recursive-commit Merkle tree through the
/// GPU session. Only the L1 (2^18 leaves at the ranked m=32 shape) and L2
/// (2^16) trees are big enough to possibly beat the wide-pool CPU hash.
/// Default OFF: in-process paired A/B measured the route ~-3.7 ms on the
/// open phase, but the trusted fresh-worker harness measured it -4.5% on
/// the MEDIAN (869,807 vs 912,904 c/s locally) — per-worker session fixed
/// costs and contention with the URM/Merkle streams dominate outside a
/// long-lived process. `FLOCK_GPU_OPEN_MERKLE=1` opts in;
/// `FLOCK_GPU_OPEN_MERKLE_MIN_LOG2` overrides the floor (diagnostics).
fn gpu_open_merkle_min_leaves() -> usize {
    static MIN: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if std::env::var_os("FLOCK_NO_GPU_OPEN_MERKLE").is_some() {
            return usize::MAX;
        }
        match std::env::var("FLOCK_GPU_OPEN_MERKLE_MIN_LOG2") {
            Ok(s) => s.parse::<u32>().map(|l| 1usize << l).unwrap_or(usize::MAX),
            Err(_) if std::env::var_os("FLOCK_GPU_OPEN_MERKLE").is_some() => 1usize << 18,
            Err(_) => usize::MAX,
        }
    });
    *MIN
}

/// The GPU builds parent levels while their node count is ≥ this; the CPU
/// finishes the top (≤ 1023 pair hashes — microseconds). Small enough that
/// every eligible tree (≥ 2^16 leaves) gets its parent levels on the GPU.
const GPU_OPEN_STOP_NODES: usize = 1 << 10;

/// Tree over-allocation, in nodes: one 16 KiB page (512 × 32 B) so
/// `gpu::merkle::begin`'s floor-page coverage check always passes for the
/// real node range (same rule as `pcs::commit::TREE_PAD_NODES`).
const GPU_OPEN_TREE_PAD_NODES: usize = 512;

/// Build one recursive-commit Merkle tree (BLAKE3, flat `merkle_tree`
/// layout) on the GPU: one leaf command buffer over the fully-encoded
/// codeword, one parent-levels command buffer, CPU top from
/// [`GPU_OPEN_STOP_NODES`]. Returns `None` on any refusal or failure —
/// nothing of the returned-tree contract is left half-done, so the caller
/// falls back to the byte-identical CPU `merkle_tree` (GPU API failures
/// latch the process-wide disable, exactly like the streamed-commit path).
///
/// These trees (2-16 MiB) sit below the 64 MiB wrap-cache floor, so the
/// buffers are wrapped fresh per prove and freed with the tree — no pool
/// retention requirement, and no prewire (prewire no-ops below the floor).
fn gpu_merkle_tree_for_open(
    data_bytes: &[u8],
    block_len: usize,
    leaf_size: usize,
    busy_ms: &mut f64,
) -> Option<Vec<Hash>> {
    let total_nodes = 2 * block_len - 1;
    let mut tree: Vec<Hash> = crate::alloc_uninit_vec(total_nodes + GPU_OPEN_TREE_PAD_NODES);
    // SAFETY (begin): `data_bytes` (the encoded codeword) and `tree` both
    // outlive the session — finish() is called below before either can drop
    // — and the CPU neither reads nor writes the GPU-owned node range
    // `[0, 2n − s_last)` until finish() returns.
    let mut session = unsafe {
        crate::gpu::merkle::begin(
            data_bytes,
            leaf_size,
            tree.as_mut_ptr(),
            tree.len(),
            GPU_OPEN_STOP_NODES,
        )
    }?;
    if !session.commit_leaves(0, block_len) || !session.commit_parent_levels() {
        // Latched inside the session; drain what was committed and rebuild
        // everything on the CPU (the tree buffer is discarded untouched).
        session.finish();
        return None;
    }
    *busy_ms = session.finish()? * 1e3;
    let from_nodes = if GPU_OPEN_STOP_NODES <= block_len / 2 {
        GPU_OPEN_STOP_NODES
    } else {
        block_len
    };
    crate::pcs::commit::build_upper_levels(&mut tree, block_len, from_nodes, HashKind::Blake3);
    tree.truncate(total_nodes);
    Some(tree)
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

    // Layout matches msg_reduce: pairs are consecutive (f[2j], f[2j+1]).
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    {
        const PAR_THRESHOLD: usize = 4096;
        if n < PAR_THRESHOLD {
            // SAFETY: features cfg-guaranteed; n even and >= 2.
            let (u_0, u_2) = unsafe { msg_reduce_avx512(f, b) };
            return SumcheckMessage { u_0, u_2 };
        }
        // Chunked parallel reduce; each chunk length is a multiple of 8 when
        // possible so the AVX-512 body stays saturated.
        const CHUNK: usize = 1024;
        let (u_0, u_2) = f
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(fc, bc)| {
                // SAFETY: equal chunk lengths; features cfg-guaranteed.
                unsafe { msg_reduce_avx512(fc, bc) }
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );
        return SumcheckMessage { u_0, u_2 };
    }

    #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
    {
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

    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    if open_ood_x4_enabled() {
        const PAR_THRESHOLD: usize = 4096;
        if n < PAR_THRESHOLD {
            // SAFETY: features cfg-guaranteed; equal lengths.
            let (u_0, u_2, y) = unsafe { msg_reduce_eval_avx512(f, b) };
            return (SumcheckMessage { u_0, u_2 }, y);
        }
        const CHUNK: usize = 1024;
        let (u_0, u_2, y) = f
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(fc, bc)| {
                // SAFETY: equal chunk lengths; features cfg-guaranteed.
                unsafe { msg_reduce_eval_avx512(fc, bc) }
            })
            .reduce(
                || (F128::ZERO, F128::ZERO, F128::ZERO),
                |(a0, a2, ay), (c0, c2, cy)| (a0 + c0, a2 + c2, ay + cy),
            );
        return (SumcheckMessage { u_0, u_2 }, y);
    }

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

/// Width of the cache-resident low equality factor retained by the ranked L1
/// OOD path.  Its 18-variable tail is represented as 2^11 low weights and
/// 2^7 high weights instead of one dense 2^18 table.
const LAZY_OOD_EQ_SPLIT_LOW_LOG: usize = 11;

/// Deferred-reduction AVX-512 sufficient statistics for a factorized LSB
/// equality. Returns
/// `a = sum f[2j] * w[j]` and
/// `s = sum (f[2j] + f[2j+1]) * w[j]`.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn factorized_eq_round0_avx512(f: &[F128], weights: &[F128]) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;

    // SAFETY: the caller guarantees the target features and two readable
    // witness elements for every weight; loop bounds preserve both contracts.
    unsafe {
        debug_assert_eq!(f.len(), 2 * weights.len());
        let mut a_acc = WideGhashX4::zero();
        let mut s_acc = WideGhashX4::zero();
        let lanes = weights.len() & !3;
        let mut j = 0usize;
        while j < lanes {
            let f0 = _mm512_loadu_si512(f.as_ptr().add(2 * j) as *const __m512i);
            let f1 = _mm512_loadu_si512(f.as_ptr().add(2 * j + 4) as *const __m512i);
            let w = _mm512_loadu_si512(weights.as_ptr().add(j) as *const __m512i);
            let even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
            let f0_sum = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
            let f1_sum = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
            let sum = _mm512_shuffle_i32x4::<0x88>(f0_sum, f1_sum);
            a_acc.mul_acc(even, w);
            s_acc.mul_acc(sum, w);
            j += 4;
        }
        let mut a = a_acc.fold().reduce();
        let mut s = s_acc.fold().reduce();
        while j < weights.len() {
            a += f[2 * j] * weights[j];
            s += (f[2 * j] + f[2 * j + 1]) * weights[j];
            j += 1;
        }
        (a, s)
    }
}

#[inline]
fn factorized_eq_round0(f: &[F128], weights: &[F128]) -> (F128, F128) {
    assert_eq!(f.len(), 2 * weights.len());
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    // SAFETY: the cfg gate supplies AVX-512 + VPCLMUL and the length contract
    // above supplies two witness values per equality weight.
    unsafe {
        factorized_eq_round0_avx512(f, weights)
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
    {
        let mut a = F128::ZERO;
        let mut s = F128::ZERO;
        for (pair, &weight) in f.chunks_exact(2).zip(weights) {
            a += pair[0] * weight;
            s += (pair[0] + pair[1]) * weight;
        }
        (a, s)
    }
}

/// Factorized equivalent of [`round_msg_and_eval_lsb`] for
/// `b = eq([z_0, z_tail...], ·)`. The dense equality tail is the tensor
/// product `eq_lo[i] * eq_hi[h]`; each low factor remains hot across the
/// witness chunk it weights.
fn round_msg_and_eval_lsb_factorized_eq(
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
        .expect("factorized OOD tail overflow");
    assert_eq!(f.len(), 2 * tail_len);

    let (a, s) = f
        .par_chunks(2 * eq_lo.len())
        .zip(eq_hi.par_iter())
        .map(|(chunk, &hi)| {
            let (a, s) = factorized_eq_round0(chunk, eq_lo);
            (a * hi, s * hi)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, s0), (a1, s1)| (a0 + a1, s0 + s1),
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
/// production path uses `fold_and_msg_lsb` instead.
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
/// Returns `(folded_f, folded_b, next_msg)` where `next_msg = round_msg_lsb
/// (folded_f, folded_b)`. Bit-identical to the unfused sequence.
///
/// `arena`: optional per-open [`FoldArena`] the parallel path carves its two
/// output buffers from (prefaulted pages, no per-round zero-fill faults).
/// `None`, an exhausted arena, or the serial path fall back to the previous
/// per-arch allocation behavior.
/// AVX-512 + VPCLMULQDQ vectorized message-term reduction for
/// [`fold_and_msg_lsb`]. Replaces the scalar `u0 += f0*b0; u2 += (f0+f1)*(b0+b1)`
/// loop with a 4-lane unreduced multiply-accumulate, folding the 4 lanes
/// and reducing once at the end.  Processes 8 F128 (4 pairs) per iteration.
///
/// `fc` and `bc` are the folded slices (length `half = n/2`, a power of two
/// ≥ `PAR_THRESHOLD/2`). The message pairs are (k, k+1) for k = 0, 2, 4, …:
///   u0 = Σ fc[k]·bc[k]           (products at even pair-positions)
///   u2 = Σ (fc[k]+fc[k+1])·(bc[k]+bc[k+1])  (products of pair sums)
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq` (cfg-gated at call site).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn msg_reduce_avx512(fc: &[F128], bc: &[F128]) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    // Process 8 F128 (4 message pairs) per iteration.
    let lanes = len & !7;
    let mut u0_acc = WideGhashX4::zero();
    let mut u2_acc = WideGhashX4::zero();

    let mut k = 0;
    while k < lanes {
        // Load 4 F128 from fc and 4 from bc (positions k..k+4 and k+4..k+8).
        let f0 = _mm512_loadu_si512(fc.as_ptr().add(k) as *const __m512i);
        let f1 = _mm512_loadu_si512(fc.as_ptr().add(k + 4) as *const __m512i);
        let b0 = _mm512_loadu_si512(bc.as_ptr().add(k) as *const __m512i);
        let b1 = _mm512_loadu_si512(bc.as_ptr().add(k + 4) as *const __m512i);

        // u0: products at even pair-positions k, k+2, k+4, k+6.
        let f_even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
        let b_even = _mm512_shuffle_i32x4::<0x88>(b0, b1);
        u0_acc.mul_acc(f_even, b_even);

        // u2: pair sums (fc[k]+fc[k+1]), (fc[k+2]+fc[k+3]),
        //               (fc[k+4]+fc[k+5]), (fc[k+6]+fc[k+7]).
        let f0s = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
        let f1s = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
        let f_sum = _mm512_shuffle_i32x4::<0x88>(f0s, f1s);
        let b0s = _mm512_xor_si512(b0, _mm512_shuffle_i32x4::<0xB1>(b0, b0));
        let b1s = _mm512_xor_si512(b1, _mm512_shuffle_i32x4::<0xB1>(b1, b1));
        let b_sum = _mm512_shuffle_i32x4::<0x88>(b0s, b1s);
        u2_acc.mul_acc(f_sum, b_sum);

        k += 8;
    }

    // Fold the 4-lane unreduced accumulators to scalar F128.
    let mut u0 = u0_acc.fold().reduce();
    let mut u2 = u2_acc.fold().reduce();

    // Scalar tail for remaining pairs.
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

/// [`msg_reduce_avx512`] with the full inner product `Σ_i fc[i]·bc[i]`
/// accumulated in the same sweep — the leaf of [`round_msg_and_eval_lsb`].
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq` (cfg-gated at the call site). `fc` and
/// `bc` must have equal length.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn msg_reduce_eval_avx512(fc: &[F128], bc: &[F128]) -> (F128, F128, F128) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    let lanes = len & !7;
    // SAFETY: caller carries the target features; every load below is inside
    // the equal-length slices.
    unsafe {
        let mut u0_acc = WideGhashX4::zero();
        let mut u2_acc = WideGhashX4::zero();
        let mut y_acc = WideGhashX4::zero();

        let mut k = 0;
        while k < lanes {
            let f0 = _mm512_loadu_si512(fc.as_ptr().add(k) as *const __m512i);
            let f1 = _mm512_loadu_si512(fc.as_ptr().add(k + 4) as *const __m512i);
            let b0 = _mm512_loadu_si512(bc.as_ptr().add(k) as *const __m512i);
            let b1 = _mm512_loadu_si512(bc.as_ptr().add(k + 4) as *const __m512i);

            let f_even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
            let b_even = _mm512_shuffle_i32x4::<0x88>(b0, b1);
            u0_acc.mul_acc(f_even, b_even);

            let f0s = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
            let f1s = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
            let f_sum = _mm512_shuffle_i32x4::<0x88>(f0s, f1s);
            let b0s = _mm512_xor_si512(b0, _mm512_shuffle_i32x4::<0xB1>(b0, b0));
            let b1s = _mm512_xor_si512(b1, _mm512_shuffle_i32x4::<0xB1>(b1, b1));
            let b_sum = _mm512_shuffle_i32x4::<0x88>(b0s, b1s);
            u2_acc.mul_acc(f_sum, b_sum);

            // y is the inner product over EVERY slot, so both registers feed it.
            y_acc.mul_acc(f0, b0);
            y_acc.mul_acc(f1, b1);

            k += 8;
        }

        let mut u0 = u0_acc.fold().reduce();
        let mut u2 = u2_acc.fold().reduce();
        let mut y = y_acc.fold().reduce();

        while k + 1 < len {
            let f0 = fc[k];
            let f1 = fc[k + 1];
            let b0 = bc[k];
            let b1 = bc[k + 1];
            let e0 = f0 * b0;
            u0 += e0;
            u2 += (f0 + f1) * (b0 + b1);
            y += e0 + f1 * b1;
            k += 2;
        }

        (u0, u2, y)
    }
}

/// Four-lane `acc[i] += alpha·src[i]` — the leaf of [`SumcheckProver::glue`].
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq` (cfg-gated at the call site). `acc` and
/// `src` must have equal length.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn glue_block_x4(acc: &mut [F128], src: &[F128], alpha: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;
    debug_assert_eq!(acc.len(), src.len());
    // SAFETY: caller carries the target features; the slices are equal-length
    // and every offset below stays inside both.
    unsafe {
        let ab = _mm512_broadcast_i32x4(_mm_set_epi64x(alpha.hi as i64, alpha.lo as i64));
        let ab_x64 = ghash_shift64_x4(ab);
        let lanes = acc.len() & !3;
        let mut i = 0usize;
        while i < lanes {
            let a = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
            let v = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
            _mm512_storeu_si512(
                acc.as_mut_ptr().add(i) as *mut __m512i,
                _mm512_xor_si512(a, ghash_mul_x4_split(v, ab, ab_x64)),
            );
            i += 4;
        }
        while i < acc.len() {
            acc[i] += alpha * src[i];
            i += 1;
        }
    }
}

/// `FLOCK_NO_OPEN_OOD_X4=1` restores the scalar per-pair loops of
/// [`round_msg_and_eval_lsb`] and [`SumcheckProver::glue`] (exact same-binary
/// A/B; the wide leaves accumulate the same canonical products and F128
/// addition is XOR, so both results are bit-identical).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
fn open_ood_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_OOD_X4").is_none());
    *ON
}

/// Publish `n` F128s with XMM non-temporal stores (`_mm_stream_si128` /
/// `MOVNTDQ`). Same helper shape as the promoted seed-fused publish
/// (`AdditiveNttF128::publish_row_nt`): XMM, not ZMM, because large pool /
/// arena slices on this lineage land 16 mod 64 — a 64-byte gate in front of
/// `_mm512_stream_si512` would silently never fire.
///
/// # Safety
/// `src`/`dst` cover `n` F128s; `dst` is 16-byte aligned. SSE2 is x86_64
/// baseline.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(dead_code)] // Retained same-binary non-temporal publish rollback.
unsafe fn publish_f128_row_nt(src: *const F128, dst: *mut F128, n: usize) {
    use core::arch::x86_64::*;
    // SAFETY: bounds and 16-byte dest alignment are the caller's contract;
    // `_mm_loadu_si128` accepts any src alignment.
    unsafe {
        let s = src as *const __m128i;
        let d = dst as *mut __m128i;
        for i in 0..n {
            _mm_stream_si128(d.add(i), _mm_loadu_si128(s.add(i)));
        }
    }
}

/// x86 NT leaf for one [`fold_and_msg_lsb`] chunk.
///
/// Value-identical to the generic chunk body (`fold_pairs` on `f`/`b` then
/// [`msg_reduce_avx512`] on the folded slices). The existing AVX-512 kernels
/// write a reused L1-resident stage; the message reduce reads that stage
/// (no destination reload); the destination is published with XMM streaming
/// stores so each output line skips write-allocate RFO. Next reader is the
/// following sumcheck round, after a Fiat–Shamir grind — DRAM-cold when
/// `half >= 2^21` (32 MiB per buffer, 64 MiB for the pair).
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq`. `fc`/`bc` have equal even length
/// `<= stage_f.len()`. `f`/`b` contain `2 * (base + fc.len())` elements.
/// `stage_*` are write-before-read scratch owned by this worker.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn fold_and_msg_chunk_nt_x86(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
    stage_f: &mut [F128],
    stage_b: &mut [F128],
) -> (F128, F128) {
    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    debug_assert!(len <= stage_f.len() && len <= stage_b.len());
    debug_assert!(len.is_multiple_of(2));

    crate::field::f128_slice::fold_pairs(f, base, &mut stage_f[..len], r);
    crate::field::f128_slice::fold_pairs(b, base, &mut stage_b[..len], r);
    // SAFETY: target features cfg-guaranteed; stage slices have equal even
    // length (caller / debug_assert).
    let (u0, u2) = unsafe { msg_reduce_avx512(&stage_f[..len], &stage_b[..len]) };

    let dst_aligned = (fc.as_mut_ptr() as usize).is_multiple_of(16)
        && (bc.as_mut_ptr() as usize).is_multiple_of(16);
    if dst_aligned {
        // SAFETY: dest slices are 16-aligned F128 buffers of length `len`;
        // stage is the just-written source of the same length.
        unsafe {
            publish_f128_row_nt(stage_f.as_ptr(), fc.as_mut_ptr(), len);
            publish_f128_row_nt(stage_b.as_ptr(), bc.as_mut_ptr(), len);
            core::arch::x86_64::_mm_sfence();
        }
    } else {
        fc.copy_from_slice(&stage_f[..len]);
        bc.copy_from_slice(&stage_b[..len]);
    }
    (u0, u2)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
thread_local! {
    static OPEN_NT_STAGE: std::cell::RefCell<(Vec<F128>, Vec<F128>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

fn fold_and_msg_lsb(
    f: &[F128],
    b: &[F128],
    r: F128,
    arena: Option<&mut FoldArena>,
) -> (FoldBuf, FoldBuf, SumcheckMessage) {
    fold_and_msg_lsb_inner(f, b, r, arena, None, None)
}

/// Core fold with optional deferred ordinary-basis and retained-equality
/// corrections. The equality correction is already in the folded domain:
/// `nb[j] += gamma * eq_hi[j / eq_lo.len()] * eq_lo[j % eq_lo.len()]`.
/// The ordinary correction stays in the input domain and is folded here:
/// `nb[j] += alpha * fold_r(deferred_basis)[j]`.
/// The ranked path fixes `eq_lo.len() == CHUNK`, so one high-factor scale is
/// hoisted per Rayon task and the low table stays in L1.
fn fold_and_msg_lsb_inner(
    f: &[F128],
    b: &[F128],
    r: F128,
    arena: Option<&mut FoldArena>,
    lazy_ood: Option<(&[F128], &[F128], F128)>,
    deferred_basis: Option<(&[F128], F128)>,
) -> (FoldBuf, FoldBuf, SumcheckMessage) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    if let Some((eq_lo, eq_hi, _)) = lazy_ood {
        assert!(eq_lo.len().is_power_of_two() && eq_lo.len() >= 2);
        assert!(eq_hi.len().is_power_of_two());
        assert_eq!(half, eq_lo.len() * eq_hi.len());
    }
    if let Some((basis, _)) = deferred_basis {
        assert_eq!(basis.len(), n, "deferred basis fold shape changed");
    }
    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        let mut nf = Vec::with_capacity(half);
        let mut nb = Vec::with_capacity(half);
        // Char-2: even*(1+r)+odd*r = even + r*(even+odd). One mul per pair.
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let b0 = b[2 * j];
            let b1 = b[2 * j + 1];
            nf.push(f0 + r * (f0 + f1));
            let mut folded_b = b0 + r * (b0 + b1);
            if let Some((basis, alpha)) = deferred_basis {
                let d0 = basis[2 * j];
                let d1 = basis[2 * j + 1];
                folded_b += alpha * (d0 + r * (d0 + d1));
            }
            nb.push(folded_b);
        }
        if let Some((eq_lo, eq_hi, gamma)) = lazy_ood {
            for (j, value) in nb.iter_mut().enumerate() {
                *value += gamma * eq_hi[j / eq_lo.len()] * eq_lo[j % eq_lo.len()];
            }
        }
        // Same AVX-512 message reduce used on the parallel path.
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        let (u_0, u_2) = unsafe { msg_reduce_avx512(&nf, &nb) };
        #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
        let (u_0, u_2) = {
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
            (u_0, u_2)
        };
        return (
            FoldBuf::Owned(nf),
            FoldBuf::Owned(nb),
            SumcheckMessage { u_0, u_2 },
        );
    }

    // Parallel path: `half` is a power of two ≥ PAR_THRESHOLD and CHUNK is a
    // power of two, so every chunk has even length and starts at an even
    // global index — message pairs (2k, 2k+1) never straddle a chunk boundary.
    const CHUNK: usize = 2048;
    // Non-temporal fold path gate: the folded `nf`/`nb` are next read only
    // after a Fiat–Shamir round trip; when each output is ≥ 32 MB (64 MB for
    // the pair) they are DRAM-cold by then on the ranked M4 Pro's SLC and
    // regular stores' write-allocate is one pure hidden DRAM read per output
    // line. The NT leaf computes the message terms from registers instead of
    // reloading the just-written pairs. `FLOCK_NO_OPEN_NT` is a
    // local-diagnostics kill switch; the ranked worker's cleared environment
    // never sets it.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_nt = lazy_ood.is_none()
        && deferred_basis.is_none()
        && half >= (1usize << 21)
        && std::env::var_os("FLOCK_NO_OPEN_NT").is_none();
    // Same DRAM-cold next-reader gate as the aarch64 NT leaf, now on the
    // ranked x86 SPR path. `#158` landed the seed-fused *publish* NT port;
    // this is the leftover L0-fold site (~960 MiB RFO across the four
    // rounds with half ≥ 2^21). `FLOCK_NO_OPEN_NT` is the same-binary A/B.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let use_nt = lazy_ood.is_none()
        && deferred_basis.is_none()
        && half >= (1usize << 21)
        && std::env::var_os("FLOCK_NO_OPEN_NT").is_none();
    // All-NEON SoA leaf (see `fold_and_msg_chunk_nt_neon_soa`) unless the
    // `FLOCK_NO_OPEN_SUMCHECK_OPT` kill switch asks for the previous GPR-mixed
    // leaf (local diagnostics / A-B; the ranked worker's cleared environment
    // never sets it). Read once per process. The SoA leaf's EOR3 needs sha3
    // (statically true under `-C target-cpu=native` on every Apple Silicon
    // target this ships to; other builds keep the previous leaf).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_soa =
        lazy_ood.is_none() && deferred_basis.is_none() && cfg!(target_feature = "sha3") && {
            static SOA: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                std::env::var_os("FLOCK_NO_OPEN_SUMCHECK_OPT").is_none()
            });
            *SOA
        };
    // Fold-output storage, in preference order:
    //   1. Per-open `FoldArena` slices: one exact-size prefaulted allocation
    //      carved round by round — removes the ~1 GiB of kernel zero-fill +
    //      page faults the fresh-per-round allocations paid inside the
    //      serial Fiat–Shamir chain.
    //   2. x86_64: prewarmed scratch pool (the prover gives the previous
    //      round's buffers back in `SumcheckProver::fold`), so the initial
    //      sumcheck reuses resident pages.
    //   3. aarch64 without an arena: fresh uninit allocation each round
    //      (cross-prove pooling measured slower here).
    let (mut nf, mut nb) = match arena.and_then(|a| a.carve_pair(half)) {
        Some(pair) => pair,
        None => {
            #[cfg(target_arch = "x86_64")]
            {
                (
                    FoldBuf::Owned(crate::scratch::take_f128(half)),
                    FoldBuf::Owned(crate::scratch::take_f128(half)),
                )
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                (
                    FoldBuf::Owned(crate::alloc_uninit_f128_vec(half)),
                    FoldBuf::Owned(crate::alloc_uninit_f128_vec(half)),
                )
            }
        }
    };
    let (nf_s, nb_s): (&mut [F128], &mut [F128]) = (&mut nf, &mut nb);
    if let Some((eq_lo, _, _)) = lazy_ood {
        assert_eq!(eq_lo.len(), CHUNK, "ranked lazy OOD chunk width changed");
    }
    let (u_0, u_2) = nf_s
        .par_chunks_mut(CHUNK)
        .zip(nb_s.par_chunks_mut(CHUNK))
        .enumerate()
        .map(|(ci, (fc, bc))| {
            let base = ci * CHUNK;
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            if use_nt {
                // Per-worker L1 stage: fold + msg_reduce stay on ~64 KiB
                // reused scratch; dest sees only the NT publish.
                return OPEN_NT_STAGE.with(|cell| {
                    let mut st = cell.borrow_mut();
                    if st.0.len() < CHUNK {
                        st.0 = crate::alloc_uninit_vec(CHUNK);
                        st.1 = crate::alloc_uninit_vec(CHUNK);
                    }
                    // SAFETY: avx512f+vpclmulqdq cfg-guaranteed; chunk
                    // geometry matches fold_and_msg_lsb's even-length
                    // power-of-two split; stage capacity is CHUNK.
                    unsafe {
                        let (sf, sb) = &mut *st;
                        fold_and_msg_chunk_nt_x86(f, b, base, fc, bc, r, sf, sb)
                    }
                });
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
            {
                // SAFETY: aes is cfg-guaranteed (sha3 checked by `use_soa`);
                // chunk geometry supplies two source elements per output
                // (bounds asserted by the caller's chunking) and every chunk
                // has even length.
                if use_soa {
                    return unsafe {
                        if use_nt {
                            fold_and_msg_chunk_nt_neon_soa::<true>(f, b, base, fc, bc, r)
                        } else {
                            // Small rounds: same fused SoA kernel, plain
                            // `stp` publish (output is re-read next round
                            // while cache-resident).
                            fold_and_msg_chunk_nt_neon_soa::<false>(f, b, base, fc, bc, r)
                        }
                    };
                }
                if use_nt {
                    return unsafe { fold_and_msg_chunk_nt_neon(f, b, base, fc, bc, r) };
                }
            }
            let len = fc.len();
            // Fold this slice, then pair up the just-folded values for the msg.
            crate::field::f128_slice::fold_pairs(f, base, fc, r);
            if let Some((basis, alpha)) = deferred_basis {
                crate::field::f128_slice::fold_pairs_with_scaled_addend(
                    b, basis, base, bc, r, alpha,
                );
            } else {
                crate::field::f128_slice::fold_pairs(b, base, bc, r);
            }
            if let Some((eq_lo, eq_hi, gamma)) = lazy_ood {
                let scale = gamma * eq_hi[ci];
                crate::field::f128_slice::add_scaled(bc, &eq_lo[..len], scale);
            }
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            {
                // SAFETY: target features cfg-guaranteed; fc/bc have equal
                // even length (caller asserts n ≥ 2, power of two).
                let (u0, u2) = unsafe { msg_reduce_avx512(fc, bc) };
                (u0, u2)
            }
            #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
            {
                let mut u0 = F128::ZERO;
                let mut u2 = F128::ZERO;
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
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
        );
    (nf, nb, SumcheckMessage { u_0, u_2 })
}

#[inline]
fn eval_lookahead(coeffs: &[F128; 6], challenge: F128) -> SumcheckMessage {
    let challenge_sq = challenge * challenge;
    SumcheckMessage {
        u_0: coeffs[0] + coeffs[1] * challenge + coeffs[2] * challenge_sq,
        u_2: coeffs[3] + coeffs[4] * challenge + coeffs[5] * challenge_sq,
    }
}

#[inline]
fn eval_quadratic_tensor(coefficients: &[F128], challenges: &[F128]) -> F128 {
    debug_assert_eq!(coefficients.len(), 3usize.pow(challenges.len() as u32));
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

#[inline]
fn eval_fold4_lookahead2(
    coefficients: &super::Fold4Lookahead2,
    r0: F128,
    r1: F128,
) -> SumcheckMessage {
    SumcheckMessage {
        u_0: eval_quadratic_tensor(&coefficients[..9], &[r0, r1]),
        u_2: eval_quadratic_tensor(&coefficients[9..], &[r0, r1]),
    }
}

#[inline]
fn eval_fold4_lookahead3(
    coefficients: &super::Fold4Lookahead3,
    r0: F128,
    r1: F128,
    r2: F128,
) -> SumcheckMessage {
    SumcheckMessage {
        u_0: eval_quadratic_tensor(&coefficients[..27], &[r0, r1, r2]),
        u_2: eval_quadratic_tensor(&coefficients[27..], &[r0, r1, r2]),
    }
}

/// `FLOCK_NO_MDF4_PF=1` restores the incumbent [`materialize_direct_fold4`],
/// which issues no software prefetch at all. Exact same-binary A/B: a
/// prefetch is a hint with no architectural effect, so both arms produce
/// byte-identical proofs.
///
/// Round 3 of the open sumcheck is the single biggest loop in the phase
/// (~10.2 ms of open's ~25.5 ms local, measured with `LIG_PROVE_TRACE`).
/// Each rayon task does exactly two things, in sequence:
///
///   1. `fold16_banked` streams this block's 2 MiB f-side slab out of the
///      512 MiB packed witness — pure DRAM, cold at the ranked shape.
///   2. the per-claim `fold_one_slot` loops fold `eq_lo` through the 64 KiB
///      composed table — 16 byte-indexed loads per slot out of L1/L2, and
///      **no DRAM traffic whatsoever**.
///
/// The ranked-shape decomposition measured that split directly (256 tasks,
/// `block_len = 8192`, `claims = 2`, 16 threads): f-side 6.47 ms wall,
/// b-side 3.39 ms wall, msg-reduce 0.32 ms wall — 10.2 ms that is the SUM of
/// a memory component and a compute component, with zero overlap. For the
/// whole b-side the memory system has no request outstanding, and for the
/// whole f-side the load ports the b-side needs are idle.
///
/// A 16-thread streaming probe on this box tops out at 67 GB/s over the same
/// 4-streams-per-KiB pattern `fold16_banked` uses, and the f-side already
/// runs at ~83 GB/s, so there is no headroom INSIDE the f-side. The win is
/// to move misses into the b-side window that currently spends 3.4 ms idle
/// on memory. This arm walks the head of the NEXT block's slab one 64-byte
/// line per b-side iteration, so those lines are already in L2 when
/// `fold16_banked` reaches them.
///
/// Distance and depth are both measured, not assumed. Depth: one line per
/// iteration covers 256 KiB of the next 2 MiB slab; at two lines and above
/// the walk evicts the b-side's own 64 KiB table and `eq_lo` stream out of
/// L2 and the b-side loses more than the f-side gains (measured: 1 line
/// f −0.62 ms / b +0.27 ms; 2 lines f −0.55 / b +0.5; 8 lines f −0.85 /
/// b +2.9). Hint: `T1`, not `T0` — the slab belongs in L2, not in the 48 KiB
/// L1 the composed table is already missing out of. Distance: the earliest
/// line is asked for ~2·(block_len/4) iterations (~0.2 ms) before the demand
/// load, and the latest ~0.05 ms before; both are far past a DRAM miss.
///
/// Read once per process, outside every loop.
#[cfg(target_arch = "x86_64")]
fn mdf4_pf_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_MDF4_PF").is_none());
    *ON
}

/// Same-binary rollback for the ranked DirectFold4 two-claim GFNI b-side.
/// Every other shape retains the table-hot scalar path below.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
fn direct_fold4_b_gfni_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_FOLD4_B_GFNI").is_none());
    *ON
}

#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )
))]
#[inline]
fn direct_fold4_b_gfni_shape(claims_len: usize, has_ordinary: bool, block_len: usize) -> bool {
    claims_len == 2 && !has_ordinary && block_len.is_multiple_of(64)
}

#[inline]
#[allow(dead_code)] // Reserved for the rollback DirectFold8 lookahead path.
fn eval_fold8_lookahead4(
    coefficients: &super::Fold8Lookahead4,
    r0: F128,
    r1: F128,
    r2: F128,
    r3: F128,
) -> SumcheckMessage {
    SumcheckMessage {
        u_0: eval_quadratic_tensor(&coefficients[..81], &[r0, r1, r2, r3]),
        u_2: eval_quadratic_tensor(&coefficients[81..], &[r0, r1, r2, r3]),
    }
}

#[inline]
#[allow(dead_code)] // Reserved for the rollback DirectFold8 lookahead path.
fn eval_fold8_lookahead5(
    coefficients: &super::Fold8Lookahead5,
    r0: F128,
    r1: F128,
    r2: F128,
    r3: F128,
    r4: F128,
) -> SumcheckMessage {
    SumcheckMessage {
        u_0: eval_quadratic_tensor(&coefficients[..243], &[r0, r1, r2, r3, r4]),
        u_2: eval_quadratic_tensor(&coefficients[243..], &[r0, r1, r2, r3, r4]),
    }
}

/// Sixteen-bank materializer (direct-fold4). Four challenges have been
/// sampled from the 16×16 product statistics; this binds the witness and the
/// direct basis in ONE N→N/16 pass and emits the round-4 message. Both ranked
/// claims are direct here (no ordinary basis), so the b side is two table-hot
/// phases of `fold_one_slot` exactly like the fold2 materializer, at a quarter
/// of the slots — first claim assigns (no memset of the uninit `take_f128`
/// chunk), later claims add. The ranked f side is `fold16_banked` (deferred
/// reduction); the nested pair-fold + mid buffer is only the fallback.
fn materialize_direct_fold4(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold4Factors],
    challenges: [F128; 4],
) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    let has_ordinary = !ordinary_basis.is_empty();
    assert!(!has_ordinary || ordinary_basis.len() == packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(16));
    let [r0, r1, r2, r3] = challenges;

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
        .par_iter()
        .map(|claim| {
            super::ring_switch::build_direct_fold4_table(&claim.low_eq, &fold_weight, &claim.table)
        })
        .collect();

    let out_len = packed_witness.len() / 16;
    let block_len = claims[0].eq_lo.len();
    assert!(block_len.is_multiple_of(4));
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && out_len == block_len * claim.eq_hi.len()
    }));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let b_gfni_on = direct_fold4_b_gfni_enabled()
        && direct_fold4_b_gfni_shape(claims.len(), has_ordinary, block_len);
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let b_gfni_on = false;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let direct_gfni_rows: Vec<(Vec<u64>, Vec<u64>)> = if b_gfni_on {
        claims
            .par_iter()
            .map(|claim| {
                (
                    claim.eq_lo.iter().map(|x| x.lo).collect(),
                    claim.eq_lo.iter().map(|x| x.hi).collect(),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    let table_len = super::ring_switch::FOLD_TABLE_TOTAL;
    // f-side sub-block: 256 output slots ⇒ 4096 inputs (64 KiB) → 1024 mids (16 KiB).
    // Ranked path has no ordinary basis and uses fold16_banked, so the mid
    // buffer is dead there — allocate it only when the nested b-side needs it.
    const SUB: usize = 256;
    let deferred_reduce = super::fold_deferred_reduce_enabled();
    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    let mid_len = if has_ordinary || !deferred_reduce {
        4 * SUB
    } else {
        0
    };
    // Grouped-gather prefetch state, resolved once for the whole
    // materialization — never inside the block / claim / slot loops.
    #[cfg(target_arch = "x86_64")]
    let pf_on = mdf4_pf_enabled();
    #[cfg(target_arch = "x86_64")]
    let n_blocks = out_len / block_len;
    let (u_0, u_2) = folded_b
        .par_chunks_mut(block_len)
        .zip(folded_f.par_chunks_mut(block_len))
        .enumerate()
        .map_init(
            || {
                (
                    vec![F128::ZERO; if b_gfni_on { 0 } else { table_len }],
                    vec![F128::ZERO; mid_len],
                )
            },
            |(scratch, mid), (block, (b_out, f_out))| {
                let start = 16 * block * block_len;
                let f_in = &packed_witness[start..start + 16 * block_len];
                // Head of the NEXT block's f-side slab, and how far into it
                // the b-side loops below may walk. Null when there is no next
                // block or the kill switch is set — the only check the loops
                // make. `block + 1 < n_blocks` is exactly the condition that
                // keeps the whole slab inside `packed_witness`:
                // `16·(block+2)·block_len ≤ 16·n_blocks·block_len = len`.
                #[cfg(target_arch = "x86_64")]
                let (pf_base, pf_span) = if pf_on && block + 1 < n_blocks {
                    // SAFETY: bounds argued above; `add` stays inside the
                    // allocation and the pointer is never dereferenced.
                    let p = unsafe { packed_witness.as_ptr().add(start + 16 * block_len) };
                    (
                        p.cast::<u8>(),
                        16 * block_len * core::mem::size_of::<F128>(),
                    )
                } else {
                    (core::ptr::null::<u8>(), 0usize)
                };
                #[cfg(target_arch = "x86_64")]
                let mut pf_at = 0usize;
                if deferred_reduce {
                    // ---- f: 16:1 in one deferred-reduction pass (one reduce
                    // per output lane; same field element as the nested form).
                    crate::field::f128_slice::fold16_banked(f_in, f_out, &fold_weight);
                } else {
                    // ---- f: 16:1 nested pair folds, sub-block at a time.
                    let mut slot = 0usize;
                    while slot < block_len {
                        let n = SUB.min(block_len - slot);
                        let mid = &mut mid[..4 * n];
                        crate::field::f128_slice::fold4_nested(
                            &f_in[16 * slot..16 * (slot + n)],
                            mid,
                            r0,
                            r1,
                        );
                        crate::field::f128_slice::fold4_nested(
                            mid,
                            &mut f_out[slot..slot + n],
                            r2,
                            r3,
                        );
                        slot += n;
                    }
                }
                // ---- b: ordinary basis (if any) folded 16:1 with the same weights.
                if has_ordinary {
                    let b_in = &ordinary_basis[start..start + 16 * block_len];
                    let mut slot = 0usize;
                    while slot < block_len {
                        let n = SUB.min(block_len - slot);
                        let mid = &mut mid[..4 * n];
                        crate::field::f128_slice::fold4_nested(
                            &b_in[16 * slot..16 * (slot + n)],
                            mid,
                            r0,
                            r1,
                        );
                        crate::field::f128_slice::fold4_nested(
                            mid,
                            &mut b_out[slot..slot + n],
                            r2,
                            r3,
                        );
                        slot += n;
                    }
                }
                // ---- b: direct claims. The ranked two-claim GFNI route folds
                // the low/high words of both maps together. Its no-prefetch
                // stack-plane schedule avoids disturbing the fused kernel's
                // 1 KiB temporary; every other geometry is the incumbent
                // table-hot scalar schedule.
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512vbmi",
                    target_feature = "vpclmulqdq",
                    target_feature = "gfni"
                ))]
                if b_gfni_on {
                    use crate::zerocheck::multilinear::kernels::x86_64::{
                        build_row_fold_mats_from_cols, gfni_fold64_four_maps_staged,
                    };
                    use core::arch::x86_64::_mm512_setzero_si512;

                    let (claim0, claim1) = (&claims[0], &claims[1]);
                    let cols0 = super::ring_switch::compose_block_cols(
                        &direct_tables[0],
                        claim0.eq_hi[block],
                    );
                    let mats0_lo = build_row_fold_mats_from_cols(&cols0[..64]);
                    let mats0_hi = build_row_fold_mats_from_cols(&cols0[64..]);
                    let cols1 = super::ring_switch::compose_block_cols(
                        &direct_tables[1],
                        claim1.eq_hi[block],
                    );
                    let mats1_lo = build_row_fold_mats_from_cols(&cols1[..64]);
                    let mats1_hi = build_row_fold_mats_from_cols(&cols1[64..]);
                    let (rows0, rows1) = (&direct_gfni_rows[0], &direct_gfni_rows[1]);
                    let mut planes = unsafe { [_mm512_setzero_si512(); 16] };
                    for slot in (0..block_len).step_by(64) {
                        // SAFETY: each row half supplies 512 bytes, the four
                        // maps cover 64 output slots, and the cfg gate
                        // supplies every feature required by the kernel.
                        unsafe {
                            gfni_fold64_four_maps_staged(
                                rows0.0.as_ptr().add(slot).cast::<u8>(),
                                &mats0_lo,
                                rows0.1.as_ptr().add(slot).cast::<u8>(),
                                &mats0_hi,
                                rows1.0.as_ptr().add(slot).cast::<u8>(),
                                &mats1_lo,
                                rows1.1.as_ptr().add(slot).cast::<u8>(),
                                &mats1_hi,
                                b_out.as_mut_ptr().add(slot),
                                planes.as_mut_ptr(),
                            );
                        }
                    }
                }
                if !b_gfni_on {
                    let table = &mut scratch[..table_len];
                    let mut claims_iter = claims.iter().zip(direct_tables.iter());
                    let (first, first_table) = claims_iter
                        .next()
                        .expect("materialize_direct_fold4: claims non-empty");
                    super::ring_switch::compose_block_table(first_table, first.eq_hi[block], table);
                    let mut s = 0usize;
                    while s + 3 < block_len {
                        #[cfg(target_arch = "x86_64")]
                        if !pf_base.is_null() && pf_at < pf_span {
                            // SAFETY: `pf_at < pf_span` and the slab is
                            // `pf_span` bytes, so the address is inside
                            // `packed_witness`. Prefetch has no architectural
                            // effect, so this arm is bit-identical to the
                            // kill-switched one.
                            unsafe {
                                core::arch::x86_64::_mm_prefetch(
                                    pf_base.add(pf_at).cast::<i8>(),
                                    core::arch::x86_64::_MM_HINT_T1,
                                );
                            }
                            pf_at += 64;
                        }
                        b_out[s] = super::ring_switch::fold_one_slot(first.eq_lo[s], table);
                        b_out[s + 1] = super::ring_switch::fold_one_slot(first.eq_lo[s + 1], table);
                        b_out[s + 2] = super::ring_switch::fold_one_slot(first.eq_lo[s + 2], table);
                        b_out[s + 3] = super::ring_switch::fold_one_slot(first.eq_lo[s + 3], table);
                        s += 4;
                    }
                    while s < block_len {
                        b_out[s] = super::ring_switch::fold_one_slot(first.eq_lo[s], table);
                        s += 1;
                    }
                    for (claim, direct_table) in claims_iter {
                        super::ring_switch::compose_block_table(
                            direct_table,
                            claim.eq_hi[block],
                            table,
                        );
                        let mut s = 0usize;
                        while s + 3 < block_len {
                            #[cfg(target_arch = "x86_64")]
                            if !pf_base.is_null() && pf_at < pf_span {
                                // SAFETY: `pf_at < pf_span` and the slab is
                                // `pf_span` bytes, so the address is inside
                                // `packed_witness`. Prefetch has no
                                // architectural effect.
                                unsafe {
                                    core::arch::x86_64::_mm_prefetch(
                                        pf_base.add(pf_at).cast::<i8>(),
                                        core::arch::x86_64::_MM_HINT_T1,
                                    );
                                }
                                pf_at += 64;
                            }
                            b_out[s] += super::ring_switch::fold_one_slot(claim.eq_lo[s], table);
                            b_out[s + 1] +=
                                super::ring_switch::fold_one_slot(claim.eq_lo[s + 1], table);
                            b_out[s + 2] +=
                                super::ring_switch::fold_one_slot(claim.eq_lo[s + 2], table);
                            b_out[s + 3] +=
                                super::ring_switch::fold_one_slot(claim.eq_lo[s + 3], table);
                            s += 4;
                        }
                        while s < block_len {
                            b_out[s] += super::ring_switch::fold_one_slot(claim.eq_lo[s], table);
                            s += 1;
                        }
                    }
                }
                // Vectorized message-term reduction over the folded chunk.
                #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
                {
                    // SAFETY: target features cfg-guaranteed; f_out/b_out
                    // have equal length (block_len, a multiple of 2).
                    unsafe { msg_reduce_avx512(f_out, b_out) }
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
                {
                    let mut u0 = F128::ZERO;
                    let mut u2 = F128::ZERO;
                    let mut k = 0;
                    while k + 1 < f_out.len() {
                        let f0 = f_out[k];
                        let f1 = f_out[k + 1];
                        let b0 = b_out[k];
                        let b1 = b_out[k + 1];
                        u0 += f0 * b0;
                        u2 += (f0 + f1) * (b0 + b1);
                        k += 2;
                    }
                    (u0, u2)
                }
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );
    crate::scratch::give_f128(packed_witness);
    if has_ordinary {
        crate::scratch::give_f128(ordinary_basis);
    }
    (folded_f, folded_b, SumcheckMessage { u_0, u_2 })
}

/// Materialize the combined state only after the first two folds. Direct
/// claims contain AB and optionally C in sufficient-stat form; a non-empty
/// ordinary basis is the direct-AB-only fallback. All contributions already
/// have their ring-switch batching challenge baked in.
#[allow(clippy::assign_op_pattern)] // Preserve the hand-scheduled indexed stores.
fn materialize_direct_ab_fold2(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold2Factors],
    r0: F128,
    r1: F128,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    assert!(ordinary_basis.is_empty() || ordinary_basis.len() == packed_witness.len());
    let fold_weight = [
        (F128::ONE + r0) * (F128::ONE + r1),
        r0 * (F128::ONE + r1),
        (F128::ONE + r0) * r1,
        r0 * r1,
    ];
    // Two claims on ranked path: build both fold2 tables in parallel.
    let direct_tables: Vec<Vec<F128>> = claims
        .par_iter()
        .map(|claim| {
            super::ring_switch::build_direct_fold2_table(&claim.low_eq, &fold_weight, &claim.table)
        })
        .collect();

    let out_len = packed_witness.len() / 4;
    let block_len = claims[0].eq_lo.len();
    assert!(block_len.is_multiple_of(2));
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && out_len == block_len * claim.eq_hi.len()
    }));
    let table_len = super::ring_switch::FOLD_TABLE_TOTAL;
    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    let (u_0, u_2) = folded_b
        .par_chunks_mut(block_len)
        .zip(folded_f.par_chunks_mut(block_len))
        .enumerate()
        .map_init(
            // Table-hot 2-claim keeps one 64 KiB composed table live per phase.
            //
            // Recycled through this thread's free list rather than allocated,
            // zeroed and freed once per job: rayon calls this init once per
            // leaf job, i.e. inside the fold loop, and every slot is written
            // by `compose_block_table` before any read (which that function
            // documents). `FLOCK_NO_PCS_FOLD_BUF_POOL=1` restores the
            // allocating form.
            || {
                crate::scratch::LocalBuf::new(
                    if claims.len() == 2 {
                        table_len
                    } else {
                        claims.len() * table_len
                    },
                    crate::scratch::fold_buf_pool_enabled(),
                )
            },
            |scratch, (block, (b_out, f_out))| {
                // Production 2-claim table-hot path composes inside each phase.
                if claims.len() != 2 {
                    for (claim_index, (claim, direct_table)) in
                        claims.iter().zip(direct_tables.iter()).enumerate()
                    {
                        super::ring_switch::compose_block_table(
                            direct_table,
                            claim.eq_hi[block],
                            &mut scratch[claim_index * table_len..(claim_index + 1) * table_len],
                        );
                    }
                }
                let start = 4 * block * block_len;
                let f_in = &packed_witness[start..start + 4 * block_len];
                let b_in = (!ordinary_basis.is_empty())
                    .then(|| &ordinary_basis[start..start + 4 * block_len]);
                // In-register nested pair-fold (r0 then r1). Writes f_out only;
                // no mid 2·block_len buffer. fold_one_slot / phase-2 / MAC stay.
                crate::field::f128_slice::fold4_nested(f_in, f_out, r0, r1);
                let fold4 = |input: &[F128], slot: usize| {
                    let a0 = input[4 * slot];
                    let a1 = input[4 * slot + 1];
                    let a2 = input[4 * slot + 2];
                    let a3 = input[4 * slot + 3];
                    let low = a0 + r0 * (a0 + a1);
                    let high = a2 + r0 * (a2 + a3);
                    low + r1 * (low + high)
                };
                if let [only] = claims {
                    // Single-claim specialization.
                    let table = &scratch[..table_len];
                    for pair in 0..(block_len / 2) {
                        let slot0 = 2 * pair;
                        let slot1 = slot0 + 1;
                        let b0 = super::ring_switch::fold_one_slot(only.eq_lo[slot0], table)
                            + b_in.map_or(F128::ZERO, |basis| fold4(basis, slot0));
                        let b1 = super::ring_switch::fold_one_slot(only.eq_lo[slot1], table)
                            + b_in.map_or(F128::ZERO, |basis| fold4(basis, slot1));
                        b_out[slot0] = b0;
                        b_out[slot1] = b1;
                    }
                } else if let [first, second] = claims {
                    debug_assert!(b_in.is_none());
                    // Table-hot two-phase: only one 64 KiB composed table live.
                    // Phase 1: first table hot → partial b (f_out already written).
                    // Phase 2: second table hot → complete b.
                    // 2× pair unroll keeps two slot pairs in flight per iteration.
                    let table = &mut scratch[..table_len];
                    super::ring_switch::compose_block_table(
                        &direct_tables[0],
                        first.eq_hi[block],
                        table,
                    );
                    let n_pairs = block_len / 2;
                    let mut pair = 0usize;
                    while pair + 1 < n_pairs {
                        let s0 = 2 * pair;
                        let s1 = s0 + 1;
                        let s2 = s0 + 2;
                        let s3 = s0 + 3;
                        b_out[s0] = super::ring_switch::fold_one_slot(first.eq_lo[s0], table);
                        b_out[s1] = super::ring_switch::fold_one_slot(first.eq_lo[s1], table);
                        b_out[s2] = super::ring_switch::fold_one_slot(first.eq_lo[s2], table);
                        b_out[s3] = super::ring_switch::fold_one_slot(first.eq_lo[s3], table);
                        pair += 2;
                    }
                    if pair < n_pairs {
                        let s0 = 2 * pair;
                        let s1 = s0 + 1;
                        b_out[s0] = super::ring_switch::fold_one_slot(first.eq_lo[s0], table);
                        b_out[s1] = super::ring_switch::fold_one_slot(first.eq_lo[s1], table);
                    }
                    super::ring_switch::compose_block_table(
                        &direct_tables[1],
                        second.eq_hi[block],
                        table,
                    );
                    pair = 0;
                    while pair + 1 < n_pairs {
                        let s0 = 2 * pair;
                        let s1 = s0 + 1;
                        let s2 = s0 + 2;
                        let s3 = s0 + 3;
                        b_out[s0] =
                            b_out[s0] + super::ring_switch::fold_one_slot(second.eq_lo[s0], table);
                        b_out[s1] =
                            b_out[s1] + super::ring_switch::fold_one_slot(second.eq_lo[s1], table);
                        b_out[s2] =
                            b_out[s2] + super::ring_switch::fold_one_slot(second.eq_lo[s2], table);
                        b_out[s3] =
                            b_out[s3] + super::ring_switch::fold_one_slot(second.eq_lo[s3], table);
                        pair += 2;
                    }
                    if pair < n_pairs {
                        let s0 = 2 * pair;
                        let s1 = s0 + 1;
                        b_out[s0] =
                            b_out[s0] + super::ring_switch::fold_one_slot(second.eq_lo[s0], table);
                        b_out[s1] =
                            b_out[s1] + super::ring_switch::fold_one_slot(second.eq_lo[s1], table);
                    }
                } else {
                    for pair in 0..(block_len / 2) {
                        let slot0 = 2 * pair;
                        let slot1 = slot0 + 1;
                        let direct_at = |slot: usize| {
                            claims
                                .iter()
                                .enumerate()
                                .map(|(claim_index, claim)| {
                                    super::ring_switch::fold_one_slot(
                                        claim.eq_lo[slot],
                                        &scratch[claim_index * table_len
                                            ..(claim_index + 1) * table_len],
                                    )
                                })
                                .fold(F128::ZERO, |sum, value| sum + value)
                        };
                        let b0 =
                            direct_at(slot0) + b_in.map_or(F128::ZERO, |basis| fold4(basis, slot0));
                        let b1 =
                            direct_at(slot1) + b_in.map_or(F128::ZERO, |basis| fold4(basis, slot1));
                        b_out[slot0] = b0;
                        b_out[slot1] = b1;
                    }
                }
                // Vectorized message-term reduction over the folded chunk.
                #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
                {
                    // SAFETY: target features cfg-guaranteed; f_out/b_out
                    // have equal length (block_len, a multiple of 2).
                    unsafe { msg_reduce_avx512(f_out, b_out) }
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
                {
                    let mut u0 = F128::ZERO;
                    let mut u2 = F128::ZERO;
                    let mut k = 0;
                    while k + 1 < f_out.len() {
                        let f0 = f_out[k];
                        let f1 = f_out[k + 1];
                        let b0 = b_out[k];
                        let b1 = b_out[k + 1];
                        u0 += f0 * b0;
                        u2 += (f0 + f1) * (b0 + b1);
                        k += 2;
                    }
                    (u0, u2)
                }
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );
    crate::scratch::give_f128(packed_witness);
    if !ordinary_basis.is_empty() {
        crate::scratch::give_f128(ordinary_basis);
    }
    (folded_f, folded_b, SumcheckMessage { u_0, u_2 })
}

/// NT leaf for one [`fold_and_msg_lsb`] chunk: fold `f`/`b` at `r` AND build
/// the (u_0, u_2) message terms from the register values, publishing the
/// folded pairs with `stnp q,q` non-temporal stores — no write-allocate, no
/// reload of the just-written pairs. Value-identical to the generic chunk
/// body: the fold uses the same `ghash_mul_vec2_neon` pair fold as
/// `f128_slice::fold_pairs`, and the message terms are the same reduced
/// products XOR-accumulated (order-independent in GF(2^128)).
///
/// # Safety
/// Requires the `aes` target feature. `fc`/`bc` must have equal, even length;
/// `f`/`b` must contain `2 * (base + fc.len())` elements.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes")]
unsafe fn fold_and_msg_chunk_nt_neon(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    /// `stnp q,q` of two adjacent F128s from NEON registers (no Rust
    /// intrinsic emits `stnp`). `dst` must be valid for 32 bytes, 16-aligned.
    #[inline(always)]
    unsafe fn store_nt_pair(dst: *mut F128, v0: F128, v1: F128) {
        // SAFETY: F128 is a plain 16-byte value; transmute to a NEON register
        // preserves the (lo LE ‖ hi LE) byte layout the store publishes.
        unsafe {
            let q0: core::arch::aarch64::uint8x16_t = core::mem::transmute(v0);
            let q1: core::arch::aarch64::uint8x16_t = core::mem::transmute(v1);
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{p}]",
                a = in(vreg) q0,
                b = in(vreg) q1,
                p = in(reg) dst,
                options(nostack, preserves_flags)
            );
        }
    }

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    debug_assert!(len.is_multiple_of(2));
    let mut u0 = F128::ZERO;
    let mut u2 = F128::ZERO;
    unsafe {
        let mut src_f = f.as_ptr().add(2 * base);
        let mut src_b = b.as_ptr().add(2 * base);
        let mut dst_f = fc.as_mut_ptr();
        let mut dst_b = bc.as_mut_ptr();
        let mut remaining = len / 2;
        while remaining != 0 {
            let fe0 = src_f.read();
            let fo0 = src_f.add(1).read();
            let fe1 = src_f.add(2).read();
            let fo1 = src_f.add(3).read();
            let be0 = src_b.read();
            let bo0 = src_b.add(1).read();
            let be1 = src_b.add(2).read();
            let bo1 = src_b.add(3).read();

            // fold(e, o) = e + r · (e ⊕ o), two lanes per array — the same
            // arithmetic as `f128_slice::fold_pairs`.
            let pf = ghash_mul_vec2_neon(
                [r, r],
                [
                    F128 {
                        lo: fe0.lo ^ fo0.lo,
                        hi: fe0.hi ^ fo0.hi,
                    },
                    F128 {
                        lo: fe1.lo ^ fo1.lo,
                        hi: fe1.hi ^ fo1.hi,
                    },
                ],
            );
            let pb = ghash_mul_vec2_neon(
                [r, r],
                [
                    F128 {
                        lo: be0.lo ^ bo0.lo,
                        hi: be0.hi ^ bo0.hi,
                    },
                    F128 {
                        lo: be1.lo ^ bo1.lo,
                        hi: be1.hi ^ bo1.hi,
                    },
                ],
            );
            let f0 = F128 {
                lo: fe0.lo ^ pf[0].lo,
                hi: fe0.hi ^ pf[0].hi,
            };
            let f1 = F128 {
                lo: fe1.lo ^ pf[1].lo,
                hi: fe1.hi ^ pf[1].hi,
            };
            let b0 = F128 {
                lo: be0.lo ^ pb[0].lo,
                hi: be0.hi ^ pb[0].hi,
            };
            let b1 = F128 {
                lo: be1.lo ^ pb[1].lo,
                hi: be1.hi ^ pb[1].hi,
            };

            store_nt_pair(dst_f, f0, f1);
            store_nt_pair(dst_b, b0, b1);

            // u_0 += f0·b0, u_2 += (f0+f1)(b0+b1) — from registers.
            let g = ghash_mul_vec2_neon([f0, f0 + f1], [b0, b0 + b1]);
            u0 += g[0];
            u2 += g[1];

            src_f = src_f.add(4);
            src_b = src_b.add(4);
            dst_f = dst_f.add(2);
            dst_b = dst_b.add(2);
            remaining -= 1;
        }
    }
    (u0, u2)
}

/// All-NEON SoA variant of [`fold_and_msg_chunk_nt_neon`].
///
/// Same values, same store order — but restructured for M4's NEON issue
/// width, which is what actually caps this loop (the fold rounds scale
/// exactly linearly with size AND with thread count, so the kernel is
/// core-issue-bound, not memory-bound: ~19 GB/s per core vs the ~100 GB/s a
/// single M4 core can stream). Three restructures against the original:
///
/// 1. **All-vector dataflow.** The original keeps `F128 {lo, hi}` in GPRs
///    and calls `ghash_mul_vec2_neon(F128, …)` — ~20 GPR→NEON `fmov`s per
///    iteration. Here loads land in q-registers (`vld1q`), pair-XORs are
///    `veor`, PMULL/PMULL2 read lane-paired (SoA) operands with no moves.
/// 2. **Karatsuba fold muls.** Both fold multiplications share the constant
///    `r`, so `r.lo ⊕ r.hi` is hoisted; each lane-paired fold is 6 PMULLs
///    (3 per mul) instead of the schoolbook 8, cross terms via `EOR3`.
///    Identical output: `dm ⊕ d0 ⊕ d2 = lh ⊕ hl` is exact in F2, and the
///    same shift-based mod-p reduction produces the canonical value.
/// 3. **Deferred message reduction.** `(u_0, u_2)` accumulate as UNREDUCED
///    Karatsuba halves (`Σd0, Σdm, Σd2` per message word, 6 XORs/iter) and
///    are reduced ONCE per chunk: mod-p reduction is F2-linear, so
///    `reduce(Σ unreduced) = Σ reduce(each)` bit-exactly (the same idiom the
///    x86_64 `f128_slice` message path documents). Kills a 17-op shift
///    reduction + 4 zips per iteration.
///
/// # Safety
/// Requires the `aes` and `sha3` target features (PMULL, EOR3). `fc`/`bc`
/// must have equal, even length; `f`/`b` must contain `2 * (base + fc.len())`
/// elements.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes,sha3")]
unsafe fn fold_and_msg_chunk_nt_neon_soa<const NT: bool>(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use core::arch::aarch64::*;

    /// `pmull` on lane 0 of both operands, staying in the vector file (the
    /// intrinsic route through `p64` scalars can round-trip through GPRs).
    #[inline(always)]
    unsafe fn pmull_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        let d: uint64x2_t;
        unsafe {
            core::arch::asm!(
                "pmull {d:v}.1q, {a:v}.1d, {b:v}.1d",
                d = lateout(vreg) d,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        d
    }
    /// `pmull2` on lane 1 of both operands.
    #[inline(always)]
    unsafe fn pmull_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        let d: uint64x2_t;
        unsafe {
            core::arch::asm!(
                "pmull2 {d:v}.1q, {a:v}.2d, {b:v}.2d",
                d = lateout(vreg) d,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        d
    }

    /// Lane-paired Karatsuba **+ Barrett** fold multiply by the loop
    /// constant `r`: given the SoA pair-XOR words `(d_lo, d_hi)` of two
    /// F128s `d0, d1`, returns the two reduced products `(r·d0, r·d1)`
    /// directly in AoS form (one `[lo, hi]` vector each) — no pack/unpack
    /// zips. 6 Karatsuba PMULLs + 6 Barrett PMULLs (fold `hi·0x87`, then the
    /// ≤7-bit overflow `ov·0x87` — `0x87 = x⁷+x²+x+1`, so `ov·0x87` IS the
    /// shift correction `ov ⊕ ov≪1 ⊕ ov≪2 ⊕ ov≪7`), replacing the ~26-op
    /// vectorised shift reduction. Word-for-word the arithmetic of
    /// [`ghash_mul_karatsuba_barrett`]; the canonical mod-p value is unique,
    /// so results are bit-identical to every other mul variant
    /// (`all_neon_variants_agree` pins this).
    ///
    /// [`ghash_mul_karatsuba_barrett`]: crate::field::gf2_128::aarch64::ghash_mul_karatsuba_barrett
    #[inline(always)]
    unsafe fn mul2_kara_barrett_aos(
        d_lo: uint64x2_t,
        d_hi: uint64x2_t,
        r_lo: uint64x2_t,
        r_hi: uint64x2_t,
        r_sum: uint64x2_t,
        c87: uint64x2_t,
    ) -> (uint64x2_t, uint64x2_t) {
        unsafe {
            let d_sum = veorq_u64(d_lo, d_hi);
            let p0_0 = pmull_lo(d_lo, r_lo);
            let p0_1 = pmull_hi(d_lo, r_lo);
            let p2_0 = pmull_lo(d_hi, r_hi);
            let p2_1 = pmull_hi(d_hi, r_hi);
            let pm_0 = pmull_lo(d_sum, r_sum);
            let pm_1 = pmull_hi(d_sum, r_sum);
            // Cross terms c = dm ⊕ d0 ⊕ d2 (≡ lh ⊕ hl).
            let c_0 = veor3q_u64(pm_0, p0_0, p2_0);
            let c_1 = veor3q_u64(pm_1, p0_1, p2_1);

            // Per-lane 256-bit product halves:
            //   lo128 = d0 ⊕ (c ≪ 64), hi128 = d2 ⊕ (c ≫ 64).
            let zero = vdupq_n_u64(0);
            let lo_0 = veorq_u64(p0_0, vextq_u64::<1>(zero, c_0));
            let hi_0 = veorq_u64(p2_0, vextq_u64::<1>(c_0, zero));
            let lo_1 = veorq_u64(p0_1, vextq_u64::<1>(zero, c_1));
            let hi_1 = veorq_u64(p2_1, vextq_u64::<1>(c_1, zero));

            // Barrett fold of hi128: r_lo = hi.lo·0x87, r_hi = hi.hi·0x87,
            // corr = ov·0x87 with ov = r_hi.hi (≤ 7 bits, product ≤ 14 bits
            // so it lands entirely in the low word).
            let rl_0 = pmull_lo(hi_0, c87);
            let rh_0 = pmull_hi(hi_0, c87);
            let rl_1 = pmull_lo(hi_1, c87);
            let rh_1 = pmull_hi(hi_1, c87);
            let cor_0 = pmull_hi(rh_0, c87);
            let cor_1 = pmull_hi(rh_1, c87);

            // res.lo = lo128.lo ⊕ r_lo.lo ⊕ corr,
            // res.hi = lo128.hi ⊕ r_lo.hi ⊕ r_hi.lo.
            let res_0 = veor3q_u64(lo_0, rl_0, vzip1q_u64(cor_0, rh_0));
            let res_1 = veor3q_u64(lo_1, rl_1, vzip1q_u64(cor_1, rh_1));
            (res_0, res_1)
        }
    }

    /// Pair store of two adjacent F128s straight from vector registers:
    /// `stnp q,q` (non-temporal, no write-allocate) when `NT`, else a plain
    /// `stp q,q` (small rounds re-read their output next round while it is
    /// still cache-resident, so the allocate is free and NT would forfeit
    /// the hits). `dst` must be valid for 32 bytes, 16-aligned.
    #[inline(always)]
    unsafe fn store_pair_v<const NT: bool>(dst: *mut F128, v0: uint64x2_t, v1: uint64x2_t) {
        unsafe {
            if NT {
                core::arch::asm!(
                    "stnp {a:q}, {b:q}, [{p}]",
                    a = in(vreg) v0,
                    b = in(vreg) v1,
                    p = in(reg) dst,
                    options(nostack, preserves_flags)
                );
            } else {
                core::arch::asm!(
                    "stp {a:q}, {b:q}, [{p}]",
                    a = in(vreg) v0,
                    b = in(vreg) v1,
                    p = in(reg) dst,
                    options(nostack, preserves_flags)
                );
            }
        }
    }

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    debug_assert!(len.is_multiple_of(2));
    unsafe {
        let mut src_f = f.as_ptr().add(2 * base) as *const u64;
        let mut src_b = b.as_ptr().add(2 * base) as *const u64;
        let mut dst_f = fc.as_mut_ptr();
        let mut dst_b = bc.as_mut_ptr();
        // r broadcast once: lane-paired lo/hi/sum words for the fold muls.
        let r_lo = vdupq_n_u64(r.lo);
        let r_hi = vdupq_n_u64(r.hi);
        let r_sum = vdupq_n_u64(r.lo ^ r.hi);
        // Barrett fold constant: p − x^128 reversed = x⁷+x²+x+1.
        let c87 = vdupq_n_u64(0x87);
        // Unreduced SoA message accumulators: lane 0 = u_0, lane 1 = u_2.
        // Karatsuba halves of Σ f·b: Σd0 (lo·lo), Σdm (sum·sum), Σd2 (hi·hi),
        // each a full 128-bit carry-less product per lane.
        let mut acc_d0_0 = vdupq_n_u64(0);
        let mut acc_d0_1 = vdupq_n_u64(0);
        let mut acc_dm_0 = vdupq_n_u64(0);
        let mut acc_dm_1 = vdupq_n_u64(0);
        let mut acc_d2_0 = vdupq_n_u64(0);
        let mut acc_d2_1 = vdupq_n_u64(0);
        let mut remaining = len / 2;
        while remaining != 0 {
            let fe0 = vld1q_u64(src_f);
            let fo0 = vld1q_u64(src_f.add(2));
            let fe1 = vld1q_u64(src_f.add(4));
            let fo1 = vld1q_u64(src_f.add(6));
            let be0 = vld1q_u64(src_b);
            let bo0 = vld1q_u64(src_b.add(2));
            let be1 = vld1q_u64(src_b.add(4));
            let bo1 = vld1q_u64(src_b.add(6));

            // fold(e, o) = e + r · (e ⊕ o), two lanes per array; the Barrett
            // mul returns each product in AoS form, ready to XOR and store.
            let fd0 = veorq_u64(fe0, fo0);
            let fd1 = veorq_u64(fe1, fo1);
            let (pf0, pf1) = mul2_kara_barrett_aos(
                vzip1q_u64(fd0, fd1),
                vzip2q_u64(fd0, fd1),
                r_lo,
                r_hi,
                r_sum,
                c87,
            );
            let bd0 = veorq_u64(be0, bo0);
            let bd1 = veorq_u64(be1, bo1);
            let (pb0, pb1) = mul2_kara_barrett_aos(
                vzip1q_u64(bd0, bd1),
                vzip2q_u64(bd0, bd1),
                r_lo,
                r_hi,
                r_sum,
                c87,
            );
            let f0 = veorq_u64(fe0, pf0);
            let f1 = veorq_u64(fe1, pf1);
            let b0 = veorq_u64(be0, pb0);
            let b1 = veorq_u64(be1, pb1);

            store_pair_v::<NT>(dst_f, f0, f1);
            store_pair_v::<NT>(dst_b, b0, b1);

            // u_0 += f0·b0, u_2 += (f0+f1)(b0+b1) — Karatsuba halves
            // accumulated UNREDUCED; reduced once after the loop.
            let fs = veorq_u64(f0, f1);
            let bs = veorq_u64(b0, b1);
            let a_lo = vzip1q_u64(f0, fs);
            let a_hi = vzip2q_u64(f0, fs);
            let b_lo = vzip1q_u64(b0, bs);
            let b_hi = vzip2q_u64(b0, bs);
            let a_sum = veorq_u64(a_lo, a_hi);
            let b_sum = veorq_u64(b_lo, b_hi);
            acc_d0_0 = veorq_u64(acc_d0_0, pmull_lo(a_lo, b_lo));
            acc_d0_1 = veorq_u64(acc_d0_1, pmull_hi(a_lo, b_lo));
            acc_dm_0 = veorq_u64(acc_dm_0, pmull_lo(a_sum, b_sum));
            acc_dm_1 = veorq_u64(acc_dm_1, pmull_hi(a_sum, b_sum));
            acc_d2_0 = veorq_u64(acc_d2_0, pmull_lo(a_hi, b_hi));
            acc_d2_1 = veorq_u64(acc_d2_1, pmull_hi(a_hi, b_hi));

            src_f = src_f.add(8);
            src_b = src_b.add(8);
            dst_f = dst_f.add(2);
            dst_b = dst_b.add(2);
            remaining -= 1;
        }

        // Final Karatsuba combine + single mod-p reduction per message word.
        // Reduction is F2-linear, so this equals the sum of per-pair reduced
        // products bit-for-bit.
        #[inline(always)]
        unsafe fn finish(d0: uint64x2_t, dm: uint64x2_t, d2: uint64x2_t) -> F128 {
            unsafe {
                let c = veor3q_u64(dm, d0, d2);
                crate::field::gf2_128::ghash_reduce(
                    vgetq_lane_u64::<0>(d0),
                    vgetq_lane_u64::<1>(d0) ^ vgetq_lane_u64::<0>(c),
                    vgetq_lane_u64::<0>(d2) ^ vgetq_lane_u64::<1>(c),
                    vgetq_lane_u64::<1>(d2),
                )
            }
        }
        (
            finish(acc_d0_0, acc_dm_0, acc_d2_0),
            finish(acc_d0_1, acc_dm_1, acc_d2_1),
        )
    }
}

/// Per-open bump arena for the sumcheck fold outputs.
///
/// Every fold round's output size is known when the open starts: round `j`
/// (1-based) of an `l`-slot open produces two `l >> j` buffers, so the
/// `initial_k` L0 rounds need exactly `2·(l/2 + … + l/2^k) = 2·(l − l/2^k)`
/// F128s in total. One allocation of that size replaces the two fresh
/// `alloc_uninit` buffers per round, and its pages are prefaulted by
/// background threads spawned at open entry — the kernel's ~1 GiB zero-fill
/// (at the ranked m=32 shape) overlaps the `b_combined` build instead of
/// being paid fault-by-fault inside the serial Fiat–Shamir fold chain.
///
/// Prefaulting is front-to-back per contiguous partition, with a per-
/// partition watermark. [`Self::carve_pair`] NEVER blocks: a carve succeeds
/// only if its whole region is already faulted; otherwise the round falls
/// back to a fresh allocation (exactly the previous behavior) and the region
/// stays at the front for the next, smaller round — so a slow prefaulter
/// degrades gracefully instead of stalling the fold chain (at worst the
/// arena's tail goes unused).
///
/// Strictly per-open: created in `pcs::open_batch_mixed_ligerito…`, moved
/// into the [`SumcheckProver`], freed when the prover drops. Never recycled
/// across proves (exact-size, no retention). `FLOCK_NO_FOLD_ARENA` disables
/// creation (local diagnostics; the ranked worker's cleared env never sets
/// it).
pub struct FoldArena {
    ptr: std::ptr::NonNull<F128>,
    /// Capacity in F128 elements.
    cap: usize,
    /// Bump offset in F128 elements; only ever grows, so carved regions are
    /// pairwise disjoint.
    offset: usize,
    /// Per-partition prefault watermarks (absolute element index reached).
    parts: Vec<PrefaultPart>,
    /// Prefault threads; joined in Drop (they write into the allocation).
    threads: Vec<std::thread::JoinHandle<()>>,
}

/// One contiguous prefault partition `[start, end)` (element indices) and
/// the watermark its thread has faulted up to (monotone, `start` → `end`).
struct PrefaultPart {
    start: usize,
    end: usize,
    done: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

// SAFETY: the arena is a plain owned allocation + bump offset; F128 is Send.
unsafe impl Send for FoldArena {}

impl FoldArena {
    /// Elements needed for every parallel fold round of an `l`-element open
    /// with `k` initial lane folds: `2 · (l/2 + l/4 + … + l/2^k)`.
    pub fn capacity_for(l: usize, k: usize) -> usize {
        debug_assert!(l.is_power_of_two());
        debug_assert!(k >= 1 && k <= l.trailing_zeros() as usize);
        2 * (l - (l >> k))
    }

    /// Allocate `cap` F128s (uninitialized) and start touching one byte per
    /// page across `PREFAULT_THREADS` contiguous partitions so the kernel's
    /// zero-fill overlaps the caller's compute.
    pub fn new_prefaulted(cap: usize) -> Self {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        assert!(cap > 0);
        let layout = std::alloc::Layout::array::<F128>(cap).expect("FoldArena layout");
        // SAFETY: layout is non-zero-sized (cap > 0, F128 is 16 bytes).
        let raw = unsafe { std::alloc::alloc(layout) } as *mut F128;
        let Some(ptr) = std::ptr::NonNull::new(raw) else {
            std::alloc::handle_alloc_error(layout);
        };
        struct SendPtr(*mut u8);
        // SAFETY: each thread dereferences only its own disjoint partition,
        // and the arena joins every thread before dealloc.
        unsafe impl Send for SendPtr {}

        // 3 threads: round 0's region is the front HALF of the arena and is
        // needed first (right after the b_combined pass + a ~19-bit grind);
        // splitting the front across two threads roughly halves its ready
        // time, while the third covers the later rounds' tail.
        const PREFAULT_THREADS: usize = 3;
        // Page-touch stride: 16 KiB matches Apple Silicon pages; elsewhere
        // 4 KiB pages are still fully faulted because partition SIZES are
        // 16 KiB-multiples only at the front — use 4 KiB to stay correct.
        #[cfg(target_os = "macos")]
        const STRIDE: usize = 16384;
        #[cfg(not(target_os = "macos"))]
        const STRIDE: usize = 4096;
        const STRIDE_ELEMS: usize = STRIDE / core::mem::size_of::<F128>();
        /// Publish the watermark every 4 MiB of progress.
        const CHUNK_ELEMS: usize = (4 << 20) / core::mem::size_of::<F128>();

        let bound = |i: usize| -> usize {
            // Even element split, rounded to a whole stride so every page
            // belongs to exactly one thread.
            let raw = cap * i / PREFAULT_THREADS;
            (raw / STRIDE_ELEMS) * STRIDE_ELEMS
        };
        let mut parts = Vec::with_capacity(PREFAULT_THREADS);
        let mut threads = Vec::with_capacity(PREFAULT_THREADS);
        for i in 0..PREFAULT_THREADS {
            let (start, end) = (
                bound(i),
                if i + 1 == PREFAULT_THREADS {
                    cap
                } else {
                    bound(i + 1)
                },
            );
            let done = Arc::new(AtomicUsize::new(start));
            parts.push(PrefaultPart {
                start,
                end,
                done: done.clone(),
            });
            if start >= end {
                done.store(end, Ordering::Release);
                continue;
            }
            let base = SendPtr(unsafe { raw.add(start) } as *mut u8);
            let elems = end - start;
            threads.push(std::thread::spawn(move || {
                let base = base;
                let mut e = 0usize;
                while e < elems {
                    let chunk_end = (e + CHUNK_ELEMS).min(elems);
                    while e < chunk_end {
                        // SAFETY: e < elems, within this thread's partition;
                        // volatile so the faulting store is not elided.
                        unsafe {
                            std::ptr::write_volatile(
                                base.0.add(e * core::mem::size_of::<F128>()),
                                0u8,
                            )
                        };
                        e += STRIDE_ELEMS;
                    }
                    let e = e.min(elems);
                    // Release: orders the page-faulting stores above before
                    // the watermark readers' carves.
                    done.store(start + e, Ordering::Release);
                }
                done.store(end, Ordering::Release);
            }));
        }
        Self {
            ptr,
            cap,
            offset: 0,
            parts,
            threads,
        }
    }

    /// Carve two disjoint `half`-element buffers, or `None` if the arena is
    /// exhausted OR the region is not fully prefaulted yet (the caller then
    /// falls back to a fresh allocation and the region is retried by the
    /// next round). Never blocks.
    fn carve_pair(&mut self, half: usize) -> Option<(FoldBuf, FoldBuf)> {
        use std::sync::atomic::Ordering;
        if self.cap - self.offset < 2 * half {
            return None;
        }
        let (lo, hi) = (self.offset, self.offset + 2 * half);
        let ready = self.parts.iter().all(|p| {
            hi <= p.start || lo >= p.end || p.done.load(Ordering::Acquire) >= hi.min(p.end)
        });
        if !ready {
            return None;
        }
        // SAFETY: offset + 2·half ≤ cap, so both regions lie inside the
        // allocation; the bump offset only grows, so they are disjoint from
        // every previously carved region. The Acquire watermark loads above
        // guarantee the prefaulter is done with (and will never revisit)
        // [lo, hi), so no concurrent writes alias the carved slices.
        let a = unsafe { self.ptr.add(self.offset) };
        let b = unsafe { self.ptr.add(self.offset + half) };
        self.offset += 2 * half;
        Some((
            FoldBuf::Arena { ptr: a, len: half },
            FoldBuf::Arena { ptr: b, len: half },
        ))
    }
}

impl Drop for FoldArena {
    fn drop(&mut self) {
        // The prefault threads write into the allocation — they MUST finish
        // before dealloc.
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
        let layout = std::alloc::Layout::array::<F128>(self.cap).expect("FoldArena layout");
        // SAFETY: ptr/layout are exactly what `new_prefaulted` allocated.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout) };
    }
}

/// Storage for one sumcheck fold buffer: an owned heap `Vec` or a region
/// carved from the per-open [`FoldArena`].
///
/// The `Arena` variant is a plain (ptr, len) view: dropping it is a no-op —
/// the arena owns the memory and outlives every carved buffer (both live in
/// the same [`SumcheckProver`], and `pcs` moves the arena into the prover
/// before any carve).
pub(crate) enum FoldBuf {
    Owned(Vec<F128>),
    Arena {
        ptr: std::ptr::NonNull<F128>,
        len: usize,
    },
}

// SAFETY: `Arena` regions are pairwise disjoint (bump carve) and uniquely
// owned by their FoldBuf; F128 is Send + Sync plain data.
unsafe impl Send for FoldBuf {}
unsafe impl Sync for FoldBuf {}

impl Default for FoldBuf {
    fn default() -> Self {
        FoldBuf::Owned(Vec::new())
    }
}

impl std::ops::Deref for FoldBuf {
    type Target = [F128];
    #[inline]
    fn deref(&self) -> &[F128] {
        match self {
            FoldBuf::Owned(v) => v,
            // SAFETY: the region lies inside the arena allocation, which
            // outlives this buffer; no other FoldBuf aliases it.
            FoldBuf::Arena { ptr, len } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), *len)
            },
        }
    }
}

impl std::ops::DerefMut for FoldBuf {
    #[inline]
    fn deref_mut(&mut self) -> &mut [F128] {
        match self {
            FoldBuf::Owned(v) => v,
            // SAFETY: as in Deref, plus &mut self guarantees exclusivity.
            FoldBuf::Arena { ptr, len } => unsafe {
                std::slice::from_raw_parts_mut(ptr.as_ptr(), *len)
            },
        }
    }
}

/// Exact ranked-production selector for retaining the L1 OOD equality as a
/// tensor product. Every other profile, dimension, sample count, direct-fold
/// mode, platform, and the explicit kill switch keep the dense-table path.
#[inline]
fn ranked_l1_lazy_ood_eq_enabled(
    config: &ProverConfig,
    log_n: usize,
    n_1: usize,
    l1_ood_count: usize,
    current_len: usize,
    direct_fold4_mode: bool,
) -> bool {
    ranked_l1_lazy_ood_eq_selected(
        config,
        log_n,
        n_1,
        l1_ood_count,
        current_len,
        direct_fold4_mode,
        cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )),
        std::env::var_os("FLOCK_NO_LIG_LAZY_OOD_EQ").is_some(),
    )
}

#[inline]
fn ranked_l1_lazy_ood_eq_selected(
    config: &ProverConfig,
    log_n: usize,
    n_1: usize,
    l1_ood_count: usize,
    current_len: usize,
    direct_fold4_mode: bool,
    platform_supported: bool,
    disabled: bool,
) -> bool {
    platform_supported
        && !disabled
        && direct_fold4_mode
        && log_n == 25
        && n_1 == 19
        && current_len == (1usize << 19)
        && l1_ood_count == 1
        && config.initial_log_msg_cols == 19
        && config.initial_log_num_interleaved == 6
        && config.initial_k == 6
        && config.log_inv_rates.first() == Some(&1)
}

/// Exact ranked-production selector for retaining a deep-level (L2..L5) OOD
/// equality as a tensor product — the same mechanism as the ranked L1
/// selector above, at the smaller recursion-ladder dimensions 16/13/10/7.
/// Every other profile, dimension, sample count, direct-fold mode, platform,
/// and either kill switch (the L1 master `FLOCK_NO_LIG_LAZY_OOD_EQ` or the
/// deep-level `FLOCK_NO_LIG_LAZY_OOD_EQ_DEEP`) keep the dense-table path.
#[inline]
fn ranked_deep_lazy_ood_eq_enabled(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    level_ood_count: usize,
    current_len: usize,
    direct_fold4_mode: bool,
) -> bool {
    ranked_deep_lazy_ood_eq_selected(
        config,
        log_n,
        n_level,
        level_ood_count,
        current_len,
        direct_fold4_mode,
        cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )),
        std::env::var_os("FLOCK_NO_LIG_LAZY_OOD_EQ").is_some()
            || std::env::var_os("FLOCK_NO_LIG_LAZY_OOD_EQ_DEEP").is_some(),
    )
}

#[inline]
fn ranked_deep_lazy_ood_eq_selected(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    level_ood_count: usize,
    current_len: usize,
    direct_fold4_mode: bool,
    platform_supported: bool,
    disabled: bool,
) -> bool {
    platform_supported
        && !disabled
        && direct_fold4_mode
        && log_n == 25
        // The ranked ladder folds 3 variables per level below n_1 = 19; the
        // L1 dimension itself belongs to the L1 selector above.
        && matches!(n_level, 16 | 13 | 10 | 7)
        && current_len == (1usize << n_level)
        && level_ood_count == 1
        && config.initial_log_msg_cols == 19
        && config.initial_log_num_interleaved == 6
        && config.initial_k == 6
        && config.log_inv_rates.first() == Some(&1)
}

/// Same-binary rollback for deferring an ordinary induced-basis glue into
/// the next fold that already consumes a factorized OOD equality.
const ENV_NO_LIG_DEFER_INDUCED_GLUE: &str = "FLOCK_NO_LIG_DEFER_INDUCED_GLUE";

/// Exact official M32 Fast selector. The selected levels are precisely the
/// five induced bases that have a following fold (L0 through L4); the final
/// L5 opening has no induced basis. Closing over the full ladder prevents a
/// neighboring profile from acquiring the deferred state accidentally.
#[allow(clippy::too_many_arguments)]
#[inline]
fn ranked_deferred_induced_glue_selected(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    current_len: usize,
    n_queries: usize,
    log_inv_rate: usize,
    lazy_ood: bool,
    direct_fold8_mode: bool,
    platform_supported: bool,
    disabled: bool,
) -> bool {
    platform_supported
        && !disabled
        && lazy_ood
        && direct_fold8_mode
        && log_n == 25
        && current_len == (1usize << n_level)
        && matches!(
            (n_level, n_queries, log_inv_rate),
            (19, 218, 1) | (16, 106, 2) | (13, 71, 3) | (10, 53, 4) | (7, 43, 5)
        )
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

#[allow(clippy::too_many_arguments)]
#[inline]
fn ranked_deferred_induced_glue_enabled(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    current_len: usize,
    n_queries: usize,
    log_inv_rate: usize,
    lazy_ood: bool,
    direct_fold8_mode: bool,
) -> bool {
    ranked_deferred_induced_glue_selected(
        config,
        log_n,
        n_level,
        current_len,
        n_queries,
        log_inv_rate,
        lazy_ood,
        direct_fold8_mode,
        cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )),
        std::env::var_os(ENV_NO_LIG_DEFER_INDUCED_GLUE).is_some(),
    )
}

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

fn fold_direct_fold8_factors_and_message(
    claims: &mut [super::ring_switch::DirectFold8Factors],
    challenge: F128,
) -> SumcheckMessage {
    // Sequential on the calling thread, for every claim count. The ranked
    // shape binds exactly two claims and this runs five times per proof, on
    // the Fiat-Shamir spine. The incumbent `rayon::join` split a per-round
    // bind whose two halves are smaller than a scope push plus the wake of a
    // parked worker, and the spine waits for both ends regardless, so the
    // split bought latency it could not hide. Value-identical to the join
    // arm: the claims are disjoint, F128 addition is XOR, and this is the
    // accumulation order the join arm already used.
    let (u0, u2) = claims
        .iter_mut()
        .fold((F128::ZERO, F128::ZERO), |mut total, claim| {
            let part = fold_one_direct_fold8_claim_and_message(claim, challenge);
            total.0 += part.0;
            total.1 += part.1;
            total
        });
    SumcheckMessage { u_0: u0, u_2: u2 }
}

fn direct_fold8_final_generators(
    claim: &super::ring_switch::DirectFold8Factors,
    challenge: F128,
) -> Vec<F128> {
    let mut generators = vec![F128::ZERO; claim.w_state.len() / 2];
    crate::field::f128_slice::fold_pairs(&claim.w_state, 0, &mut generators, challenge);
    generators
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
fn direct_fold8_b_gfni_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_FOLD8_B_GFNI").is_none());
    *ON
}

/// Commit parameters for the ranked DirectFold8 -> L1 overlap. This carries
/// no challenger: the early arm can only compute (not observe) the future
/// root.
#[derive(Clone, Copy)]
#[allow(dead_code)] // Fields are consumed only by the ranked AVX-512 route.
struct DirectFold8L1Precommit {
    log_msg_cols: usize,
    log_num_interleaved: usize,
    log_inv_rate: usize,
    kind: HashKind,
}

/// Same-binary rollback for the DirectFold8 producer/L1-commit overlap.
/// The optimization is additionally gated on the exact two-claim GFNI route
/// inside materialize_direct_fold8; every other geometry keeps the incumbent
/// single fused f+b pass.
#[inline]
#[allow(dead_code)] // Selector is consumed only by the ranked AVX-512 route.
fn direct_fold8_l1_precommit_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_L1_PRECOMMIT").is_none());
    *ON
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn materialize_direct_fold8_f_for_precommit(
    packed_witness: &[F128],
    folded_f: &mut [F128],
    block_len: usize,
    fold16_weight: &[F128; 16],
    r4: F128,
    r5: F128,
) {
    use rayon::prelude::*;

    const SUB: usize = 256;
    const ALIGN64_MIN_F128: usize = (32 * 1024) / core::mem::size_of::<F128>();
    folded_f
        .par_chunks_mut(block_len)
        .enumerate()
        .for_each_init(
            || {
                crate::scratch::LocalBuf::new(
                    (4 * SUB).max(ALIGN64_MIN_F128),
                    crate::scratch::fold_buf_pool_enabled(),
                )
            },
            |mid4, (block, f_out)| {
                let start = 64 * block * block_len;
                let f_in = &packed_witness[start..start + 64 * block_len];
                let mut slot = 0usize;
                while slot < block_len {
                    let n = SUB.min(block_len - slot);
                    let m4 = &mut mid4[..4 * n];
                    crate::field::f128_slice::fold16_banked(
                        &f_in[64 * slot..64 * (slot + n)],
                        m4,
                        fold16_weight,
                    );
                    crate::field::f128_slice::fold4_nested(m4, &mut f_out[slot..slot + n], r4, r5);
                    slot += n;
                }
            },
        );
}

/// Materialize the exact ranked two-claim GFNI basis and M6 after folded_f is
/// complete. This is deliberately separate from the incumbent fused loop so
/// the precommit arm can lend folded_f immutably to both Rayon branches while
/// the kill switch and every non-ranked shape retain the old mutable loop.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn materialize_direct_fold8_b_gfni_for_precommit(
    folded_f: &[F128],
    mut folded_b: Vec<F128>,
    claims: &[super::ring_switch::DirectFold8Factors],
    challenges: [F128; 6],
    block_len: usize,
) -> (Vec<F128>, SumcheckMessage) {
    use crate::zerocheck::multilinear::kernels::x86_64::{
        build_row_fold_mats_from_cols, gfni_fold64_four_maps_staged,
    };
    use rayon::prelude::*;

    assert_eq!(claims.len(), 2);
    assert!(block_len.is_multiple_of(64));
    assert_eq!(folded_b.len(), folded_f.len());
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && claim.eq_hi.len() * block_len == folded_f.len()
    }));

    let direct_tables: Vec<Vec<F128>> = claims
        .par_iter()
        .map(|claim| {
            let generators = direct_fold8_final_generators(claim, challenges[5]);
            super::ring_switch::build_direct_fold8_table_from_generators(&generators)
        })
        .collect();
    let direct_gfni_rows: Vec<(Vec<u64>, Vec<u64>)> = claims
        .par_iter()
        .map(|claim| {
            (
                claim.eq_lo.iter().map(|x| x.lo).collect(),
                claim.eq_lo.iter().map(|x| x.hi).collect(),
            )
        })
        .collect();

    const ALIGN64_MIN_F128: usize = (32 * 1024) / core::mem::size_of::<F128>();
    let stats = folded_b
        .par_chunks_mut(block_len)
        .zip(folded_f.par_chunks(block_len))
        .enumerate()
        .map_init(
            || {
                crate::scratch::LocalBuf::new(
                    64usize.max(ALIGN64_MIN_F128),
                    crate::scratch::fold_buf_pool_enabled(),
                )
            },
            |gfni_tmp, (block, (b_out, f_out))| {
                let (claim0, claim1) = (&claims[0], &claims[1]);
                let cols0 =
                    super::ring_switch::compose_block_cols(&direct_tables[0], claim0.eq_hi[block]);
                let mats0_lo = build_row_fold_mats_from_cols(&cols0[..64]);
                let mats0_hi = build_row_fold_mats_from_cols(&cols0[64..]);
                let cols1 =
                    super::ring_switch::compose_block_cols(&direct_tables[1], claim1.eq_hi[block]);
                let mats1_lo = build_row_fold_mats_from_cols(&cols1[..64]);
                let mats1_hi = build_row_fold_mats_from_cols(&cols1[64..]);
                let (rows0, rows1) = (&direct_gfni_rows[0], &direct_gfni_rows[1]);
                for slot in (0..block_len).step_by(64) {
                    // SAFETY: both packed row halves supply 512 bytes, both
                    // outputs cover 64 F128s, and this helper's cfg fixes all
                    // target features required by the kernel.
                    unsafe {
                        gfni_fold64_four_maps_staged(
                            rows0.0.as_ptr().add(slot).cast::<u8>(),
                            &mats0_lo,
                            rows0.1.as_ptr().add(slot).cast::<u8>(),
                            &mats0_hi,
                            rows1.0.as_ptr().add(slot).cast::<u8>(),
                            &mats1_lo,
                            rows1.1.as_ptr().add(slot).cast::<u8>(),
                            &mats1_hi,
                            b_out.as_mut_ptr().add(slot),
                            gfni_tmp.as_mut_ptr().cast(),
                        );
                    }
                }
                // SAFETY: this helper's cfg guarantees AVX-512F+VPCLMUL;
                // slices have the same even power-of-two block length.
                unsafe { msg_reduce_avx512(f_out, b_out) }
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );

    (
        folded_b,
        SumcheckMessage {
            u_0: stats.0,
            u_2: stats.1,
        },
    )
}
/// Sixty-four-bank materializer. Six challenges are sampled from direct
/// product statistics before this function binds the witness and combined
/// basis in one N→N/64 pass. It emits M6 — the round message of the folded
/// 2^19 state — fused into the same pass; no lookahead follows because the
/// initial cadence is exhausted (the fold2 pair of the fold4 route never
/// runs and the 2^21/2^20 states never exist).
fn materialize_direct_fold8(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold8Factors],
    challenges: [F128; 6],
    l1_precommit: Option<DirectFold8L1Precommit>,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage, Option<LigeroWitness>) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    let has_ordinary = !ordinary_basis.is_empty();
    assert!(!has_ordinary || ordinary_basis.len() == packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(64));
    let [r0, r1, r2, r3, r4, r5] = challenges;
    let fold16_weight: [F128; 16] = std::array::from_fn(|bank| {
        let mut weight = F128::ONE;
        for (bit, &challenge) in challenges[..4].iter().enumerate() {
            weight *= if (bank >> bit) & 1 == 0 {
                F128::ONE + challenge
            } else {
                challenge
            };
        }
        weight
    });
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let b_gfni_candidate = direct_fold8_b_gfni_enabled() && claims.len() == 2 && !has_ordinary;
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let b_gfni_candidate = false;

    let out_len = packed_witness.len() / 64;
    let block_len = claims[0].eq_lo.len();
    assert!(block_len.is_multiple_of(4));
    // The GFNI kernel owns complete 64-slot batches. Keep unusual DirectFold8
    // geometries on the scalar path instead of reading through a short tail.
    let b_gfni_on = b_gfni_candidate && block_len.is_multiple_of(64);
    assert_eq!(out_len, block_len * claims[0].eq_hi.len());
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && claim.eq_hi.len() * block_len == out_len
    }));

    // Ranked-only producer/consumer overlap. First finish f8, then let the
    // L1 encoder+Merkle commit consume it while the disjoint branch builds
    // the two-claim GFNI basis and M6. Both branches only read folded_f. The
    // returned witness owns its codeword/tree, and its root remains entirely
    // outside the challenger until the original post-M6 observation point.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    if b_gfni_on && direct_fold8_l1_precommit_enabled() {
        if let Some(precommit) = l1_precommit {
            assert_eq!(
                out_len,
                (1usize << precommit.log_msg_cols) * (1usize << precommit.log_num_interleaved)
            );
            let mut folded_f = crate::scratch::take_f128(out_len);
            materialize_direct_fold8_f_for_precommit(
                &packed_witness,
                &mut folded_f,
                block_len,
                &fold16_weight,
                r4,
                r5,
            );
            // Preserve incumbent buffer custody: acquire both folded outputs
            // before returning the much larger input to the shared pool.
            let folded_b = crate::scratch::take_f128(out_len);
            crate::scratch::give_f128(packed_witness);
            crate::scratch::give_f128(ordinary_basis);

            let (precommitted, (folded_b, msg)) = rayon::join(
                || {
                    let ntt =
                        AdditiveNttF128::standard(precommit.log_msg_cols + precommit.log_inv_rate);
                    ligero_commit(
                        &folded_f,
                        precommit.log_msg_cols,
                        precommit.log_num_interleaved,
                        precommit.log_inv_rate,
                        &ntt,
                        precommit.kind,
                    )
                },
                || {
                    materialize_direct_fold8_b_gfni_for_precommit(
                        &folded_f, folded_b, claims, challenges, block_len,
                    )
                },
            );
            return (folded_f, folded_b, msg, Some(precommitted));
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let _ = l1_precommit;

    let direct_tables: Vec<Vec<F128>> = claims
        .par_iter()
        .map(|claim| {
            let generators = direct_fold8_final_generators(claim, challenges[5]);
            super::ring_switch::build_direct_fold8_table_from_generators(&generators)
        })
        .collect();

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let direct_gfni_rows: Vec<(Vec<u64>, Vec<u64>)> = if b_gfni_on {
        claims
            .par_iter()
            .map(|claim| {
                (
                    claim.eq_lo.iter().map(|x| x.lo).collect(),
                    claim.eq_lo.iter().map(|x| x.hi).collect(),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    const SUB: usize = 256;
    /// `RecycleAlloc` (`flock-prover/src/recycle_alloc.rs`) only hands back
    /// 64-byte-aligned pointers for allocations of at least `RECYCLE_MIN`
    /// = 32 KiB; anything smaller goes straight to the system allocator and
    /// lands 16 mod 64 (that file's own doc records the 16-mod-64 history and
    /// the three costs it pays). The per-job `mid4` staging buffer below is
    /// `4 * SUB` F128 = 16 KiB, i.e. it sits UNDER the gate, so every wide
    /// load and store the 64->4->1 fold makes against it is a cache-line
    /// split. Ask for the gate's own minimum instead so the buffer lands in
    /// the aligned class. CAPACITY ONLY: the kernels below still address
    /// exactly `4 * n` elements with `n <= SUB`, so no value, no element and
    /// no order changes.
    const ALIGN64_MIN_F128: usize = (32 * 1024) / core::mem::size_of::<F128>();
    let stats = folded_b
        .par_chunks_mut(block_len)
        .zip(folded_f.par_chunks_mut(block_len))
        .enumerate()
        .map_init(
            // Per-job working buffers, recycled through this thread's free
            // list instead of being allocated, zeroed and freed once per job:
            // rayon calls this init once per leaf job, i.e. inside the fold
            // loop, and every one of the four is written before it is read
            // (`compose_block_table` documents it for `scratch`; the fold
            // kernels write every output slot; the GFNI kernel stores all
            // sixteen planes before loading any). `FLOCK_NO_PCS_FOLD_BUF_POOL=1`
            // restores the allocating form.
            || {
                let pooled = crate::scratch::fold_buf_pool_enabled();
                (
                    crate::scratch::LocalBuf::new(
                        if b_gfni_on {
                            0
                        } else {
                            super::ring_switch::FOLD_TABLE_TOTAL
                        },
                        pooled,
                    ),
                    crate::scratch::LocalBuf::new(if has_ordinary { 16 * SUB } else { 0 }, pooled),
                    crate::scratch::LocalBuf::new((4 * SUB).max(ALIGN64_MIN_F128), pooled),
                    // `gfni_tmp` sits under the same sub-`RECYCLE_MIN` gate as `mid4`.
                    // The kernel touches only the first 64 elements; the larger capacity
                    // selects the pool allocator’s 64-byte-aligned class.
                    crate::scratch::LocalBuf::new(
                        if b_gfni_on {
                            (64usize).max(ALIGN64_MIN_F128)
                        } else {
                            0
                        },
                        pooled,
                    ),
                )
            },
            |(scratch, mid16, mid4, gfni_tmp), (block, (b_out, f_out))| {
                let _ = &gfni_tmp;
                let start = 64 * block * block_len;
                let f_in = &packed_witness[start..start + 64 * block_len];
                let b_in: &[F128] = if has_ordinary {
                    &ordinary_basis[start..start + 64 * block_len]
                } else {
                    &[]
                };
                // Deferred-reduction 16-bank fold followed by one nested
                // fold4: 64 -> 4 -> 1. On SPR this halves the VPCLMUL issue
                // count versus three nested passes while keeping bounded
                // scratch and eliminating the scalar 64-product chain.
                let mut slot = 0usize;
                while slot < block_len {
                    let n = SUB.min(block_len - slot);
                    let m4 = &mut mid4[..4 * n];
                    crate::field::f128_slice::fold16_banked(
                        &f_in[64 * slot..64 * (slot + n)],
                        m4,
                        &fold16_weight,
                    );
                    crate::field::f128_slice::fold4_nested(m4, &mut f_out[slot..slot + n], r4, r5);
                    if has_ordinary {
                        let m16 = &mut mid16[..16 * n];
                        crate::field::f128_slice::fold4_nested(
                            &b_in[64 * slot..64 * (slot + n)],
                            m16,
                            r0,
                            r1,
                        );
                        crate::field::f128_slice::fold4_nested(m16, m4, r2, r3);
                        crate::field::f128_slice::fold4_nested(
                            m4,
                            &mut b_out[slot..slot + n],
                            r4,
                            r5,
                        );
                    }
                    slot += n;
                }

                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512vbmi",
                    target_feature = "vpclmulqdq",
                    target_feature = "gfni"
                ))]
                if b_gfni_on {
                    use crate::zerocheck::multilinear::kernels::x86_64::{
                        build_row_fold_mats_from_cols, gfni_fold64_four_maps_staged,
                    };
                    let (claim0, claim1) = (&claims[0], &claims[1]);
                    let cols0 = super::ring_switch::compose_block_cols(
                        &direct_tables[0],
                        claim0.eq_hi[block],
                    );
                    let mats0_lo = build_row_fold_mats_from_cols(&cols0[..64]);
                    let mats0_hi = build_row_fold_mats_from_cols(&cols0[64..]);
                    let cols1 = super::ring_switch::compose_block_cols(
                        &direct_tables[1],
                        claim1.eq_hi[block],
                    );
                    let mats1_lo = build_row_fold_mats_from_cols(&cols1[..64]);
                    let mats1_hi = build_row_fold_mats_from_cols(&cols1[64..]);
                    let (rows0, rows1) = (&direct_gfni_rows[0], &direct_gfni_rows[1]);
                    for slot in (0..block_len).step_by(64) {
                        // SAFETY: each packed-u64 row half supplies 512 bytes;
                        // both output buffers cover 64 F128s; cfg features hold.
                        unsafe {
                            gfni_fold64_four_maps_staged(
                                rows0.0.as_ptr().add(slot).cast::<u8>(),
                                &mats0_lo,
                                rows0.1.as_ptr().add(slot).cast::<u8>(),
                                &mats0_hi,
                                rows1.0.as_ptr().add(slot).cast::<u8>(),
                                &mats1_lo,
                                rows1.1.as_ptr().add(slot).cast::<u8>(),
                                &mats1_hi,
                                b_out.as_mut_ptr().add(slot),
                                gfni_tmp.as_mut_ptr().cast(),
                            );
                        }
                    }
                }
                if !b_gfni_on {
                    let (first_claim, rest_claims) = claims.split_first().unwrap();
                    let (first_table, _) = direct_tables.split_first().unwrap();
                    super::ring_switch::compose_block_table(
                        first_table,
                        first_claim.eq_hi[block],
                        scratch,
                    );
                    for slot in 0..block_len {
                        let direct =
                            super::ring_switch::fold_one_slot(first_claim.eq_lo[slot], scratch);
                        b_out[slot] = if has_ordinary {
                            direct + b_out[slot]
                        } else {
                            direct
                        };
                    }
                    for (claim, table) in rest_claims.iter().zip(direct_tables.iter().skip(1)) {
                        super::ring_switch::compose_block_table(table, claim.eq_hi[block], scratch);
                        for (slot, out) in b_out.iter_mut().enumerate() {
                            *out += super::ring_switch::fold_one_slot(claim.eq_lo[slot], scratch);
                        }
                    }
                }
                #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
                {
                    // SAFETY: features cfg-guaranteed; slices are equal and
                    // block_len is an even power of two.
                    unsafe { msg_reduce_avx512(f_out, b_out) }
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
                {
                    super::round0_scalar(f_out, b_out)
                }
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );

    crate::scratch::give_f128(packed_witness);
    crate::scratch::give_f128(ordinary_basis);
    (
        folded_f,
        folded_b,
        SumcheckMessage {
            u_0: stats.0,
            u_2: stats.1,
        },
        None,
    )
}

pub struct SumcheckProver {
    f: FoldBuf,
    /// Single combined basis poly. After every `glue(β)`, the introduced
    /// `b_new` is folded into here as `combined_basis += β · b_new`. This
    /// keeps fold cost O(1 + 1) = (f + combined_basis) regardless of how
    /// many recursive intro/glue pairs have happened.
    combined_basis: FoldBuf,
    /// Per-open fold-output arena (see [`FoldArena`]). `None` keeps the
    /// previous per-round allocation behavior. Declared after `f` /
    /// `combined_basis` purely for clarity; `FoldBuf::Arena` drops are no-ops
    /// so field drop order is irrelevant for safety.
    fold_arena: Option<FoldArena>,
    t_r: F128,
    transcript: Vec<SumcheckMessage>,
    pending_glue: Option<(Vec<F128>, F128)>,
    /// One ordinary induced basis whose challenge has already updated `t_r`.
    /// At the exact ranked shape its pointwise glue is deferred into the fold
    /// that also consumes `pending_ood_eq`.
    pending_fold_basis: Option<(Vec<F128>, F128)>,
    /// The ranked L1 OOD equality remains as 2^11 × 2^7 factors until the
    /// next fold consumes its glued correction.
    pending_ood_eq: Option<PendingOodEq>,
}

impl SumcheckProver {
    pub fn new(f: Vec<F128>, b1: Vec<F128>, h1: F128) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let mut inst = Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(b1),
            fold_arena: None,
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

    /// Hand the prover a per-open [`FoldArena`]; every subsequent parallel
    /// [`Self::fold`] carves its output buffers from it until exhausted.
    pub fn set_fold_arena(&mut self, arena: FoldArena) {
        self.fold_arena = Some(arena);
    }

    /// Diagnostics only: whether the current `f` buffer is an arena carve.
    pub(crate) fn f_is_arena(&self) -> bool {
        matches!(self.f, FoldBuf::Arena { .. })
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
        let mut inst = Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(b1),
            fold_arena: None,
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        };
        inst.transcript.push(first_msg);
        (inst, first_msg)
    }

    fn new_after_direct_fold4(
        f: Vec<F128>,
        basis: Vec<F128>,
        target: F128,
        transcript: [SumcheckMessage; 5],
        fold_arena: Option<FoldArena>,
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(basis),
            fold_arena,
            t_r: target,
            transcript: transcript.to_vec(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        }
    }

    fn new_after_direct_fold2(
        f: Vec<F128>,
        basis: Vec<F128>,
        target: F128,
        transcript: [SumcheckMessage; 3],
        fold_arena: Option<FoldArena>,
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(basis),
            fold_arena,
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
        fold_arena: Option<FoldArena>,
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(basis),
            fold_arena,
            t_r: target,
            transcript: transcript.to_vec(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        }
    }

    pub fn fold(&mut self, r: F128) -> SumcheckMessage {
        // Fused: fold f and combined_basis at r AND build the next-round
        // message in one parallel pass (was three passes). See
        // [`fold_and_msg_lsb`].
        assert!(self.pending_glue.is_none(), "fold before ordinary glue");
        let pending_ood = self.pending_ood_eq.take();
        let pending_basis = self.pending_fold_basis.take();
        let (nf, nb, msg) = match (pending_ood, pending_basis) {
            (
                Some(PendingOodEq::Glued {
                    eq_lo,
                    eq_hi,
                    z_0,
                    beta,
                }),
                deferred_basis,
            ) => {
                // Folding eq([z0, z_tail], ·) at LSB challenge r leaves
                // (1 + z0 + r) * eq(z_tail, ·) in characteristic two.
                let gamma = beta * (F128::ONE + z_0 + r);
                fold_and_msg_lsb_inner(
                    &self.f,
                    &self.combined_basis,
                    r,
                    self.fold_arena.as_mut(),
                    Some((&eq_lo, &eq_hi, gamma)),
                    deferred_basis
                        .as_ref()
                        .map(|(basis, alpha)| (basis.as_slice(), *alpha)),
                )
            }
            (Some(PendingOodEq::Introduced { .. }), _) => {
                panic!("fold before factorized OOD glue")
            }
            (None, Some(_)) => {
                panic!("deferred ordinary glue without a consuming factorized OOD fold")
            }
            (None, None) => {
                fold_and_msg_lsb(&self.f, &self.combined_basis, r, self.fold_arena.as_mut())
            }
        };
        // On x86_64, recycle the just-consumed OWNED buffers into the scratch
        // pool (same ownership as the Drop impl) so the next round's
        // `fold_and_msg_lsb` takes resident pages. aarch64 measured slower with
        // this pooling, so there we just move the new buffers in and drop the
        // old ones (a no-op for arena regions — the arena outlives the open).
        #[cfg(target_arch = "x86_64")]
        {
            if let FoldBuf::Owned(v) = std::mem::replace(&mut self.f, nf) {
                crate::scratch::give_f128(v);
            }
            if let FoldBuf::Owned(v) = std::mem::replace(&mut self.combined_basis, nb) {
                crate::scratch::give_f128(v);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.f = nf;
            self.combined_basis = nb;
        }
        self.transcript.push(msg);
        msg
    }

    /// Introduce a fresh basis poly with claimed sum `h_new`. Sends the
    /// (u_0, u_2) for `Σ_x f(x) · b_new(x)` at the current dim.
    pub fn introduce_new(&mut self, b_new: Vec<F128>, h_new: F128) -> SumcheckMessage {
        assert!(
            self.pending_fold_basis.is_none(),
            "ordinary introduction across deferred basis"
        );
        assert_eq!(b_new.len(), self.f.len());
        let msg = round_msg_lsb(&self.f, &b_new);
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
        assert!(
            self.pending_fold_basis.is_none(),
            "OOD introduction across deferred basis"
        );
        assert_eq!(b_new.len(), self.f.len());
        let (msg, h_new) = round_msg_and_eval_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        (msg, h_new)
    }

    /// Introduce `eq(z, ·)` without building its dense 2^19 table. The LSB
    /// coordinate is kept separately and the remaining equality is split into
    /// cache-resident low/high tensor factors.
    fn introduce_new_ood_factorized(&mut self, z: &[F128]) -> Option<(SumcheckMessage, F128)> {
        let expected_len = 1usize.checked_shl(z.len().try_into().ok()?);
        if z.is_empty()
            || expected_len != Some(self.f.len())
            || self.pending_glue.is_some()
            || self.pending_fold_basis.is_some()
            || self.pending_ood_eq.is_some()
        {
            return None;
        }
        let tail = &z[1..];
        let split = tail.len().min(LAZY_OOD_EQ_SPLIT_LOW_LOG);
        let eq_lo = build_eq_table(&tail[..split]);
        let eq_hi = build_eq_table(&tail[split..]);
        let z_0 = z[0];
        let (msg, h_new) = round_msg_and_eval_lsb_factorized_eq(&self.f, &eq_lo, &eq_hi, z_0);
        self.transcript.push(msg);
        self.pending_ood_eq = Some(PendingOodEq::Introduced {
            eq_lo,
            eq_hi,
            z_0,
            h_new,
        });
        Some((msg, h_new))
    }

    /// Apply the transcript separation challenge while retaining the equality
    /// factors for injection into the next fold.
    fn glue_factorized_ood(&mut self, beta: F128) {
        assert!(
            self.pending_glue.is_none(),
            "factorized OOD glue across ordinary pending glue"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "factorized OOD glue across deferred ordinary basis"
        );
        let pending = self
            .pending_ood_eq
            .take()
            .expect("factorized OOD glue without introduction");
        let PendingOodEq::Introduced {
            eq_lo,
            eq_hi,
            z_0,
            h_new,
        } = pending
        else {
            panic!("factorized OOD equality glued twice");
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
            "ordinary glue across deferred ordinary basis"
        );
        let (b_new, h_new) = self
            .pending_glue
            .take()
            .expect("glue without introduce_new");
        assert_eq!(b_new.len(), self.combined_basis.len());
        const PAR_THRESHOLD: usize = 4096;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        let x4 = open_ood_x4_enabled();
        #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
        let x4 = false;
        if x4 {
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            {
                const CHUNK: usize = 2048;
                let acc: &mut [F128] = &mut self.combined_basis;
                if acc.len() < PAR_THRESHOLD {
                    // SAFETY: features cfg-guaranteed; equal lengths.
                    unsafe { glue_block_x4(acc, &b_new, alpha) };
                } else {
                    acc.par_chunks_mut(CHUNK)
                        .zip(b_new.par_chunks(CHUNK))
                        .for_each(|(a, v)| {
                            // SAFETY: equal chunk lengths; features cfg-guaranteed.
                            unsafe { glue_block_x4(a, v, alpha) }
                        });
                }
            }
        } else if self.combined_basis.len() < PAR_THRESHOLD {
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

    /// Apply the ordinary separation challenge at its transcript position,
    /// but retain the basis for the immediately following factorized-OOD
    /// fold. Linearity makes the later update exact:
    /// `fold_r(B + alpha*D) = fold_r(B) + alpha*fold_r(D)`.
    fn glue_deferred_into_factorized_ood_fold(&mut self, alpha: F128) {
        assert!(
            matches!(self.pending_ood_eq, Some(PendingOodEq::Glued { .. })),
            "deferred ordinary glue requires a glued factorized OOD term"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "more than one ordinary basis deferred"
        );
        let (basis, h_new) = self
            .pending_glue
            .take()
            .expect("deferred glue without ordinary introduction");
        assert_eq!(basis.len(), self.combined_basis.len());
        self.t_r += alpha * h_new;
        self.pending_fold_basis = Some((basis, alpha));
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
/// Ranked default parallelizes the query-phase gathers — the opened-row
/// copies and the Merkle multi-proof sibling reads. Both are random-access
/// walks over DRAM-cold structures (the 1 GiB L0 codeword and 64 MiB L0 tree
/// were written a full commit + zerocheck + lincheck earlier; each deeper
/// level's tree is bigger than L2) that run alone on the Fiat–Shamir chain
/// between the query sample and the induce — serial, latency-bound, with 15
/// of 16 ranked cores idle. Width cannot change wire bytes: both gathers are
/// pure reads producing an output whose order is fixed by index arithmetic
/// alone, and the indexed parallel collects preserve that order exactly.
/// Read once per process. `FLOCK_NO_SERIAL_PAR=1` restores the exact
/// incumbent sequential gathers (shared switch — see
/// [`crate::serial_par_enabled`]).
fn serial_par_enabled() -> bool {
    crate::serial_par_enabled()
}

/// Work floor (total F128 copied) below which the opened-row gather stays on
/// the sequential path: tiny gathers (deep levels of test geometries) are
/// below rayon dispatch break-even.
const ROW_GATHER_PAR_MIN_ELEMS: usize = 512;

/// Copy the queried codeword rows, in query order. `par` selects an
/// order-preserving indexed parallel map (bit-identical to the sequential
/// map: each row copy is a pure function of its own query index); `false` is
/// the incumbent sequential gather, kept verbatim as the kill-switch path
/// and the byte-identity oracle.
fn gather_opened_rows<'a, F>(queries: &[usize], row: F, par: bool) -> Vec<Vec<F128>>
where
    F: Fn(usize) -> &'a [F128] + Sync,
{
    use rayon::prelude::*;
    let row_len = queries.first().map_or(0, |&q| row(q).len());
    if par && queries.len() * row_len >= ROW_GATHER_PAR_MIN_ELEMS {
        queries.par_iter().map(|&q| row(q).to_vec()).collect()
    } else {
        queries.iter().map(|&q| row(q).to_vec()).collect()
    }
}

/// Sibling-count floor below which the multi-proof gather stays on the
/// incumbent single-pass walk (index walk + parallel gather costs a rayon
/// dispatch; tiny test trees don't earn it).
const MULTIPROOF_PAR_MIN_SIBLINGS: usize = 512;

/// One rayon task per this many sibling gathers: each gather is a single
/// (usually cold) 32-byte tree read, so a task is ~64 line fills — big
/// enough to amortize dispatch, small enough to spread the misses wide.
const MULTIPROOF_PAR_CHUNK: usize = 64;

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
    multi_proof_gather(tree, block_len, queries, serial_par_enabled())
}

/// Multi-proof body. `par` splits the incumbent walk into its two halves —
/// a pure index walk (integer arithmetic, no tree reads) followed by an
/// order-preserving parallel gather of the sibling hashes — which emits the
/// exact bytes of the fused sequential walk (see
/// [`merkle::merkle_multi_proof_sibling_indices`]); `false` is the incumbent
/// single-pass walk, kept verbatim as the kill-switch path and the
/// byte-identity oracle.
fn multi_proof_gather(tree: &[Hash], block_len: usize, queries: &[usize], par: bool) -> Vec<Hash> {
    use rayon::prelude::*;
    if par {
        let indices = merkle::merkle_multi_proof_sibling_indices(block_len, queries);
        if indices.len() >= MULTIPROOF_PAR_MIN_SIBLINGS {
            return indices
                .par_iter()
                .with_min_len(MULTIPROOF_PAR_CHUNK)
                .map(|&i| tree[i])
                .collect();
        }
    }
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
    fold_arena: Option<FoldArena>,
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
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        fold_arena,
        challenger,
    )
}

/// Production AB-only direct-fold2 entry. `ordinary_basis` contains C while
/// `direct` contains AB's four-bank sufficient statistic.
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
    fold_arena: Option<FoldArena>,
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
        fold_arena,
        challenger,
    )
}

/// Sixteen-bank direct entry: the first four transcript messages come
/// entirely from the claims' 16×16 product matrices; after four sequential
/// FS samples the state is materialized once at N/16 and rejoins the
/// ordinary fused fold cadence.
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
    fold_arena: Option<FoldArena>,
    challenger: &mut Ch,
) -> LigeritoProof {
    assert!(
        config.initial_k >= 4,
        "direct-fold4 scaffold requires initial_k >= 4"
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
        fold_arena,
        challenger,
    )
}
/// Direct-fold8 entry. The first six transcript messages come entirely from
/// `direct` 64×64 product matrices; after six sequential FS samples the state
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
    fold_arena: Option<FoldArena>,
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
        fold_arena,
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
    fold_arena: Option<FoldArena>,
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
    if direct_fold2.is_some() || direct_fold4.is_some() || direct_fold8.is_some() {
        assert!(b_initial.is_empty() || b_initial.len() == 1usize << log_n);
    } else {
        assert_eq!(b_initial.len(), 1usize << log_n);
    }
    if direct_fold8.is_some() {
        assert_eq!(initial_k, 6, "direct-fold8 scaffold requires initial_k=6");
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

    let trace = std::env::var("LIG_PROVE_TRACE").is_ok() || open_timing();
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
    assert!(
        direct_fold2.is_none() || (direct_fold4.is_none() && direct_fold8.is_none()),
        "direct-fold2 and direct-fold4/fold8 modes are mutually exclusive"
    );
    assert!(direct_fold4.is_none() || direct_fold8.is_none());
    let direct_fold4_mode = direct_fold4.is_some();
    let direct_fold8_mode = direct_fold8.is_some();
    let direct_mode = direct_fold2.is_some() || direct_fold4_mode || direct_fold8_mode;
    if direct_fold8_mode {
        assert_eq!(initial_k, 6, "direct fold8 needs six initial rounds");
    } else if direct_fold4_mode {
        assert!(initial_k >= 4, "direct fold4 needs four initial rounds");
    } else if direct_mode {
        assert!(initial_k >= 2, "direct AB fold2 needs two initial rounds");
    }
    let mut packed_witness = Some(packed_witness);
    let mut b_initial = Some(b_initial);
    let mut direct_fold2 = direct_fold2;
    let mut direct_fold4 = direct_fold4;
    let mut direct_fold8 = direct_fold8;
    // L1 shape is transcript-independent. Only the exact ranked DirectFold8
    // geometry is allowed to request an early commit; unsupported targets,
    // ordinary-basis mixtures, non-two-claim shapes, and either kill switch
    // all retain the incumbent post-M6 call below.
    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1);
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let l1_precommit_candidate = direct_fold8_mode
        && log_n == 25
        && initial_k == 6
        && log_msg_cols_1 == 16
        && log_num_interleaved_1 == 3
        && log_inv_rate_1 == 2
        && config.merkle_hash == HashKind::Blake3
        && direct_fold8_l1_precommit_enabled()
        && direct_fold8_b_gfni_enabled()
        && b_initial.as_ref().is_some_and(|basis| basis.is_empty())
        && direct_fold8
            .as_ref()
            .is_some_and(|claims| claims.len() == 2 && claims[0].eq_lo.len().is_multiple_of(64));
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let l1_precommit_candidate = false;
    let mut early_wtns_1: Option<LigeroWitness> = None;
    let mut fold4_challenges: Vec<F128> = Vec::with_capacity(6);
    let mut fold4_msgs: Vec<SumcheckMessage> = Vec::with_capacity(6);
    let mut fold_arena = fold_arena;
    let (mut sc_prover, start_msg) = if direct_mode {
        (
            None,
            first_msg.expect("direct mode requires a precomputed round-zero message"),
        )
    } else {
        let (mut prover, msg) = match first_msg {
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
        if let Some(arena) = fold_arena.take() {
            prover.set_fold_arena(arena);
        }
        (Some(prover), msg)
    };
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    let mut direct_r0 = None;
    let mut direct_msg1 = None;
    // FLOCK_OPEN_TIMING diagnostics: per-round (grind ms, fold ms, arena?)
    let mut round_diag: Vec<(f64, f64, bool)> = Vec::new();
    // (level, log_msg_cols, n_queries, ms, used_ntt)
    let mut induce_diag: Vec<(usize, usize, usize, f64, bool)> = Vec::new();
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
        let _tg = std::time::Instant::now();
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            fold_grinding_nonces.push(challenger.grind_pow(bits));
        }
        let grind_ms = _tg.elapsed().as_secs_f64() * 1e3;
        let r = challenger.sample_f128();
        let _tf = std::time::Instant::now();
        let msg = if direct_fold8_mode && j < 6 {
            let msg = if j < 5 {
                fold_direct_fold8_factors_and_message(
                    direct_fold8
                        .as_mut()
                        .expect("direct-fold8 factors remain live through round five"),
                    r,
                )
            } else {
                let direct = direct_fold8
                    .take()
                    .expect("direct-fold8 factors consumed once");
                let l1_precommit = l1_precommit_candidate.then_some(DirectFold8L1Precommit {
                    log_msg_cols: log_msg_cols_1,
                    log_num_interleaved: log_num_interleaved_1,
                    log_inv_rate: log_inv_rate_1,
                    kind: config.merkle_hash,
                });
                let (f8, b8, msg, precommitted) = materialize_direct_fold8(
                    packed_witness.take().unwrap(),
                    b_initial.take().unwrap(),
                    &direct,
                    [
                        fold4_challenges[0],
                        fold4_challenges[1],
                        fold4_challenges[2],
                        fold4_challenges[3],
                        fold4_challenges[4],
                        r,
                    ],
                    l1_precommit,
                );
                early_wtns_1 = precommitted;
                debug_assert_eq!(l1_precommit_candidate, early_wtns_1.is_some());
                sc_prover = Some(SumcheckProver::new_after_direct_fold8(
                    f8,
                    b8,
                    target,
                    [
                        start_msg,
                        fold4_msgs[0],
                        fold4_msgs[1],
                        fold4_msgs[2],
                        fold4_msgs[3],
                        fold4_msgs[4],
                        msg,
                    ],
                    fold_arena.take(),
                ));
                msg
            };
            fold4_challenges.push(r);
            fold4_msgs.push(msg);
            msg
        } else if direct_fold4_mode && j < 4 {
            let msg = match j {
                0 => eval_lookahead(
                    round1_lookahead
                        .as_ref()
                        .expect("direct-fold4 requires round-one lookahead"),
                    r,
                ),
                1 => eval_fold4_lookahead2(
                    round2_lookahead
                        .as_ref()
                        .expect("direct-fold4 requires round-two lookahead"),
                    fold4_challenges[0],
                    r,
                ),
                2 => eval_fold4_lookahead3(
                    round3_lookahead
                        .as_ref()
                        .expect("direct-fold4 requires round-three lookahead"),
                    fold4_challenges[0],
                    fold4_challenges[1],
                    r,
                ),
                _ => {
                    let direct = direct_fold4
                        .take()
                        .expect("direct-fold4 factors consumed once");
                    let (f4, b4, msg) = materialize_direct_fold4(
                        packed_witness.take().unwrap(),
                        b_initial.take().unwrap(),
                        &direct,
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
                        [start_msg, fold4_msgs[0], fold4_msgs[1], fold4_msgs[2], msg],
                        fold_arena.take(),
                    ));
                    msg
                }
            };
            fold4_challenges.push(r);
            fold4_msgs.push(msg);
            msg
        } else if direct_mode && j == 0 {
            let msg = eval_lookahead(
                round1_lookahead
                    .as_ref()
                    .expect("direct AB fold2 requires round-one lookahead"),
                r,
            );
            direct_r0 = Some(r);
            direct_msg1 = Some(msg);
            msg
        } else if direct_mode && j == 1 {
            let direct = direct_fold2
                .take()
                .expect("direct AB factors consumed once");
            let (f2, b2, msg) = materialize_direct_ab_fold2(
                packed_witness.take().unwrap(),
                b_initial.take().unwrap(),
                &direct,
                direct_r0.take().unwrap(),
                r,
            );
            sc_prover = Some(SumcheckProver::new_after_direct_fold2(
                f2,
                b2,
                target,
                [start_msg, direct_msg1.take().unwrap(), msg],
                fold_arena.take(),
            ));
            msg
        } else {
            sc_prover.as_mut().unwrap().fold(r)
        };
        if trace {
            round_diag.push((
                grind_ms,
                _tf.elapsed().as_secs_f64() * 1e3,
                sc_prover.as_ref().is_some_and(SumcheckProver::f_is_arena),
            ));
        }
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        r_lane_fold.push(r);
    }
    if trace {
        t_init_sumcheck += _t.elapsed();
    }
    let mut sc_prover = sc_prover.expect("direct state must materialize during the initial rounds");

    // Commit f^1 = folded packed witness as wtns_1.
    let wtns_1 = match early_wtns_1.take() {
        // Its wall time is already inside initial-round/M6 timing. Do not
        // double-count the overlapped branch in the mutually serial commit
        // diagnostic bucket.
        Some(witness) => witness,
        None => {
            let _t = std::time::Instant::now();
            let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
            let witness = ligero_commit(
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
            witness
        }
    };
    // Transcript order is unchanged: M6 was observed inside the initial loop;
    // only now is the already-computed root appended to Fiat-Shamir.
    challenger.observe_bytes(&wtns_1.root());

    let lazy_l1_ood = ranked_l1_lazy_ood_eq_enabled(
        config,
        log_n,
        n1,
        ood_count(1),
        sc_prover.f().len(),
        direct_fold4_mode || direct_fold8_mode,
    );

    // OOD binding for the L1 commit: each sample evaluates f1's multilinear
    // extension at a random transcript point z ∈ F^{n1}, sends the claimed
    // value, and folds the claim `Σ_x f1(x)·eq(z,x) = y` into the running
    // sumcheck (introduce + glue). Binds the prover to a single codeword of
    // the interleaved list before any of L0's queries are drawn.
    {
        let _t = std::time::Instant::now();
        for _ in 0..ood_count(1) {
            let z = challenger.sample_f128_vec(n1);
            // The exact ranked L1 path retains eq(z[1..], ·) as 2^11 × 2^7
            // factors. Every rollback and other geometry builds the incumbent
            // dense table.
            let (intro, y) = if lazy_l1_ood {
                sc_prover
                    .introduce_new_ood_factorized(&z)
                    .expect("ranked factorized OOD introduction shape changed")
            } else {
                let eq_z = build_eq_table_split(&z);
                sc_prover.introduce_new_with_eval(eq_z)
            };
            challenger.observe_f128(y);
            ood_values.push(y);
            challenger.observe_f128(intro.u_0);
            challenger.observe_f128(intro.u_2);
            let beta = challenger.sample_f128();
            if lazy_l1_ood {
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
    let opened_rows_0: Vec<Vec<F128>> =
        gather_opened_rows(&queries_0, l0_row, serial_par_enabled());
    let merkle_proof_0 = merkle_multi_proof_for(l0_tree, l0_block_len, &queries_0);
    if trace {
        t_opens += _t.elapsed();
    }
    // The proof's copy of the opened rows is the SAME data the induce below
    // reads. The incumbent cloned all 218 rows (218 allocations, 223 KiB) here,
    // on one core, purely to keep `opened_rows_0` alive for the induce; moving
    // the rows in after the induce deletes the copy outright.
    let mut merkle_proof_0 = Some(merkle_proof_0);
    let mut initial_proof = if open_fill_enabled() {
        None
    } else {
        Some(RecursiveProof {
            opened_rows: opened_rows_0.clone(),
            merkle_proof: merkle_proof_0.take().expect("merkle proof taken once"),
        })
    };

    // Induce basis_0 from wtns_0 opens. L0 dominates the induce phase, where the
    // sparse-prefix Fᵀ-NTT path wins; the dispatcher auto-selects it (deeper
    // levels stay dense). The dense arm's `sks_vks` table is built inside the
    // dispatcher only when that arm is taken (see [`induce_lazy_sks_enabled`]).
    let sks_vks_n1 = if induce_lazy_sks_enabled() {
        None
    } else {
        Some(eval_sk_at_vks(n1))
    };
    let _t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        sks_vks_n1.as_deref(),
        &opened_rows_0,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );
    if trace {
        let d = _t.elapsed();
        t_induce += d;
        let lb = n1 + log_inv_rate_0;
        induce_diag.push((
            0,
            n1,
            queries_0.len(),
            d.as_secs_f64() * 1e3,
            n1 >= 12 && queries_0.len() > 4 * (1usize << log_inv_rate_0) * lb.max(1),
        ));
    }

    let initial_proof = initial_proof.take().unwrap_or_else(|| RecursiveProof {
        opened_rows: opened_rows_0,
        merkle_proof: merkle_proof_0.take().expect("merkle proof taken once"),
    });

    // Introduce + glue basis_0.
    let _t = std::time::Instant::now();
    let intro_msg_0 = sc_prover.introduce_new(basis_0_induced, enforced_sum_0);
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    if ranked_deferred_induced_glue_enabled(
        config,
        log_n,
        n1,
        sc_prover.f().len(),
        num_queries_0,
        log_inv_rate_0,
        lazy_l1_ood,
        direct_fold8_mode,
    ) {
        sc_prover.glue_deferred_into_factorized_ood_fold(beta_0);
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
            let opened_rows_last: Vec<Vec<F128>> =
                gather_opened_rows(&queries_last, |q| wtns_prev.row(q), serial_par_enabled());
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
                for (j, (g, f, a)) in round_diag.iter().enumerate() {
                    eprintln!("    round {j}: grind {g:.2} ms, fold {f:.2} ms, arena={a}",);
                }
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
                for (lvl, lmc, nq, ms, ntt) in induce_diag.iter() {
                    eprintln!(
                        "    induce L{lvl}: log_msg_cols={lmc} queries={nq} ntt={ntt} {ms:.2} ms",
                    );
                }
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

        let lazy_deep_ood = ranked_deep_lazy_ood_eq_enabled(
            config,
            log_n,
            n_next,
            ood_count(i + 2),
            sc_prover.f().len(),
            direct_fold4_mode || direct_fold8_mode,
        );

        // OOD binding for the L_{i+2} commit (same as the L1 block above).
        {
            let _t = std::time::Instant::now();
            for _ in 0..ood_count(i + 2) {
                let z = challenger.sample_f128_vec(n_next);
                // The exact ranked deep levels retain eq(z[1..], ·) as
                // 2^min(11, n−1) low × high factors, exactly as the L1 block
                // above; the correction rides the level's first fold. Every
                // rollback and other geometry builds the incumbent dense
                // table.
                let (intro, y) = if lazy_deep_ood {
                    sc_prover
                        .introduce_new_ood_factorized(&z)
                        .expect("ranked factorized deep OOD introduction shape changed")
                } else {
                    let eq_z = build_eq_table_split(&z);
                    sc_prover.introduce_new_with_eval(eq_z)
                };
                challenger.observe_f128(y);
                ood_values.push(y);
                challenger.observe_f128(intro.u_0);
                challenger.observe_f128(intro.u_2);
                let beta = challenger.sample_f128();
                if lazy_deep_ood {
                    sc_prover.glue_factorized_ood(beta);
                } else {
                    sc_prover.glue(beta);
                }
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
        let opened_rows_i: Vec<Vec<F128>> =
            gather_opened_rows(&queries_i, |q| wtns_prev.row(q), serial_par_enabled());
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        if trace {
            t_opens += _t.elapsed();
        }
        // Same deletion as the L0 block: the rows move into the proof after the
        // induce has read them instead of being cloned before it.
        let mut merkle_proof_i = Some(merkle_proof_i);
        if !open_fill_enabled() {
            recursive_proofs.push(RecursiveProof {
                opened_rows: opened_rows_i.clone(),
                merkle_proof: merkle_proof_i.take().expect("merkle proof taken once"),
            });
        }

        // The dispatcher builds the dense arm's table itself on the lazy
        // path (see [`induce_lazy_sks_enabled`]); the sched-disabled arm
        // below always runs dense and so always needs it eagerly.
        let sks_vks_i = if induce_sched_enabled() && induce_lazy_sks_enabled() {
            None
        } else {
            Some(eval_sk_at_vks(n_next))
        };
        let _t = std::time::Instant::now();
        // `n_next` is exactly wtns_prev's message-column count and
        // `log_inv_rates[i+1]` its rate, so the same dense-vs-Fᵀ-NTT cost
        // dispatch L0 uses applies here. (Was hard-wired to the dense arm
        // back when the transposed NTT cost one DRAM pass per layer; the
        // blocked sweep moved the crossover well below level 1's shape.)
        let (basis_i_induced, enforced_sum_i) = if induce_sched_enabled() {
            induce_sumcheck_poly_auto(
                n_next,
                config.log_inv_rates[i + 1],
                sks_vks_i.as_deref(),
                &opened_rows_i,
                &level_rs,
                &queries_i,
                &alpha_i,
            )
        } else {
            induce_sumcheck_poly(
                n_next,
                sks_vks_i
                    .as_deref()
                    .expect("sched-disabled induce always precomputes sks_vks"),
                &opened_rows_i,
                &level_rs,
                &queries_i,
                &alpha_i,
            )
        };
        if trace {
            let d = _t.elapsed();
            t_induce += d;
            let rate = config.log_inv_rates[i + 1];
            induce_diag.push((
                i + 1,
                n_next,
                queries_i.len(),
                d.as_secs_f64() * 1e3,
                induce_sched_enabled()
                    && n_next >= 12
                    && queries_i.len()
                        > induce_ntt_crossover_c() * (1usize << rate) * (n_next + rate).max(1),
            ));
        }

        if let Some(mp) = merkle_proof_i.take() {
            recursive_proofs.push(RecursiveProof {
                opened_rows: opened_rows_i,
                merkle_proof: mp,
            });
        }

        let _t = std::time::Instant::now();
        let intro_msg_i = sc_prover.introduce_new(basis_i_induced, enforced_sum_i);
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        if ranked_deferred_induced_glue_enabled(
            config,
            log_n,
            n_next,
            sc_prover.f().len(),
            num_queries_i,
            config.log_inv_rates[i + 1],
            lazy_deep_ood,
            direct_fold8_mode,
        ) {
            sc_prover.glue_deferred_into_factorized_ood_fold(beta_i);
        } else {
            sc_prover.glue(beta_i);
        }
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
        Some(&sks_vks_n1),
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
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| wtns_0.row(q).to_vec()).collect();
    let merkle_proof_0 = merkle_multi_proof_for(&wtns_0.tree, wtns_0.block_len, &queries_0);
    t_opens += t.elapsed();
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0.clone(),
        merkle_proof: merkle_proof_0,
    };

    // ---- Induce basis from wtns_0 opens ----
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        Some(&sks_vks_n1),
        &opened_rows_0,
        &v_challenges_0,
        &queries_0,
        &alpha_0,
    );
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
        let t = std::time::Instant::now();
        let wtns_next = ligero_commit(
            sc_prover.f(),
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
        Some(&sks_vks_n1),
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
    use super::*;

    /// The gated parallel query-phase gathers must match the sequential
    /// oracles bit-for-bit — the opened rows against the incumbent
    /// `iter().map(to_vec)` gather, the multi-proof against the incumbent
    /// fused `merkle_multi_proof` walk — and a corrupted source must be
    /// caught by both routes.
    #[test]
    fn query_phase_gathers_match_sequential_oracles() {
        let mut state = 0x9A7B_5EED_F00D_u64;
        let mut random_u64 = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        // Ranked-L0-like shape (218 queries, wide rows, deep tree — both
        // gathers above their par floors) plus a tiny shape (below both
        // floors — the par route must fall back and still match).
        for &(log_leaves, n_queries, row_len) in &[(12usize, 218usize, 64usize), (4, 3, 2)] {
            let num_leaves = 1usize << log_leaves;
            let mut tree: Vec<Hash> = Vec::with_capacity(2 * num_leaves - 1);
            for _ in 0..(2 * num_leaves - 1) {
                let mut h = [0u8; 32];
                h[..8].copy_from_slice(&random_u64().to_le_bytes());
                h[8..16].copy_from_slice(&random_u64().to_le_bytes());
                tree.push(h);
            }
            let codeword: Vec<F128> = (0..num_leaves * row_len)
                .map(|_| F128::new(random_u64(), random_u64()))
                .collect();
            let mut queries: Vec<usize> = (0..n_queries)
                .map(|_| random_u64() as usize % num_leaves)
                .collect();
            queries.sort_unstable();
            queries.dedup();

            let row = |q: usize| &codeword[q * row_len..(q + 1) * row_len];
            let rows_seq = gather_opened_rows(&queries, row, false);
            let rows_par = gather_opened_rows(&queries, row, true);
            assert_eq!(rows_par, rows_seq, "rows log_leaves={log_leaves}");

            let proof_seq = multi_proof_gather(&tree, num_leaves, &queries, false);
            let proof_par = multi_proof_gather(&tree, num_leaves, &queries, true);
            assert_eq!(proof_par, proof_seq, "multi-proof log_leaves={log_leaves}");
            assert_eq!(
                proof_seq,
                merkle::merkle_multi_proof(&tree, num_leaves, &queries),
                "sequential arm drifted from the merkle oracle"
            );

            // Negative controls: a corrupted codeword row and a corrupted
            // emitted sibling must both reach the parallel outputs.
            let mut bad_codeword = codeword.clone();
            bad_codeword[queries[0] * row_len] += F128::ONE;
            let bad_row = |q: usize| &bad_codeword[q * row_len..(q + 1) * row_len];
            assert_ne!(
                gather_opened_rows(&queries, bad_row, true),
                rows_seq,
                "corrupted row went undetected log_leaves={log_leaves}"
            );
            let sibling_indices = merkle::merkle_multi_proof_sibling_indices(num_leaves, &queries);
            assert!(!sibling_indices.is_empty());
            let mut bad_tree = tree.clone();
            bad_tree[sibling_indices[0]][0] ^= 1;
            assert_ne!(
                multi_proof_gather(&bad_tree, num_leaves, &queries, true),
                proof_seq,
                "corrupted sibling went undetected log_leaves={log_leaves}"
            );
        }
    }

    #[test]
    fn direct_ab_materialization_matches_full_basis_oracle() {
        let mut state = 0xD1CE_F01D_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(state, state.rotate_left(29))
        };
        let n = 1usize << 8;
        let f: Vec<F128> = (0..n).map(|_| random()).collect();
        let ordinary_c: Vec<F128> = (0..n).map(|_| random()).collect();
        let r0 = random();
        let r1 = random();
        let suffix: Vec<F128> = (0..8).map(|_| random()).collect();
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
        let (eq_lo, eq_hi) = super::super::ring_switch::build_eq_split(&suffix[2..], 3);
        let direct = vec![super::super::ring_switch::DirectFold2Factors {
            eq_lo,
            eq_hi,
            low_eq: build_eq_table(&suffix[..2]).try_into().unwrap(),
            table: super::super::ring_switch::build_fold_byte_table(&scaled_rdp),
            products: None,
        }];

        let mut want_f = f.clone();
        let mut want_b = combined_full;
        partial_eval_lsb_one(&mut want_f, r0);
        partial_eval_lsb_one(&mut want_b, r0);
        partial_eval_lsb_one(&mut want_f, r1);
        partial_eval_lsb_one(&mut want_b, r1);
        let want_msg = round_msg_lsb(&want_f, &want_b);
        let (got_f, got_b, got_msg) = materialize_direct_ab_fold2(f, ordinary_c, &direct, r0, r1);
        assert_eq!(got_f, want_f);
        assert_eq!(got_b, want_b);
        assert_eq!(got_msg, want_msg);
    }

    #[test]
    fn direct_all_full_proof_matches_ordinary_transcript() {
        use crate::challenger::Challenger;

        let log_n = 12;
        let initial_k = 2;
        let k_0 = 2;
        let log_inv_rate = 1;
        let mut rng = crate::challenger::RandomChallenger::new(0xD1CE_AB02);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let suffix_ab: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let suffix_c: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let scaled_rdp_ab: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| rng.sample_f128())
                .collect::<Vec<_>>(),
        );
        let scaled_rdp_c: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| rng.sample_f128())
                .collect::<Vec<_>>(),
        );
        let basis_ab =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix_ab), &scaled_rdp_ab);
        let basis_c =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix_c), &scaled_rdp_c);
        let combined_basis: Vec<F128> = basis_ab
            .iter()
            .zip(basis_c)
            .map(|(&ab, c)| ab + c)
            .collect();
        let target = poly
            .iter()
            .zip(combined_basis.iter())
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |acc, value| acc + value);
        let (round0, lookahead) = super::super::round0_and_round1_lookahead(&poly, &combined_basis);
        let direct = [(&suffix_ab, &scaled_rdp_ab), (&suffix_c, &scaled_rdp_c)]
            .into_iter()
            .map(|(suffix, scaled_rdp)| {
                let (eq_lo, eq_hi) =
                    super::super::ring_switch::build_eq_split(&suffix[2..], (log_n - 2) / 2);
                super::super::ring_switch::DirectFold2Factors {
                    eq_lo,
                    eq_hi,
                    low_eq: build_eq_table(&suffix[..2]).try_into().unwrap(),
                    table: super::super::ring_switch::build_fold_byte_table(scaled_rdp),
                    products: None,
                }
            })
            .collect();

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
            crate::challenger::FsChallenger::new(b"direct-all-proof-byte-oracle");
        let ordinary = recursive_prover_with_basis_precomputed_round0(
            &cfg,
            poly.clone(),
            combined_basis,
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            None,
            &mut ordinary_challenger,
        );
        let mut direct_challenger =
            crate::challenger::FsChallenger::new(b"direct-all-proof-byte-oracle");
        let got = recursive_prover_with_basis_direct_ab_fold2(
            &cfg,
            poly,
            Vec::new(),
            direct,
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            lookahead,
            None,
            &mut direct_challenger,
        );

        assert_eq!(got, ordinary);
    }

    /// Direct-fold4 (two direct claims, no ordinary basis, rounds 0..3 from
    /// the 16×16 product matrices, one 16:1 materialize) must emit the SAME
    /// proof bytes as the ordinary route with a materialized combined basis,
    /// and the proof must verify. `initial_k = 6`, `k_0 = 2` mirror the ranked
    /// cadence at a small `log_n`.
    #[test]
    fn direct_fold4_full_proof_and_claim_bytes_match_ordinary() {
        use crate::challenger::Challenger;

        for (log_n, seed) in [(12usize, 0xD1CE_F004u64), (13, 0xD1CE_F005)] {
            let initial_k = 6;
            let k_0 = 2;
            let log_inv_rate = 3;
            let mut rng = crate::challenger::RandomChallenger::new(seed);
            let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
            let suffix_ab: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
            let suffix_c: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
            let scaled_rdp_ab: Vec<F128> = build_eq_table(
                &(0..crate::pcs::LOG_PACKING)
                    .map(|_| rng.sample_f128())
                    .collect::<Vec<_>>(),
            );
            let scaled_rdp_c: Vec<F128> = build_eq_table(
                &(0..crate::pcs::LOG_PACKING)
                    .map(|_| rng.sample_f128())
                    .collect::<Vec<_>>(),
            );
            let basis_ab = super::super::ring_switch::fold_b128_elems(
                &build_eq_table(&suffix_ab),
                &scaled_rdp_ab,
            );
            let basis_c = super::super::ring_switch::fold_b128_elems(
                &build_eq_table(&suffix_c),
                &scaled_rdp_c,
            );
            let combined_basis: Vec<F128> = basis_ab
                .iter()
                .zip(basis_c.iter())
                .map(|(&ab, &c)| ab + c)
                .collect();
            let target = poly
                .iter()
                .zip(combined_basis.iter())
                .map(|(&f, &b)| f * b)
                .fold(F128::ZERO, |acc, value| acc + value);

            let direct: Vec<super::super::ring_switch::DirectFold4Factors> = [
                (&suffix_ab, &scaled_rdp_ab, &basis_ab),
                (&suffix_c, &scaled_rdp_c, &basis_c),
            ]
            .into_iter()
            .map(|(suffix, scaled_rdp, basis)| {
                let mut products = [F128::ZERO; 256];
                for high in 0..poly.len() / 16 {
                    for e in 0..16 {
                        for d in 0..16 {
                            products[16 * e + d] += poly[16 * high + e] * basis[16 * high + d];
                        }
                    }
                }
                let (eq_lo, eq_hi) =
                    super::super::ring_switch::build_eq_split(&suffix[4..], (log_n - 4) / 2);
                super::super::ring_switch::DirectFold4Factors {
                    eq_lo,
                    eq_hi,
                    low_eq: build_eq_table(&suffix[..4]).try_into().unwrap(),
                    table: super::super::ring_switch::build_fold_byte_table(scaled_rdp),
                    products,
                }
            })
            .collect();
            let (round0, round1, round2, round3) =
                super::super::messages_from_direct_products_fold4(&direct);
            // The product-derived round-0/1 messages must equal the sweep's.
            let (round0_ref, round1_ref) =
                super::super::round0_and_round1_lookahead(&poly, &combined_basis);
            assert_eq!(round0, round0_ref, "round-0 message from products");
            assert_eq!(round1, round1_ref, "round-1 lookahead from products");

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
                None,
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
                None,
                &mut direct_challenger,
            );

            assert_eq!(got, ordinary, "direct-fold4 proof differs at log_n={log_n}");
            assert_eq!(
                bincode::serialize(&(got.clone(), target))
                    .expect("serialize direct-fold4 proof/claim"),
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
    }

    /// x86 NT fold+message leaf must be bit-identical to `fold_pairs` +
    /// the reload message loop (NT is a cache-hint only).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold_and_msg_nt_leaf_x86_matches_generic() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut f128 = || F128 {
            lo: next(),
            hi: next(),
        };
        for (n_pairs, base) in [(8usize, 0usize), (32, 4), (64, 16), (2048, 0)] {
            let total = 2 * (base + n_pairs);
            let f: Vec<F128> = (0..total).map(|_| f128()).collect();
            let b: Vec<F128> = (0..total).map(|_| f128()).collect();
            let r = f128();

            let mut fc_ref = vec![F128::ZERO; n_pairs];
            let mut bc_ref = vec![F128::ZERO; n_pairs];
            crate::field::f128_slice::fold_pairs(&f, base, &mut fc_ref, r);
            crate::field::f128_slice::fold_pairs(&b, base, &mut bc_ref, r);
            let mut u0_ref = F128::ZERO;
            let mut u2_ref = F128::ZERO;
            let mut k = 0;
            while k + 1 < n_pairs {
                let (f0, f1, b0, b1) = (fc_ref[k], fc_ref[k + 1], bc_ref[k], bc_ref[k + 1]);
                u0_ref += f0 * b0;
                u2_ref += (f0 + f1) * (b0 + b1);
                k += 2;
            }

            let mut fc_nt = vec![F128::ZERO; n_pairs];
            let mut bc_nt = vec![F128::ZERO; n_pairs];
            let mut stage_f = vec![F128::ZERO; n_pairs.max(8)];
            let mut stage_b = vec![F128::ZERO; n_pairs.max(8)];
            // SAFETY: avx512f+vpclmulqdq cfg-guaranteed; slices sized per contract.
            let (u0_nt, u2_nt) = unsafe {
                super::fold_and_msg_chunk_nt_x86(
                    &f,
                    &b,
                    base,
                    &mut fc_nt,
                    &mut bc_nt,
                    r,
                    &mut stage_f,
                    &mut stage_b,
                )
            };
            assert_eq!(fc_ref, fc_nt, "folded f mismatch n_pairs={n_pairs}");
            assert_eq!(bc_ref, bc_nt, "folded b mismatch n_pairs={n_pairs}");
            assert_eq!(u0_ref, u0_nt, "u0 mismatch n_pairs={n_pairs}");
            assert_eq!(u2_ref, u2_nt, "u2 mismatch n_pairs={n_pairs}");
        }
    }

    /// The NT fold+message leaf must produce bit-identical folded outputs
    /// and (u_0, u_2) partials to the generic chunk body it replaces on
    /// large rounds (the NT hint changes cache allocation only).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fold_and_msg_nt_leaf_matches_generic() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut f128 = || F128 {
            lo: next(),
            hi: next(),
        };
        for (n_pairs, base) in [(8usize, 0usize), (32, 4), (64, 16)] {
            let total = 2 * (base + n_pairs);
            let f: Vec<F128> = (0..total).map(|_| f128()).collect();
            let b: Vec<F128> = (0..total).map(|_| f128()).collect();
            let r = f128();

            // Generic reference: fold_pairs then the reload message loop.
            let mut fc_ref = vec![F128::ZERO; n_pairs];
            let mut bc_ref = vec![F128::ZERO; n_pairs];
            crate::field::f128_slice::fold_pairs(&f, base, &mut fc_ref, r);
            crate::field::f128_slice::fold_pairs(&b, base, &mut bc_ref, r);
            let mut u0_ref = F128::ZERO;
            let mut u2_ref = F128::ZERO;
            let mut k = 0;
            while k + 1 < n_pairs {
                let (f0, f1, b0, b1) = (fc_ref[k], fc_ref[k + 1], bc_ref[k], bc_ref[k + 1]);
                u0_ref += f0 * b0;
                u2_ref += (f0 + f1) * (b0 + b1);
                k += 2;
            }

            let mut fc_nt = vec![F128::ZERO; n_pairs];
            let mut bc_nt = vec![F128::ZERO; n_pairs];
            // SAFETY: aes is cfg-guaranteed; slices sized per the contract.
            let (u0_nt, u2_nt) =
                unsafe { fold_and_msg_chunk_nt_neon(&f, &b, base, &mut fc_nt, &mut bc_nt, r) };
            assert_eq!(fc_ref, fc_nt, "folded f mismatch n_pairs={n_pairs}");
            assert_eq!(bc_ref, bc_nt, "folded b mismatch n_pairs={n_pairs}");
            assert_eq!(u0_ref, u0_nt, "u0 mismatch n_pairs={n_pairs}");
            assert_eq!(u2_ref, u2_nt, "u2 mismatch n_pairs={n_pairs}");
        }
    }

    /// Oracle: the all-NEON SoA leaf is bit-identical to the previous
    /// GPR-mixed NT leaf AND the generic fold+reload reference, on random
    /// inputs at several shapes (incl. an odd base and a larger power-of-two
    /// chunk like the production `CHUNK`).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fold_and_msg_soa_leaf_matches_nt_and_generic() {
        let mut state = 0x0fed_cba9_8765_4321_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut f128 = || F128 {
            lo: next(),
            hi: next(),
        };
        for (n_pairs, base) in [(2usize, 0usize), (8, 0), (32, 4), (64, 16), (2048, 2048)] {
            let total = 2 * (base + n_pairs);
            let f: Vec<F128> = (0..total).map(|_| f128()).collect();
            let b: Vec<F128> = (0..total).map(|_| f128()).collect();
            let r = f128();

            // Generic reference: fold_pairs then the reload message loop.
            let mut fc_ref = vec![F128::ZERO; n_pairs];
            let mut bc_ref = vec![F128::ZERO; n_pairs];
            crate::field::f128_slice::fold_pairs(&f, base, &mut fc_ref, r);
            crate::field::f128_slice::fold_pairs(&b, base, &mut bc_ref, r);
            let mut u0_ref = F128::ZERO;
            let mut u2_ref = F128::ZERO;
            let mut k = 0;
            while k + 1 < n_pairs {
                let (f0, f1, b0, b1) = (fc_ref[k], fc_ref[k + 1], bc_ref[k], bc_ref[k + 1]);
                u0_ref += f0 * b0;
                u2_ref += (f0 + f1) * (b0 + b1);
                k += 2;
            }

            // Previous NT leaf.
            let mut fc_nt = vec![F128::ZERO; n_pairs];
            let mut bc_nt = vec![F128::ZERO; n_pairs];
            // SAFETY: aes is cfg-guaranteed; slices sized per the contract.
            let (u0_nt, u2_nt) =
                unsafe { fold_and_msg_chunk_nt_neon(&f, &b, base, &mut fc_nt, &mut bc_nt, r) };

            // New SoA leaf, both store variants.
            let mut fc_soa = vec![F128::ZERO; n_pairs];
            let mut bc_soa = vec![F128::ZERO; n_pairs];
            // SAFETY: aes is cfg-guaranteed; slices sized per the contract.
            let (u0_soa, u2_soa) = unsafe {
                fold_and_msg_chunk_nt_neon_soa::<true>(&f, &b, base, &mut fc_soa, &mut bc_soa, r)
            };
            let mut fc_soa_r = vec![F128::ZERO; n_pairs];
            let mut bc_soa_r = vec![F128::ZERO; n_pairs];
            // SAFETY: as above.
            let (u0_soa_r, u2_soa_r) = unsafe {
                fold_and_msg_chunk_nt_neon_soa::<false>(
                    &f,
                    &b,
                    base,
                    &mut fc_soa_r,
                    &mut bc_soa_r,
                    r,
                )
            };

            assert_eq!(fc_ref, fc_soa, "folded f mismatch n_pairs={n_pairs}");
            assert_eq!(bc_ref, bc_soa, "folded b mismatch n_pairs={n_pairs}");
            assert_eq!(u0_ref, u0_soa, "u0 vs generic n_pairs={n_pairs}");
            assert_eq!(u2_ref, u2_soa, "u2 vs generic n_pairs={n_pairs}");
            assert_eq!(fc_nt, fc_soa, "folded f vs NT n_pairs={n_pairs}");
            assert_eq!(bc_nt, bc_soa, "folded b vs NT n_pairs={n_pairs}");
            assert_eq!(u0_nt, u0_soa, "u0 vs NT n_pairs={n_pairs}");
            assert_eq!(u2_nt, u2_soa, "u2 vs NT n_pairs={n_pairs}");
            assert_eq!(fc_soa, fc_soa_r, "folded f NT vs stp n_pairs={n_pairs}");
            assert_eq!(bc_soa, bc_soa_r, "folded b NT vs stp n_pairs={n_pairs}");
            assert_eq!(u0_soa, u0_soa_r, "u0 NT vs stp n_pairs={n_pairs}");
            assert_eq!(u2_soa, u2_soa_r, "u2 NT vs stp n_pairs={n_pairs}");
        }
    }

    /// Serial `half<4096` leaf: one-mul fold is bit-identical to two-mul +
    /// the scalar message loop.
    #[test]
    fn fold_and_msg_serial_leaf_matches_two_mul() {
        let mut state = 0xA5A5_5A5A_1234_5678_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut f128 = || F128 {
            lo: next(),
            hi: next(),
        };
        for n in [2usize, 4, 8, 16, 64, 256] {
            let f: Vec<F128> = (0..n).map(|_| f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| f128()).collect();
            let r = f128();
            let one_plus_r = F128::ONE + r;
            let half = n / 2;
            let mut nf_ref = Vec::with_capacity(half);
            let mut nb_ref = Vec::with_capacity(half);
            for j in 0..half {
                nf_ref.push(f[2 * j] * one_plus_r + f[2 * j + 1] * r);
                nb_ref.push(b[2 * j] * one_plus_r + b[2 * j + 1] * r);
            }
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            let mut k = 0;
            while k + 1 < half {
                u0 += nf_ref[k] * nb_ref[k];
                u2 += (nf_ref[k] + nf_ref[k + 1]) * (nb_ref[k] + nb_ref[k + 1]);
                k += 2;
            }
            let (nf, nb, msg) = fold_and_msg_lsb(&f, &b, r, None);
            assert_eq!(&nf[..], nf_ref.as_slice(), "nf n={n}");
            assert_eq!(&nb[..], nb_ref.as_slice(), "nb n={n}");
            assert_eq!(msg.u_0, u0, "u0 n={n}");
            assert_eq!(msg.u_2, u2, "u2 n={n}");
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
            // Level-1-shaped: rate 1/4 with many queries — the case the
            // retuned crossover now sends down the Fᵀ-NTT arm.
            (14, 2, 3, 106),
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
                // Baseline: scatter then dense transpose.
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

    /// Every possible local source position must expand to the incumbent
    /// dense window, across low/high global window indices and ranked depths.
    #[test]
    fn singleton_window_matches_dense_for_every_position() {
        use crate::challenger::Challenger;
        const K: usize = 8;
        for &log_d in &[12usize, 18, 20] {
            let ntt = AdditiveNttF128::standard(log_d);
            let last_w = (1usize << (log_d - K)) - 1;
            for &w in &[0usize, 1, last_w] {
                let mut ch = crate::challenger::RandomChallenger::new(
                    0x51A6_1E70 ^ ((log_d * 257 + w) as u64),
                );
                for p in 0..(1usize << K) {
                    let value = if p == 0 { F128::ONE } else { ch.sample_f128() };
                    let mut dense = vec![F128::ZERO; 1usize << K];
                    dense[p] = value;
                    let want = transpose_forward_ntt_window_dense(&ntt, log_d, K, w, dense);
                    let got = transpose_forward_ntt_window_singleton(&ntt, log_d, K, w, p, value);
                    assert_eq!(got, want, "log_d={log_d} w={w} p={p}");
                }
            }
        }
    }

    fn sparse_dense_oracle(log_d: usize, positions: &[usize], values: &[F128]) -> Vec<F128> {
        let mut dense = vec![F128::ZERO; 1usize << log_d];
        for (&p, &v) in positions.iter().zip(values) {
            dense[p] += v;
        }
        let ntt = AdditiveNttF128::standard(log_d);
        transpose_forward_ntt(&ntt, &mut dense, log_d);
        dense
    }

    /// A window with multiple query positions must take the incumbent path;
    /// the forced singleton mode must report no hits and one collision.
    #[test]
    fn singleton_sparse_collision_falls_back_exactly() {
        use crate::challenger::Challenger;
        let log_d = 12usize;
        let w = 3usize;
        let positions = vec![(w << 8) + 1, (w << 8) + 7, (w << 8) + 201];
        let mut ch = crate::challenger::RandomChallenger::new(0xC011_1510);
        let values: Vec<F128> = positions.iter().map(|_| ch.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_d);
        let want = sparse_dense_oracle(log_d, &positions, &values);
        let (got, stats) = transpose_forward_ntt_sparse_inner(
            &ntt,
            &positions,
            &values,
            log_d,
            Some(true),
            Some(8),
        );
        let (off, _) = transpose_forward_ntt_sparse_inner(
            &ntt,
            &positions,
            &values,
            log_d,
            Some(false),
            Some(8),
        );
        assert_eq!(got, want);
        assert_eq!(off, want);
        assert_eq!(stats.singleton_hits, 0);
        assert_eq!(stats.multi_hits, 1);
        assert_eq!(stats.collision_fallbacks, 0);
    }

    /// Mixed singleton and collision windows exercise both branches in one
    /// sparse transform and must still match the full dense transpose.
    #[test]
    fn singleton_sparse_mixed_windows_match_dense() {
        use crate::challenger::Challenger;
        let log_d = 12usize;
        let positions = vec![
            (1usize << 8) + 2,
            (1usize << 8) + 91,
            (5usize << 8) + 17,
            (11usize << 8) + 233,
        ];
        let mut ch = crate::challenger::RandomChallenger::new(0x51A6_C011);
        let values: Vec<F128> = positions.iter().map(|_| ch.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_d);
        let want = sparse_dense_oracle(log_d, &positions, &values);
        let (got, stats) = transpose_forward_ntt_sparse_inner(
            &ntt,
            &positions,
            &values,
            log_d,
            Some(true),
            Some(8),
        );
        assert_eq!(got, want);
        assert_eq!(stats.singleton_hits, 2);
        assert_eq!(stats.multi_hits, 1);
        assert_eq!(stats.collision_fallbacks, 0);
    }

    #[test]
    fn induce_window_k_selects_documented_optima() {
        // Ranked L0 (log_d = 20, 218 queries, 16 threads): k = 12.
        assert_eq!(induce_window_k(20, 218, 16, true), 12);
        // Ranked L1 induce (log_d = 18, 106 queries): k = 11.
        assert_eq!(induce_window_k(18, 106, 16, true), 11);
        // Shortcut off: the dense-window optimum stays at the incumbent 8.
        assert_eq!(induce_window_k(20, 218, 16, false), 8);
        assert_eq!(induce_window_k(18, 106, 16, false), 8);
        assert_eq!(induce_window_k(12, 218, 16, false), 8);
        // Small domains and empty query sets: no sparse prefix.
        assert_eq!(induce_window_k(11, 218, 16, true), 0);
        assert_eq!(induce_window_k(20, 0, 16, true), 0);
        // Parallelism clamp: 256 threads at L0 caps k_par = 20 − (8 + 2) = 10.
        assert_eq!(induce_window_k(20, 218, 256, true), 10);
        // k_par below 8 floors at the incumbent 8.
        assert_eq!(induce_window_k(12, 106, 16, true), 8);
        assert_eq!(induce_window_k(20, 218, 4096, true), 8);
        // Memory cap binds: Q = 1, 1 thread would pick k = 18; cap holds 12.
        assert_eq!(induce_window_k(20, 1, 1, true), 12);
        // Very high query counts push k back DOWN.
        assert_eq!(induce_window_k(20, 100_000, 16, true), 8);
    }

    /// Multi-singleton XOR expansion vs the dense window across the
    /// `nnz ≤ k/2` threshold, including one step past it (forced multi at
    /// `nnz = k/2 + 1`): the shortcut is F2-linear, so it must match the
    /// dense window for ANY nnz — the threshold is a cost boundary, not a
    /// correctness one.
    #[test]
    fn multi_singleton_matches_dense_across_threshold() {
        use crate::challenger::Challenger;
        let log_d = 14usize;
        for &k in &[8usize, 10, 12] {
            let ntt = AdditiveNttF128::standard(log_d);
            let wlen = 1usize << k;
            let last_w = (1usize << (log_d - k)) - 1;
            for &w in &[0usize, last_w] {
                let mut ch =
                    crate::challenger::RandomChallenger::new(0x0117_15EE ^ ((k * 8191 + w) as u64));
                for nnz in 1..=(k / 2 + 1) {
                    let mut ps: Vec<usize> = vec![0, wlen - 1];
                    ps.truncate(nnz);
                    while ps.len() < nnz {
                        let p = (ch.sample_f128().lo as usize) % wlen;
                        if !ps.contains(&p) {
                            ps.push(p);
                        }
                    }
                    let vs: Vec<F128> = ps.iter().map(|_| ch.sample_f128()).collect();
                    let mut dense = vec![F128::ZERO; wlen];
                    for (&p, &v) in ps.iter().zip(&vs) {
                        dense[p] += v;
                    }
                    let (_, want) = transpose_forward_ntt_window_dense(&ntt, log_d, k, w, dense);
                    let mut buf = take_singleton_buf(k);
                    expand_singleton_into(&ntt, log_d, k, w, ps[0], vs[0], &mut buf);
                    let mut scratch = take_singleton_buf(k);
                    for i in 1..nnz {
                        expand_singleton_into(&ntt, log_d, k, w, ps[i], vs[i], &mut scratch);
                        for (d, &s) in buf.iter_mut().zip(scratch.iter()) {
                            *d += s;
                        }
                    }
                    assert_eq!(buf, want, "k={k} w={w} nnz={nnz}");
                }
            }
        }
    }

    /// End-to-end sparse-vs-dense at forced widths for ranked-relevant depths.
    #[test]
    fn sparse_transpose_matches_dense_at_every_forced_k() {
        use crate::challenger::Challenger;
        for &log_d in &[12usize, 16, 18] {
            let n = 1usize << log_d;
            let ntt = AdditiveNttF128::standard(log_d);
            let mut ch = crate::challenger::RandomChallenger::new(0x7357_0A11 ^ log_d as u64);
            let mut positions: Vec<usize> = Vec::new();
            while positions.len() < 80.min(n / 4) {
                let p = (ch.sample_f128().lo as usize) % n;
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
            let values: Vec<F128> = positions.iter().map(|_| ch.sample_f128()).collect();
            let want = sparse_dense_oracle(log_d, &positions, &values);
            for k in 8..=12.min(log_d) {
                let (got, _) = transpose_forward_ntt_sparse_inner(
                    &ntt,
                    &positions,
                    &values,
                    log_d,
                    None,
                    Some(k),
                );
                assert_eq!(got, want, "log_d={log_d} k={k}");
            }
        }
    }

    /// The fused parallel densify must be BYTE-identical to the incumbent
    /// two-pass form (`vec![F128::ZERO; n]` + serial scatter of the active
    /// windows) — including at the ranked L0 shape (log_d=20, k=8, ~214 of
    /// 4096 windows active) and at the degenerate all-active / none-active
    /// ends. Covers the arm the env switch selects, which resolves once per
    /// process and so cannot be flipped inside one test binary.
    #[test]
    fn densify_windows_fused_matches_two_pass() {
        use crate::challenger::Challenger;
        for &(log_d, k) in &[
            (12usize, 8usize),
            (16, 8),
            (20, 8),
            (13, 8),
            (20, 12),
            (18, 11),
            (16, 10),
        ] {
            let n = 1usize << log_d;
            let nwin = n >> k;
            for &active in &[0usize, 1, 214.min(nwin), nwin] {
                let mut ch = crate::challenger::RandomChallenger::new(
                    0x0DE5_1F17 ^ ((log_d * 8191 + active) as u64),
                );
                // Distinct window ids, spread over the whole range.
                let mut processed: Vec<(usize, Vec<F128>)> = Vec::new();
                let mut seen = vec![false; nwin];
                while processed.len() < active {
                    let w = (ch.sample_f128().lo as usize) % nwin;
                    if seen[w] {
                        continue;
                    }
                    seen[w] = true;
                    processed.push((w, (0..(1usize << k)).map(|_| ch.sample_f128()).collect()));
                }
                // Incumbent two-pass reference.
                let mut want = vec![F128::ZERO; n];
                for (w, buf) in &processed {
                    want[(w << k)..((w + 1) << k)].copy_from_slice(buf);
                }
                let got = densify_windows_fused(n, k, window_slots(nwin, processed));
                assert_eq!(got.len(), want.len(), "log_d={log_d} active={active}");
                assert!(got == want, "log_d={log_d} k={k} active={active}");
            }
        }
    }

    /// The split-product eq table must be BYTE-identical to the serial
    /// doubling recurrence at every width the open can ask for.
    #[test]
    fn eq_split_matches_serial() {
        use crate::challenger::Challenger;
        for d in 0..=20usize {
            let mut ch = crate::challenger::RandomChallenger::new(0x3EE7_u64 ^ d as u64);
            let point: Vec<F128> = (0..d).map(|_| ch.sample_f128()).collect();
            assert_eq!(
                build_eq_table_split(&point),
                build_eq_table(&point),
                "d={d}"
            );
        }
    }

    /// Retaining the OOD equality as low/high factors must reproduce both
    /// transcript coefficients and the claimed multilinear evaluation from
    /// the ordinary dense equality table, including the ranked d=19 shape.
    #[test]
    fn factorized_ood_intro_matches_dense() {
        use crate::challenger::Challenger;
        for &d in &[4usize, 12, 19] {
            let mut ch = crate::challenger::RandomChallenger::new(0xFA17_00D0 ^ d as u64);
            let z = ch.sample_f128_vec(d);
            let f: Vec<F128> = (0..(1usize << d)).map(|_| ch.sample_f128()).collect();
            let dense = build_eq_table(&z);
            let want = round_msg_and_eval_lsb(&f, &dense);
            let split = (d - 1).min(LAZY_OOD_EQ_SPLIT_LOW_LOG);
            let eq_lo = build_eq_table(&z[1..1 + split]);
            let eq_hi = build_eq_table(&z[1 + split..]);
            let got = round_msg_and_eval_lsb_factorized_eq(&f, &eq_lo, &eq_hi, z[0]);
            assert_eq!(got, want, "factorized OOD introduction d={d}");
        }
    }

    /// Injecting the retained equality after folding the incumbent basis is
    /// exactly the same state and next message as materializing/gluing the
    /// full equality before that fold.
    #[test]
    fn lazy_ood_corrected_fold_matches_dense_glue() {
        use crate::challenger::Challenger;
        let d = 13usize;
        let n = 1usize << d;
        let mut ch = crate::challenger::RandomChallenger::new(0x1A2E_00D5);
        let z = ch.sample_f128_vec(d);
        let f: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let b: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let beta = ch.sample_f128();
        let r = ch.sample_f128();

        let dense_eq = build_eq_table(&z);
        let mut dense_basis = b.clone();
        crate::field::f128_slice::add_scaled(&mut dense_basis, &dense_eq, beta);
        let (want_f, want_b, want_msg) = fold_and_msg_lsb(&f, &dense_basis, r, None);

        let eq_lo = build_eq_table(&z[1..12]);
        let eq_hi = build_eq_table(&z[12..]);
        assert_eq!(eq_lo.len(), 2048);
        let gamma = beta * (F128::ONE + z[0] + r);
        let (got_f, got_b, got_msg) =
            fold_and_msg_lsb_inner(&f, &b, r, None, Some((&eq_lo, &eq_hi, gamma)), None);
        assert_eq!(&*got_f, &*want_f);
        assert_eq!(&*got_b, &*want_b);
        assert_eq!(got_msg, want_msg);
    }

    /// Exercise the actual prover state machine across the operation that sits
    /// between L1 OOD glue and its consuming fold: an ordinary induced-basis
    /// introduce/glue. The retained equality must commute with that update and
    /// leave the complete transcript, target, witness, and basis identical.
    #[test]
    fn lazy_ood_state_machine_matches_dense_across_ordinary_glue() {
        use crate::challenger::Challenger;
        let d = 13usize;
        let n = 1usize << d;
        let mut ch = crate::challenger::RandomChallenger::new(0x57A7_E00D);
        let f: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let b: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let z = ch.sample_f128_vec(d);
        let ordinary: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let ordinary_claim = ch.sample_f128();
        let beta = ch.sample_f128();
        let alpha = ch.sample_f128();
        let r = ch.sample_f128();
        let target = ch.sample_f128();

        let (mut dense, dense_start) = SumcheckProver::new(f.clone(), b.clone(), target);
        let (mut lazy, lazy_start) = SumcheckProver::new(f.clone(), b.clone(), target);
        let (mut deferred, deferred_start) = SumcheckProver::new(f, b, target);
        assert_eq!(lazy_start, dense_start);
        assert_eq!(deferred_start, dense_start);

        let (dense_intro, dense_y) = dense.introduce_new_with_eval(build_eq_table(&z));
        let (lazy_intro, lazy_y) = lazy.introduce_new_ood_factorized(&z).unwrap();
        let (deferred_intro, deferred_y) = deferred.introduce_new_ood_factorized(&z).unwrap();
        assert_eq!((lazy_intro, lazy_y), (dense_intro, dense_y));
        assert_eq!((deferred_intro, deferred_y), (dense_intro, dense_y));
        dense.glue(beta);
        lazy.glue_factorized_ood(beta);
        deferred.glue_factorized_ood(beta);

        let dense_ordinary = dense.introduce_new(ordinary.clone(), ordinary_claim);
        let lazy_ordinary = lazy.introduce_new(ordinary.clone(), ordinary_claim);
        let deferred_ordinary = deferred.introduce_new(ordinary, ordinary_claim);
        assert_eq!(lazy_ordinary, dense_ordinary);
        assert_eq!(deferred_ordinary, dense_ordinary);
        dense.glue(alpha);
        lazy.glue(alpha);
        deferred.glue_deferred_into_factorized_ood_fold(alpha);
        assert!(deferred.pending_fold_basis.is_some());

        let dense_msg = dense.fold(r);
        assert_eq!(lazy.fold(r), dense_msg);
        assert_eq!(deferred.fold(r), dense_msg);
        assert!(deferred.pending_fold_basis.is_none());
        assert_eq!(&*lazy.f, &*dense.f);
        assert_eq!(&*lazy.combined_basis, &*dense.combined_basis);
        assert_eq!(lazy.t_r, dense.t_r);
        assert_eq!(lazy.transcript, dense.transcript);
        assert_eq!(&*deferred.f, &*dense.f);
        assert_eq!(&*deferred.combined_basis, &*dense.combined_basis);
        assert_eq!(deferred.t_r, dense.t_r);
        assert_eq!(deferred.transcript, dense.transcript);
    }

    #[test]
    fn ranked_lazy_ood_selector_is_exact() {
        let security = LigeritoSecurityConfig::from_toml_str(include_str!(
            "../../configs/ligerito/m32_fast.toml"
        ))
        .unwrap();
        let (config, _) = security.to_prover_verifier_configs().unwrap();
        let selected = |log_n, n1, count, len, direct, platform, disabled| {
            ranked_l1_lazy_ood_eq_selected(
                &config, log_n, n1, count, len, direct, platform, disabled,
            )
        };
        assert!(selected(25, 19, 1, 1 << 19, true, true, false));
        assert!(!selected(24, 19, 1, 1 << 19, true, true, false));
        assert!(!selected(25, 18, 1, 1 << 19, true, true, false));
        assert!(!selected(25, 19, 2, 1 << 19, true, true, false));
        assert!(!selected(25, 19, 1, 1 << 18, true, true, false));
        assert!(!selected(25, 19, 1, 1 << 19, false, true, false));
        assert!(!selected(25, 19, 1, 1 << 19, true, false, false));
        assert!(!selected(25, 19, 1, 1 << 19, true, true, true));
    }

    #[test]
    fn ranked_deferred_induced_glue_selector_is_exact() {
        let security = LigeritoSecurityConfig::from_toml_str(include_str!(
            "../../configs/ligerito/m32_fast.toml"
        ))
        .unwrap();
        let (mut config, _) = security.to_prover_verifier_configs().unwrap();
        config.merkle_hash = HashKind::Blake3;
        let selected = |config: &ProverConfig,
                        log_n,
                        n_level,
                        len,
                        queries,
                        rate,
                        lazy,
                        direct8,
                        platform,
                        disabled| {
            ranked_deferred_induced_glue_selected(
                config, log_n, n_level, len, queries, rate, lazy, direct8, platform, disabled,
            )
        };
        for &(n_level, queries, rate) in &[
            (19, 218, 1),
            (16, 106, 2),
            (13, 71, 3),
            (10, 53, 4),
            (7, 43, 5),
        ] {
            assert!(selected(
                &config,
                25,
                n_level,
                1 << n_level,
                queries,
                rate,
                true,
                true,
                true,
                false,
            ));
        }
        assert!(!selected(
            &config,
            25,
            4,
            1 << 4,
            36,
            6,
            true,
            true,
            true,
            false,
        ));
        assert!(!selected(
            &config,
            25,
            19,
            1 << 19,
            218,
            1,
            false,
            true,
            true,
            false,
        ));
        assert!(!selected(
            &config,
            25,
            19,
            1 << 19,
            218,
            1,
            true,
            false,
            true,
            false,
        ));
        assert!(!selected(
            &config,
            25,
            19,
            1 << 19,
            218,
            1,
            true,
            true,
            false,
            false,
        ));
        assert!(!selected(
            &config,
            25,
            19,
            1 << 19,
            218,
            1,
            true,
            true,
            true,
            true,
        ));
        let mut wrong_ladder = config.clone();
        wrong_ladder.queries[1] += 1;
        assert!(!selected(
            &wrong_ladder,
            25,
            19,
            1 << 19,
            218,
            1,
            true,
            true,
            true,
            false,
        ));
    }

    /// The deep-level (L2..L5) factorized introduction must reproduce the
    /// dense equality table's transcript coefficients and claimed evaluation
    /// at every ranked ladder dimension — and the identity must have teeth:
    /// corrupting one retained factor changes the result.
    #[test]
    fn deep_factorized_ood_intro_matches_dense() {
        use crate::challenger::Challenger;
        for &d in &[7usize, 10, 13, 16] {
            let mut ch = crate::challenger::RandomChallenger::new(0xDEE9_00D0 ^ d as u64);
            let z = ch.sample_f128_vec(d);
            let f: Vec<F128> = (0..(1usize << d)).map(|_| ch.sample_f128()).collect();
            let dense = build_eq_table(&z);
            let want = round_msg_and_eval_lsb(&f, &dense);
            let split = (d - 1).min(LAZY_OOD_EQ_SPLIT_LOW_LOG);
            let mut eq_lo = build_eq_table(&z[1..1 + split]);
            let eq_hi = build_eq_table(&z[1 + split..]);
            let got = round_msg_and_eval_lsb_factorized_eq(&f, &eq_lo, &eq_hi, z[0]);
            assert_eq!(got, want, "deep factorized OOD introduction d={d}");
            // Negative control: one corrupted low weight must not still match.
            eq_lo[1] += F128::ONE;
            let bad = round_msg_and_eval_lsb_factorized_eq(&f, &eq_lo, &eq_hi, z[0]);
            assert_ne!(bad, want, "corrupted low factor went undetected d={d}");
        }
    }

    /// The retained-equality fold correction must match the dense glue at
    /// every ranked deep shape — covering the serial small-round path
    /// (d = 7, 10: `eq_hi` collapses to one entry), the parallel path at its
    /// minimum size (d = 13: exactly two CHUNK slices), and the L2 shape
    /// (d = 16). Negative control: one corrupted high weight must be caught.
    #[test]
    fn deep_lazy_ood_corrected_fold_matches_dense_glue() {
        use crate::challenger::Challenger;
        for &d in &[7usize, 10, 13, 16] {
            let n = 1usize << d;
            let mut ch = crate::challenger::RandomChallenger::new(0xDEE9_F01D ^ d as u64);
            let z = ch.sample_f128_vec(d);
            let f: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let beta = ch.sample_f128();
            let r = ch.sample_f128();

            let dense_eq = build_eq_table(&z);
            let mut dense_basis = b.clone();
            crate::field::f128_slice::add_scaled(&mut dense_basis, &dense_eq, beta);
            let (want_f, want_b, want_msg) = fold_and_msg_lsb(&f, &dense_basis, r, None);

            let split = (d - 1).min(LAZY_OOD_EQ_SPLIT_LOW_LOG);
            let eq_lo = build_eq_table(&z[1..1 + split]);
            let mut eq_hi = build_eq_table(&z[1 + split..]);
            let gamma = beta * (F128::ONE + z[0] + r);
            let (got_f, got_b, got_msg) =
                fold_and_msg_lsb_inner(&f, &b, r, None, Some((&eq_lo, &eq_hi, gamma)), None);
            assert_eq!(&*got_f, &*want_f, "folded witness d={d}");
            assert_eq!(&*got_b, &*want_b, "corrected basis d={d}");
            assert_eq!(got_msg, want_msg, "next-round message d={d}");

            // Negative control: one corrupted high weight must not still match.
            eq_hi[0] += F128::ONE;
            let (_, bad_b, _) =
                fold_and_msg_lsb_inner(&f, &b, r, None, Some((&eq_lo, &eq_hi, gamma)), None);
            assert_ne!(
                &*bad_b, &*want_b,
                "corrupted high factor went undetected d={d}"
            );
        }
    }

    /// Exercise the prover state machine at the deep shapes across the
    /// operation that sits between a level's OOD glue and its consuming fold:
    /// the level's ordinary induced-basis introduce/glue. Covers both the
    /// serial (d = 10) and parallel (d = 16) fold paths.
    #[test]
    fn deep_lazy_ood_state_machine_matches_dense_across_ordinary_glue() {
        use crate::challenger::Challenger;
        for &d in &[10usize, 16] {
            let n = 1usize << d;
            let mut ch = crate::challenger::RandomChallenger::new(0xDEE9_57A7 ^ d as u64);
            let f: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let z = ch.sample_f128_vec(d);
            let ordinary: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let ordinary_claim = ch.sample_f128();
            let beta = ch.sample_f128();
            let alpha = ch.sample_f128();
            let r = ch.sample_f128();
            let target = ch.sample_f128();

            let (mut dense, dense_start) = SumcheckProver::new(f.clone(), b.clone(), target);
            let (mut lazy, lazy_start) = SumcheckProver::new(f, b, target);
            assert_eq!(lazy_start, dense_start);

            let (dense_intro, dense_y) = dense.introduce_new_with_eval(build_eq_table(&z));
            let (lazy_intro, lazy_y) = lazy.introduce_new_ood_factorized(&z).unwrap();
            assert_eq!((lazy_intro, lazy_y), (dense_intro, dense_y));
            dense.glue(beta);
            lazy.glue_factorized_ood(beta);

            let dense_ordinary = dense.introduce_new(ordinary.clone(), ordinary_claim);
            let lazy_ordinary = lazy.introduce_new(ordinary, ordinary_claim);
            assert_eq!(lazy_ordinary, dense_ordinary);
            dense.glue(alpha);
            lazy.glue(alpha);

            assert_eq!(lazy.fold(r), dense.fold(r), "d={d}");
            assert_eq!(&*lazy.f, &*dense.f, "d={d}");
            assert_eq!(&*lazy.combined_basis, &*dense.combined_basis, "d={d}");
            assert_eq!(lazy.t_r, dense.t_r, "d={d}");
            assert_eq!(lazy.transcript, dense.transcript, "d={d}");
        }
    }

    #[test]
    fn ranked_deep_lazy_ood_selector_is_exact() {
        let security = LigeritoSecurityConfig::from_toml_str(include_str!(
            "../../configs/ligerito/m32_fast.toml"
        ))
        .unwrap();
        let (config, _) = security.to_prover_verifier_configs().unwrap();
        let selected = |log_n, nl, count, len, direct, platform, disabled| {
            ranked_deep_lazy_ood_eq_selected(
                &config, log_n, nl, count, len, direct, platform, disabled,
            )
        };
        for &nl in &[16usize, 13, 10, 7] {
            assert!(selected(25, nl, 1, 1 << nl, true, true, false));
            assert!(!selected(24, nl, 1, 1 << nl, true, true, false));
            assert!(!selected(25, nl, 2, 1 << nl, true, true, false));
            assert!(!selected(25, nl, 1, 1 << (nl - 1), true, true, false));
            assert!(!selected(25, nl, 1, 1 << nl, false, true, false));
            assert!(!selected(25, nl, 1, 1 << nl, true, false, false));
            assert!(!selected(25, nl, 1, 1 << nl, true, true, true));
        }
        // Off-ladder dimensions stay dense; 19 belongs to the L1 selector.
        assert!(!selected(25, 19, 1, 1 << 19, true, true, false));
        assert!(!selected(25, 15, 1, 1 << 15, true, true, false));
        assert!(!selected(25, 4, 1, 1 << 4, true, true, false));
    }

    /// The cache-blocked transpose schedule must be BYTE-identical to the
    /// incumbent per-layer schedule for every `(log_d, top)` shape the open
    /// can reach — including `top < log_d` (the sparse-prefix tail),
    /// `top == 0`, and sizes on both sides of the split heuristic.
    #[test]
    fn transpose_blocked_matches_per_layer() {
        use crate::challenger::Challenger;
        for &log_d in &[0usize, 1, 2, 5, 8, 12, 16, 18, 20] {
            let ntt = AdditiveNttF128::standard(log_d.max(1));
            for top in 0..=log_d {
                let n = 1usize << log_d;
                let mut ch =
                    crate::challenger::RandomChallenger::new(0x7A5E ^ ((log_d * 37 + top) as u64));
                let base: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
                let mut a = base.clone();
                let mut b = base;
                transpose_forward_ntt_dense_layers_per_layer(&ntt, &mut a, top);
                transpose_forward_ntt_dense_layers_blocked(&ntt, &mut b, top);
                assert_eq!(a, b, "log_d={log_d}, top={top}");
            }
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

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn direct_fold8_four_map_gfni_matches_fold_one_slot_sum() {
        use crate::zerocheck::multilinear::kernels::x86_64::{
            build_row_fold_mats, gfni_fold64_four_maps_staged,
        };
        let mut state = 0xB6F1_6408_5EED_191Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let tables: Vec<Vec<F128>> = (0..4)
            .map(|_| {
                let generators: Vec<F128> = (0..128)
                    .map(|_| F128 {
                        lo: next(),
                        hi: next(),
                    })
                    .collect();
                super::super::ring_switch::build_fold_byte_table(&generators)
            })
            .collect();
        let values: Vec<Vec<u64>> = (0..4).map(|_| (0..64).map(|_| next()).collect()).collect();
        let mats: Vec<[u64; 128]> = tables
            .iter()
            .map(|table| build_row_fold_mats(&table[..8 * 256]))
            .collect();
        let mut got = vec![F128::ZERO; 64];
        let mut planes = vec![F128::ZERO; 64];
        // SAFETY: four exact 512-byte inputs, complete matrices, 64 outputs,
        // sixteen-ZMM scratch, and cfg-guaranteed target features.
        unsafe {
            gfni_fold64_four_maps_staged(
                values[0].as_ptr().cast::<u8>(),
                &mats[0],
                values[1].as_ptr().cast::<u8>(),
                &mats[1],
                values[2].as_ptr().cast::<u8>(),
                &mats[2],
                values[3].as_ptr().cast::<u8>(),
                &mats[3],
                got.as_mut_ptr(),
                planes.as_mut_ptr().cast(),
            );
        }
        for i in 0..64 {
            let expect = (0..4).fold(F128::ZERO, |acc, map| {
                acc + super::super::ring_switch::fold_one_slot(
                    F128::new(values[map][i], 0),
                    &tables[map],
                )
            });
            assert_eq!(got[i], expect);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn direct_fold4_two_claim_gfni_matches_composed_table_oracle() {
        use crate::zerocheck::multilinear::kernels::x86_64::{
            build_row_fold_mats_from_cols, gfni_fold64_four_maps_staged,
        };
        use core::arch::x86_64::_mm512_setzero_si512;

        let mut state = 0xD1CE_F004_5EED_191Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let direct_tables: Vec<Vec<F128>> = (0..2)
            .map(|_| {
                let generators: Vec<F128> = (0..128).map(|_| F128::new(next(), next())).collect();
                super::super::ring_switch::build_fold_byte_table(&generators)
            })
            .collect();
        let eq_hi = [F128::new(next(), next()), F128::new(next(), next())];
        let rows: Vec<(Vec<u64>, Vec<u64>)> = (0..2)
            .map(|_| {
                (
                    (0..64).map(|_| next()).collect(),
                    (0..64).map(|_| next()).collect(),
                )
            })
            .collect();
        let cols0 = super::super::ring_switch::compose_block_cols(&direct_tables[0], eq_hi[0]);
        let cols1 = super::super::ring_switch::compose_block_cols(&direct_tables[1], eq_hi[1]);
        let mats0_lo = build_row_fold_mats_from_cols(&cols0[..64]);
        let mats0_hi = build_row_fold_mats_from_cols(&cols0[64..]);
        let mats1_lo = build_row_fold_mats_from_cols(&cols1[..64]);
        let mats1_hi = build_row_fold_mats_from_cols(&cols1[64..]);
        let mut got = vec![F128::ZERO; 64];
        let mut planes = unsafe { [_mm512_setzero_si512(); 16] };
        // SAFETY: four exact 512-byte inputs, four complete composed maps,
        // 64 outputs, sixteen-ZMM scratch, and cfg-guaranteed features.
        unsafe {
            gfni_fold64_four_maps_staged(
                rows[0].0.as_ptr().cast::<u8>(),
                &mats0_lo,
                rows[0].1.as_ptr().cast::<u8>(),
                &mats0_hi,
                rows[1].0.as_ptr().cast::<u8>(),
                &mats1_lo,
                rows[1].1.as_ptr().cast::<u8>(),
                &mats1_hi,
                got.as_mut_ptr(),
                planes.as_mut_ptr(),
            );
        }
        let mut composed0 = vec![F128::ZERO; super::super::ring_switch::FOLD_TABLE_TOTAL];
        let mut composed1 = vec![F128::ZERO; super::super::ring_switch::FOLD_TABLE_TOTAL];
        super::super::ring_switch::compose_block_table(&direct_tables[0], eq_hi[0], &mut composed0);
        super::super::ring_switch::compose_block_table(&direct_tables[1], eq_hi[1], &mut composed1);
        for slot in 0..64 {
            let expect = super::super::ring_switch::fold_one_slot(
                F128::new(rows[0].0[slot], rows[0].1[slot]),
                &composed0,
            ) + super::super::ring_switch::fold_one_slot(
                F128::new(rows[1].0[slot], rows[1].1[slot]),
                &composed1,
            );
            assert_eq!(got[slot], expect, "slot {slot}");
        }
    }

    #[test]
    fn direct_fold4_gfni_gate_is_ranked_shape_only() {
        assert!(direct_fold4_b_gfni_shape(2, false, 64));
        assert!(direct_fold4_b_gfni_shape(2, false, 8192));
        assert!(!direct_fold4_b_gfni_shape(1, false, 64));
        assert!(!direct_fold4_b_gfni_shape(3, false, 64));
        assert!(!direct_fold4_b_gfni_shape(2, true, 64));
        assert!(!direct_fold4_b_gfni_shape(2, false, 60));
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

        let n_packed = 1usize << crate::pcs::LOG_PACKING;
        let tail_eq = build_eq_table(&suffix[6..]);
        let table = super::super::ring_switch::build_fold_byte_table(&scaled_rdp);
        let low_eq = build_eq_table(&suffix[..6]);
        let mut a_state = vec![F128::ZERO; 64 * n_packed];
        for e in 0..64 {
            let strided: Vec<F128> = (0..poly.len() / 64)
                .map(|rest| poly[64 * rest + e])
                .collect();
            let bank = super::super::ring_switch::fold_1b_rows_naive(&strided, &tail_eq);
            let transposed = super::super::ring_switch::tensor_algebra_transpose(&bank);
            for (bit, value) in transposed.into_iter().enumerate() {
                a_state[bit * 64 + e] = value;
            }
        }
        let mut w_state = vec![F128::ZERO; 64 * n_packed];
        for d_low in 0..64 {
            let mut basis_product = low_eq[d_low];
            w_state[d_low] = super::super::ring_switch::fold_one_slot(basis_product, &table);
            for bit in 1..n_packed {
                basis_product = crate::field::mul_by_x(basis_product);
                w_state[bit * 64 + d_low] =
                    super::super::ring_switch::fold_one_slot(basis_product, &table);
            }
        }
        let mut round0 = (F128::ZERO, F128::ZERO);
        for i in (0..a_state.len()).step_by(2) {
            let a0 = a_state[i];
            let a1 = a_state[i + 1];
            let b0 = w_state[i];
            let b1 = w_state[i + 1];
            round0.0 += a0 * b0;
            round0.1 += (a0 + a1) * (b0 + b1);
        }
        let (eq_lo, eq_hi) =
            super::super::ring_switch::build_eq_split(&suffix[6..], (log_n - 6) / 2);
        let direct = vec![super::super::ring_switch::DirectFold8Factors {
            eq_lo,
            eq_hi,
            a_state,
            w_state,
            round0,
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
            crate::challenger::FsChallenger::new(b"direct-fold8-proof-byte-oracle");
        let ordinary = recursive_prover_with_basis_precomputed_round0(
            &cfg,
            poly.clone(),
            combined_basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            None,
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
            None,
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

    /// Cross-process transcript oracle for the initial-sumcheck fold kernels
    /// (`#[ignore]`: 64 MB poly; run explicitly). m=29 (log_n=22) is the
    /// smallest shape whose round-0 fold half (2^21) engages the NT/SoA leaf.
    /// Prints a deterministic digest of the full proof — run once with
    /// default env and once with `FLOCK_NO_OPEN_SUMCHECK_OPT=1` (and/or
    /// `FLOCK_NO_OPEN_NT=1`) and diff the `PROOF_DIGEST` lines: identical
    /// digests = identical proof bytes = identical transcript.
    #[test]
    #[ignore]
    fn open_sumcheck_kernel_e2e_transcript_digest() {
        use crate::challenger::Challenger;
        use std::hash::{Hash as _, Hasher as _};
        let m = 29usize;
        let log_n = m - crate::pcs::LOG_PACKING;
        let initial_k = 6;
        let mut p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m29 fast prover config");
        let mut v_cfg = verifier_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m29 fast verifier config");
        // Mirror the ranked worker: BLAKE3 Merkle + BLAKE3 FS.
        p_cfg.merkle_hash = HashKind::Blake3;
        v_cfg.merkle_hash = HashKind::Blake3;

        let mut rng = crate::challenger::RandomChallenger::new(0x50A0_0AC1E_u64);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rate_0 = p_cfg.log_inv_rates[0];
        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate_0);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate_0,
            &ntt_0,
            HashKind::Blake3,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::with_hash(b"soa-oracle", HashKind::Blake3);
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly,
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        // Deterministic digest of the whole proof (Debug string through the
        // fixed-key DefaultHasher — stable across processes of one binary).
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{proof:?}").hash(&mut h);
        eprintln!("PROOF_DIGEST {:016x}", h.finish());

        let mut v_ch = crate::challenger::FsChallenger::with_hash(b"soa-oracle", HashKind::Blake3);
        assert!(
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch),
            "m29 blake3 proof must verify"
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

    /// The GPU recursive-commit Merkle route must produce the exact
    /// node-for-node tree the CPU `merkle_tree` builds, at the production
    /// shapes: 2^18 leaves × 128 B (the L1 tree at the ranked m=32 open,
    /// GPU-routed by default) and 2^16 × 128 B (the L2 shape the floor can
    /// be lowered to via `FLOCK_GPU_OPEN_MERKLE_MIN_LOG2`). Also covers the
    /// CPU top above [`GPU_OPEN_STOP_NODES`]. SKIPS without Metal.
    #[test]
    fn gpu_open_merkle_tree_matches_cpu() {
        if !crate::gpu::merkle::available() {
            eprintln!("SKIP gpu_open_merkle_tree_matches_cpu: Metal unavailable");
            return;
        }
        for log_leaves in [16usize, 18] {
            let n_leaves = 1usize << log_leaves;
            let leaf_size = 128usize;
            let data: Vec<F128> = (0..n_leaves * leaf_size / 16)
                .map(|i| F128::new(i as u64, (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
                .collect();
            let bytes: &[u8] =
                unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 16) };
            let mut busy_ms = 0.0f64;
            let gpu_tree = gpu_merkle_tree_for_open(bytes, n_leaves, leaf_size, &mut busy_ms)
                .expect("GPU open-merkle session must complete when Metal is available");
            let cpu_tree = merkle::merkle_tree(bytes, n_leaves, HashKind::Blake3);
            assert_eq!(
                gpu_tree, cpu_tree,
                "GPU open-merkle tree != CPU at 2^{log_leaves} leaves"
            );
        }
    }
    /// The peeled block-zero leaf is exactly the established general
    /// butterfly at twiddle `(layer, 0)`, for every pass-B stride and for
    /// vector/tail tile lengths. The explicit selector checks the production
    /// and kill-switch decisions without process-global environment races.
    #[test]
    fn transpose_pass_b_zero_leaf_matches_general() {
        use crate::challenger::Challenger;
        assert!(tnt_pass_b_zero_selected(false));
        assert!(!tnt_pass_b_zero_selected(true));
        let ntt = AdditiveNttF128::standard(12);
        let mut ch = crate::challenger::RandomChallenger::new(0xB10C_0000_7E80);
        for &(stride, layer) in &[(1usize, 3usize), (2, 2), (4, 1), (8, 0)] {
            assert_eq!(ntt.twiddle(layer, 0), F128::ZERO);
            for &tile in &[1usize, 3, 4, 7, 16, 65] {
                let base: Vec<Vec<F128>> = (0..2 * stride)
                    .map(|_| (0..tile).map(|_| ch.sample_f128()).collect())
                    .collect();
                let mut got = base.clone();
                let mut want = base.clone();
                {
                    let mut cols: Vec<&mut [F128]> =
                        got.iter_mut().map(Vec::as_mut_slice).collect();
                    transpose_pass_b_block_zero_xor(&mut cols, stride, layer, &ntt);
                }
                {
                    let mut cols: Vec<&mut [F128]> =
                        want.iter_mut().map(Vec::as_mut_slice).collect();
                    transpose_pass_b_block_zero_general(&mut cols, stride, layer, &ntt);
                }
                assert_eq!(got, want, "stride={stride} tile={tile}");
                for u in 0..stride {
                    for i in 0..tile {
                        assert_eq!(got[u][i], base[u][i] + base[u + stride][i]);
                        assert_eq!(got[u + stride][i], base[u + stride][i]);
                    }
                }
            }
        }
    }
}
