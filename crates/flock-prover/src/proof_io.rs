//! Serialize / deserialize proofs to bytes (and files).
//!
//! Two bundle types: [`R1csProofBundleLigerito`] for the base R1CS proof and
//! [`ChainProofBundleLigerito`] for the hash-chain proof. Both pair a proof
//! with its commitment (which the verifier needs); the chain bundle
//! additionally carries the public endpoint bits.
//!
//! On-disk format:
//! ```text
//!   bytes 0..5    "FLOCK"                  (5-byte magic)
//!   byte  5       VERSION                  (currently 1)
//!   bytes 6..7    flavor: 2 = R1cs, 3 = Chain (0/1 reserved: legacy BaseFold)
//!   bytes 7..     bincode-serialized payload
//! ```
//!
//! Versioning is here to make schema changes detectable cleanly: bump
//! `VERSION` whenever a payload field is added/removed/reordered. Forward
//! compatibility is NOT promised — `from_bytes` of a different version is
//! rejected (`UnsupportedVersion`).
//!
//! ## Round-trip example
//! ```ignore
//! let bundle = R1csProofBundleLigerito { commitment, proof };
//! let bytes = bundle.to_bytes();
//! std::fs::write("proof.bin", &bytes)?;
//! ...
//! let bytes = std::fs::read("proof.bin")?;
//! let bundle = R1csProofBundleLigerito::from_bytes(&bytes)?;
//! // Then call e.g. `setup.verify(&bundle.commitment, &bundle.proof, ...)`.
//! ```

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use flock_core::pcs::Commitment;

/// Magic bytes prepended to every serialized proof. Lets readers reject
/// random binary data early.
pub const MAGIC: [u8; 5] = *b"FLOCK";

/// Format version. Bumped on incompatible serialization changes.
/// v4 (current) adds `ood_values` + `fold_grinding_nonces` to
/// `LigeritoProof` and `profile` to `PcsParams` (Johnson+OOD profiles).
/// v3 restructures `BaseFoldProof`: per-query Merkle paths are replaced by
/// shared octopus multi-proofs (one per Merkle tree). v2 added `HashKind`
/// to [`ChainProofBundle`].
pub const VERSION: u8 = 4;

/// Which hash function a chain proof is over. Carried in
/// [`ChainProofBundle`] so the verifier (e.g. the CLI) can pick the right
/// `*_chain` setup without out-of-band info.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashKind {
    Blake3,
    Sha2,
    Keccak,
}

impl HashKind {
    /// Parse a CLI-style name; case-insensitive. Accepts `blake3`, `sha2` /
    /// `sha256`, `keccak` / `keccak_f`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "blake3" => Some(Self::Blake3),
            "sha2" | "sha256" | "sha-2" | "sha-256" => Some(Self::Sha2),
            "keccak" | "keccak_f" | "keccak-f" => Some(Self::Keccak),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Blake3 => "blake3",
            Self::Sha2 => "sha2",
            Self::Keccak => "keccak",
        }
    }
}

/// Flavor discriminator (1 byte). Lets a generic reader peek what kind of
/// bundle a file holds without parsing the payload first. Values 0/1 are
/// reserved: they were the legacy BaseFold R1cs/Chain flavors.
const FLAVOR_R1CS_LIGERITO: u8 = 2;
const FLAVOR_CHAIN_LIGERITO: u8 = 3;

/// Header size = 5-byte magic + 1-byte version + 1-byte flavor.
const HEADER_LEN: usize = 7;

/// Errors from `from_bytes` / `read_from_file`.
#[derive(Debug)]
pub enum DeserializeError {
    /// The 5-byte magic prefix did not match `FLOCK`.
    BadMagic,
    /// The version byte didn't match this build's `VERSION`. The number is
    /// the version found in the file.
    UnsupportedVersion(u8),
    /// The flavor byte was neither `2` (R1cs Ligerito) nor `3` (Chain Ligerito).
    UnknownFlavor(u8),
    /// `from_bytes` was called with a slice shorter than `HEADER_LEN`.
    Truncated,
    /// The expected flavor and the file's flavor disagree (e.g. trying to
    /// load a `ChainProofBundle` from an R1CS bundle file).
    FlavorMismatch { expected: u8, found: u8 },
    /// The bincode-deserialization step failed (corrupted payload, etc.).
    Bincode(bincode::Error),
}

impl std::fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad magic: not a FLOCK proof file"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported version {v} (this build expects {VERSION})")
            }
            Self::UnknownFlavor(v) => write!(f, "unknown flavor byte: {v}"),
            Self::Truncated => write!(f, "input shorter than header ({HEADER_LEN} bytes)"),
            Self::FlavorMismatch { expected, found } => {
                write!(f, "flavor mismatch: expected {expected}, found {found}")
            }
            Self::Bincode(e) => write!(f, "bincode error: {e}"),
        }
    }
}

impl std::error::Error for DeserializeError {}

impl From<bincode::Error> for DeserializeError {
    fn from(e: bincode::Error) -> Self {
        Self::Bincode(e)
    }
}

/// Bundles a base R1CS proof with its commitment for self-contained
/// serialization. Verification still needs the relevant [`flock_core::r1cs::BlockR1cs`]
/// (or a `*Setup`) on the verifier side — that's a public artifact derived
/// from the setup parameters, not part of the proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofBundleLigerito {
    pub commitment: Commitment,
    pub proof: flock_core::proof::R1csProofLigerito,
}

/// Bundles a hash-chain proof with its commitment + public endpoint bits
/// (`cv_0_phys` and `cv_last_phys` are the physical within-slot bool layouts
/// returned by per-hash `*_to_phys_bits` helpers — `region_bits` long each)
/// plus the [`HashKind`] discriminator so a verifier can pick the right
/// per-hash setup from the bundle alone.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainProofBundleLigerito {
    pub hash_kind: HashKind,
    pub commitment: Commitment,
    pub proof: crate::r1cs_hashes::chain_common::ChainProofLigerito,
    pub cv_0_phys: Vec<bool>,
    pub cv_last_phys: Vec<bool>,
}

impl R1csProofBundleLigerito {
    pub fn to_bytes(&self) -> Vec<u8> {
        // Fast path: the prover pre-encoded `header ‖ commitment ‖ zerocheck
        // ‖ lincheck` concurrently with the PCS open (those sections are
        // transcript-final before the open starts — see
        // [`stash_pre_encoded_prefix`]); only `pcs_open` remains. bincode
        // struct encoding is untagged field concatenation, so appending the
        // last field reproduces the single-shot encode byte-for-byte; the
        // fingerprint gate below guarantees the stash is for THIS bundle.
        if let Some(mut out) = take_matching_pre_encoded(self) {
            encode_pcs_open_into(&mut out, &self.proof.pcs_open);
            return out;
        }
        // Ranked m=32 proofs serialize to ~437-440 KB; reserving up front
        // avoids ~19 doubling reallocs (~0.9 MB of copies) inside the timed
        // window. Larger proofs still grow normally.
        let mut out = Vec::with_capacity(HEADER_LEN + 460_000);
        write_header(&mut out, FLAVOR_R1CS_LIGERITO);
        if fast_pcs_open_encode_enabled() {
            // Same untagged-field-concatenation identity as the fast path
            // above: `enc(bundle) = enc(commitment) ‖ enc(zerocheck) ‖
            // enc(lincheck) ‖ enc(pcs_open)`, so encoding the first three
            // sections with bincode and the dominant `pcs_open` (~99% of the
            // bytes) with the flat encoder reproduces the single-shot encode
            // byte-for-byte.
            bincode::serialize_into(&mut out, &self.commitment)
                .expect("bincode serialize Commitment");
            bincode::serialize_into(&mut out, &self.proof.zerocheck)
                .expect("bincode serialize ZerocheckProof");
            bincode::serialize_into(&mut out, &self.proof.lincheck)
                .expect("bincode serialize LincheckProof");
            encode_pcs_open_into(&mut out, &self.proof.pcs_open);
        } else {
            bincode::serialize_into(&mut out, self)
                .expect("bincode serialize R1csProofBundleLigerito");
        }
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_R1CS_LIGERITO)?;
        Ok(bincode::deserialize(payload)?)
    }
}

// ---------------------------------------------------------------------------
// Publish-prefix pre-encode
// ---------------------------------------------------------------------------

/// A publish prefix — `header ‖ enc(commitment) ‖ enc(zerocheck) ‖
/// enc(lincheck)` — encoded while the PCS open runs. Those three sections are
/// transcript-committed before the open starts and the open only produces
/// `pcs_open`, so their serialization can leave the publish tail (which sits
/// between `prove_fast` returning and the proof file becoming visible to the
/// harness — all inside the scored interval). The prefix is small (~4.3 kB of
/// a ~437 kB ranked bundle); the tail win is that encode plus the 460 kB
/// output allocation, which the stash performs off-tail. `pcs_open` (~99% of
/// the bytes) exists only after the open and can never leave the tail.
struct PreEncodedPrefix {
    /// The commitment's Merkle root. Two distinct proves sharing a root would
    /// be a hash collision, so a root match pins the stash to this prove.
    root: flock_core::merkle::Hash,
    /// bincode-fixint encoded lengths of commitment / zerocheck / lincheck,
    /// re-derived from the candidate bundle at publish via `serialized_size`
    /// (same default fixint config as `serialize_into`).
    sec_lens: [u64; 3],
    /// `HEADER_LEN + sec_lens` bytes, with capacity already covering the full
    /// bundle so the publish-tail `pcs_open` append never reallocates.
    bytes: Vec<u8>,
}

/// Latest stashed prefix. Process-global and single-slot: the ranked worker
/// never overlaps proves, so the slot always holds the publishing prove's
/// prefix. Any staleness — foreign stash, concurrent test proves — is caught
/// by the fingerprint and falls back to the full encode.
static PRE_ENCODED: std::sync::Mutex<Option<PreEncodedPrefix>> = std::sync::Mutex::new(None);

/// `FLOCK_NO_PRE_ENCODE=1` restores the incumbent single-shot bundle encode.
/// The ranked harness `env_clear()`s, so pre-encode is the default.
pub(crate) fn pre_encode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FLOCK_NO_PRE_ENCODE").map_or(true, |v| v != "1"))
}

/// Encode and stash the publish prefix for a prove whose commitment /
/// zerocheck / lincheck are final. Called by the prover on a detached helper
/// thread concurrently with the PCS open (tens of µs of alloc + encode
/// against a ~20 ms open). Replaces any previous stash.
pub fn stash_pre_encoded_prefix(
    commitment: &Commitment,
    zerocheck: &flock_core::zerocheck::ZerocheckProof,
    lincheck: &flock_core::lincheck::LincheckProof,
) {
    if !pre_encode_enabled() {
        return;
    }
    // Same capacity as the incumbent `to_bytes`: the publish-tail `pcs_open`
    // append extends this exact Vec, so it must never need to grow.
    let mut bytes = Vec::with_capacity(HEADER_LEN + 460_000);
    write_header(&mut bytes, FLAVOR_R1CS_LIGERITO);
    let mut sec_lens = [0u64; 3];
    let mut mark = bytes.len();
    bincode::serialize_into(&mut bytes, commitment).expect("bincode serialize Commitment");
    sec_lens[0] = (bytes.len() - mark) as u64;
    mark = bytes.len();
    bincode::serialize_into(&mut bytes, zerocheck).expect("bincode serialize ZerocheckProof");
    sec_lens[1] = (bytes.len() - mark) as u64;
    mark = bytes.len();
    bincode::serialize_into(&mut bytes, lincheck).expect("bincode serialize LincheckProof");
    sec_lens[2] = (bytes.len() - mark) as u64;
    // Assignment-only critical section — nothing that can panic runs while
    // the guard is held, so the lock cannot poison. A poisoned lock
    // (unreachable in practice) skips the stash; publish then full-encodes.
    if let Ok(mut slot) = PRE_ENCODED.lock() {
        *slot = Some(PreEncodedPrefix {
            root: commitment.root,
            sec_lens,
            bytes,
        });
    }
}

/// Take the stash iff it fingerprint-matches `bundle`: identical Merkle root
/// AND identical bincode-fixint size for each of the three prefix sections.
/// bincode-fixint encoding of equal values is deterministic, so a full match
/// means the stashed bytes equal what a fresh encode of `bundle`'s prefix
/// would produce. `None` (full-encode fallback) on any doubt — no stash,
/// disabled switch, poisoned lock, any fingerprint miss. The stash is
/// consumed on take; a repeat publish of the same bundle re-encodes from
/// scratch, byte-identically.
fn take_matching_pre_encoded(bundle: &R1csProofBundleLigerito) -> Option<Vec<u8>> {
    if !pre_encode_enabled() {
        return None;
    }
    let stash = PRE_ENCODED.lock().ok()?.take()?;
    if stash.root != bundle.commitment.root {
        return None;
    }
    let sec_lens = [
        bincode::serialized_size(&bundle.commitment).ok()?,
        bincode::serialized_size(&bundle.proof.zerocheck).ok()?,
        bincode::serialized_size(&bundle.proof.lincheck).ok()?,
    ];
    if sec_lens != stash.sec_lens {
        return None;
    }
    let want = HEADER_LEN + sec_lens.iter().sum::<u64>() as usize;
    (stash.bytes.len() == want).then_some(stash.bytes)
}

// ---------------------------------------------------------------------------
// Flat bincode-compatible `pcs_open` encoder
// ---------------------------------------------------------------------------
//
// `bincode::serialize_into` (fixint, little-endian — the bincode 1.x
// free-function config) drives the ~433 kB `pcs_open` section through serde
// element-at-a-time: every F128 costs two `write_u64` calls and every Merkle
// hash 32 single-byte writes, all inside the measured publish tail. The
// encoder below emits the *identical bytes* with bulk slice copies instead:
//
//   - struct        → field concatenation in declaration order (no framing)
//   - Vec<T>        → u64 LE length ‖ elements
//   - u64           → 8 LE bytes
//   - F128 {lo,hi}  → lo LE ‖ hi LE = its in-memory bytes on a little-endian
//                     target (`repr(C, align(16))`, two u64s, no padding) —
//                     whole slices are memcpy'd
//   - Hash [u8;32]  → 32 raw bytes (serde fixed arrays carry no length)
//
// Every struct is destructured exhaustively (no `..`), so adding/removing/
// reordering a field breaks compilation here instead of silently changing
// the encoding; `flat_pcs_open_encoder_matches_bincode` byte-checks the
// result against `bincode::serialize` on nonuniform random proofs.
//
// `FLOCK_NO_FAST_ENCODE=1` restores the incumbent serde encode; big-endian
// targets always fall back.

use flock_core::field::F128;
use flock_core::merkle::Hash as MerkleHash;
use flock_core::pcs::BatchOpeningProofLigerito;
use flock_core::pcs::ligerito::{FinalProof, LigeritoProof, RecursiveProof, SumcheckMessage};
use flock_core::pcs::ring_switch::RingSwitchProof;

fn fast_pcs_open_encode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(target_endian = "little")
        && *ON.get_or_init(|| std::env::var("FLOCK_NO_FAST_ENCODE").map_or(true, |v| v != "1"))
}

/// Append the bincode-fixint encoding of `p` to `out`. Byte-identical to
/// `bincode::serialize_into(out, p)` (falls back to exactly that when the
/// fast path is disabled).
fn encode_pcs_open_into(out: &mut Vec<u8>, p: &BatchOpeningProofLigerito) {
    if !fast_pcs_open_encode_enabled() {
        bincode::serialize_into(&mut *out, p).expect("bincode serialize pcs_open");
        return;
    }
    let BatchOpeningProofLigerito {
        ring_switches,
        ligerito,
    } = p;
    put_u64(out, ring_switches.len() as u64);
    for rs in ring_switches {
        let RingSwitchProof { s_hat_v } = rs;
        put_f128_vec(out, s_hat_v);
    }
    let LigeritoProof {
        initial_root,
        initial_proof,
        recursive_roots,
        recursive_proofs,
        final_proof,
        sumcheck_transcript,
        grinding_nonces,
        ood_values,
        fold_grinding_nonces,
    } = ligerito;
    out.extend_from_slice(initial_root);
    put_recursive_proof(out, initial_proof);
    put_hash_vec(out, recursive_roots);
    put_u64(out, recursive_proofs.len() as u64);
    for rp in recursive_proofs {
        put_recursive_proof(out, rp);
    }
    let FinalProof {
        yr,
        opened_rows,
        merkle_proof,
    } = final_proof;
    put_f128_vec(out, yr);
    put_rows(out, opened_rows);
    put_hash_vec(out, merkle_proof);
    put_u64(out, sumcheck_transcript.len() as u64);
    for m in sumcheck_transcript {
        let SumcheckMessage { u_0, u_2 } = m;
        put_f128(out, *u_0);
        put_f128(out, *u_2);
    }
    put_u64_vec(out, grinding_nonces);
    put_f128_vec(out, ood_values);
    put_u64_vec(out, fold_grinding_nonces);
}

fn put_recursive_proof(out: &mut Vec<u8>, rp: &RecursiveProof) {
    let RecursiveProof {
        opened_rows,
        merkle_proof,
    } = rp;
    put_rows(out, opened_rows);
    put_hash_vec(out, merkle_proof);
}

#[inline]
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn put_f128(out: &mut Vec<u8>, v: F128) {
    put_u64(out, v.lo);
    put_u64(out, v.hi);
}

#[inline]
fn put_f128_vec(out: &mut Vec<u8>, v: &[F128]) {
    put_u64(out, v.len() as u64);
    // SAFETY: `F128` is `repr(C, align(16))` with exactly two `u64` fields
    // (size 16, no padding), so on a little-endian target the in-memory bytes
    // of a slice are precisely the bincode-fixint encoding of its elements.
    // The fast path is compile-time gated to little-endian above.
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    out.extend_from_slice(bytes);
}

#[inline]
fn put_hash_vec(out: &mut Vec<u8>, v: &[MerkleHash]) {
    put_u64(out, v.len() as u64);
    out.extend_from_slice(v.as_flattened());
}

#[inline]
fn put_u64_vec(out: &mut Vec<u8>, v: &[u64]) {
    put_u64(out, v.len() as u64);
    for &x in v {
        put_u64(out, x);
    }
}

fn put_rows(out: &mut Vec<u8>, rows: &[Vec<F128>]) {
    put_u64(out, rows.len() as u64);
    for row in rows {
        put_f128_vec(out, row);
    }
}

impl ChainProofBundleLigerito {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_CHAIN_LIGERITO);
        bincode::serialize_into(&mut out, self)
            .expect("bincode serialize ChainProofBundleLigerito");
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_CHAIN_LIGERITO)?;
        Ok(bincode::deserialize(payload)?)
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

fn write_header(out: &mut Vec<u8>, flavor: u8) {
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(flavor);
}

fn parse_header(bytes: &[u8], expected_flavor: u8) -> Result<&[u8], DeserializeError> {
    if bytes.len() < HEADER_LEN {
        return Err(DeserializeError::Truncated);
    }
    if bytes[0..5] != MAGIC {
        return Err(DeserializeError::BadMagic);
    }
    let v = bytes[5];
    if v != VERSION {
        return Err(DeserializeError::UnsupportedVersion(v));
    }
    let flavor = bytes[6];
    if flavor != FLAVOR_R1CS_LIGERITO && flavor != FLAVOR_CHAIN_LIGERITO {
        return Err(DeserializeError::UnknownFlavor(flavor));
    }
    if flavor != expected_flavor {
        return Err(DeserializeError::FlavorMismatch {
            expected: expected_flavor,
            found: flavor,
        });
    }
    Ok(&bytes[HEADER_LEN..])
}

// ---------------------------------------------------------------------------
// File-IO conveniences
// ---------------------------------------------------------------------------

/// Atomically write `bytes` to `path` (write-then-rename via the
/// stdlib — best-effort; on error the rename may leave a temp file behind).
pub fn write_bytes_to_file<P: AsRef<Path>>(path: P, bytes: &[u8]) -> io::Result<()> {
    let path = path.as_ref();
    let tmp = match path.parent() {
        Some(dir) => dir.join(format!(
            ".{}.tmp",
            path.file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("flock-proof")
        )),
        None => Path::new(".flock-proof.tmp").to_path_buf(),
    };
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Read raw bytes from a file. Thin wrapper over `std::fs::read`.
pub fn read_bytes_from_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Write a Ligerito chain bundle to `path`.
pub fn write_chain_bundle_ligerito_to_file<P: AsRef<Path>>(
    path: P,
    bundle: &ChainProofBundleLigerito,
) -> io::Result<()> {
    write_bytes_to_file(path, &bundle.to_bytes())
}

/// Read a Ligerito chain bundle from `path`.
pub fn read_chain_bundle_ligerito_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<ChainProofBundleLigerito, BundleReadError> {
    let bytes = read_bytes_from_file(path).map_err(BundleReadError::Io)?;
    ChainProofBundleLigerito::from_bytes(&bytes).map_err(BundleReadError::Deserialize)
}

/// Combined error returned by file-read helpers: either IO failed or the
/// bytes weren't a valid bundle.
#[derive(Debug)]
pub enum BundleReadError {
    Io(io::Error),
    Deserialize(DeserializeError),
}

impl std::fmt::Display for BundleReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Deserialize(e) => write!(f, "deserialize error: {e}"),
        }
    }
}

impl std::error::Error for BundleReadError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::blake3::{Blake3Setup, Compression, blake3_compress, cv_to_phys_bits};
    use flock_core::challenger::FsChallenger;

    /// SplitMix64.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn nx(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// The flat `pcs_open` encoder emits the identical bytes to
    /// `bincode::serialize` on nonuniform synthetic proofs (empty, small,
    /// and ragged shapes), including edge-value F128s.
    #[test]
    fn flat_pcs_open_encoder_matches_bincode() {
        let f128 = |rng: &mut Rng| F128::new(rng.nx(), rng.nx());
        let f128_vec = |rng: &mut Rng, n: usize| -> Vec<F128> {
            (0..n).map(|_| F128::new(rng.nx(), rng.nx())).collect()
        };
        let hash_vec = |rng: &mut Rng, n: usize| -> Vec<MerkleHash> {
            (0..n)
                .map(|_| std::array::from_fn(|_| rng.nx() as u8))
                .collect()
        };
        let rows = |rng: &mut Rng, n: usize, w: usize| -> Vec<Vec<F128>> {
            (0..n)
                .map(|i| {
                    (0..(w + i % 3))
                        .map(|_| F128::new(rng.nx(), rng.nx()))
                        .collect()
                })
                .collect()
        };

        let mut rng = Rng::new(0x515E_D0F1);
        for (n_rs, n_rec, n_rows, n_msgs) in [
            (0usize, 0usize, 0usize, 0usize),
            (1, 1, 3, 2),
            (3, 4, 17, 55),
        ] {
            let recursive_proof = |rng: &mut Rng| RecursiveProof {
                opened_rows: rows(rng, n_rows, 4),
                merkle_proof: hash_vec(rng, n_rows * 2 + 1),
            };
            let proof = BatchOpeningProofLigerito {
                ring_switches: (0..n_rs)
                    .map(|i| RingSwitchProof {
                        s_hat_v: f128_vec(&mut rng, i * 7),
                    })
                    .collect(),
                ligerito: LigeritoProof {
                    initial_root: std::array::from_fn(|_| rng.nx() as u8),
                    initial_proof: recursive_proof(&mut rng),
                    recursive_roots: hash_vec(&mut rng, n_rec),
                    recursive_proofs: (0..n_rec).map(|_| recursive_proof(&mut rng)).collect(),
                    final_proof: FinalProof {
                        yr: f128_vec(&mut rng, n_msgs * 3),
                        opened_rows: rows(&mut rng, n_rows, 2),
                        merkle_proof: hash_vec(&mut rng, n_rows),
                    },
                    sumcheck_transcript: (0..n_msgs)
                        .map(|_| SumcheckMessage {
                            u_0: f128(&mut rng),
                            u_2: f128(&mut rng),
                        })
                        .collect(),
                    grinding_nonces: (0..n_rec).map(|_| rng.nx()).collect(),
                    ood_values: {
                        let mut v = f128_vec(&mut rng, n_rec);
                        if let Some(first) = v.first_mut() {
                            *first = F128::ZERO;
                        }
                        if let Some(last) = v.last_mut() {
                            *last = F128::new(u64::MAX, u64::MAX);
                        }
                        v
                    },
                    fold_grinding_nonces: (0..n_msgs).map(|_| rng.nx()).collect(),
                },
            };
            let incumbent = bincode::serialize(&proof).expect("bincode serialize");
            let mut flat = Vec::new();
            encode_pcs_open_into(&mut flat, &proof);
            assert_eq!(
                flat, incumbent,
                "flat encoder diverged at shape (n_rs={n_rs}, n_rec={n_rec}, n_rows={n_rows}, n_msgs={n_msgs})"
            );
        }
    }

    /// Build a small honest BLAKE3 chain (n=8) for the bundle tests.
    fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
        let mut rng = Rng::new(seed);
        let mut cv: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
        let cv0 = cv;
        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            let m: [u32; 16] = std::array::from_fn(|_| rng.nx() as u32);
            let counter = 0u64;
            let block_len = 64u32;
            let flags = 0u32;
            blocks.push((cv, m, counter, block_len, flags));
            let st = blake3_compress(&cv, &m, counter, block_len, flags);
            cv = st[0..8].try_into().unwrap();
        }
        (blocks, cv0, cv)
    }

    /// Default Ligerito bundle roundtrip, byte-flip rejection, and file
    /// roundtrip. Requires m ≥ 21 — use n_blocks=256 (m=22 with K_LOG=14).
    #[test]
    #[ignore] // Heavier — run with `cargo test r1cs_bundle_roundtrip -- --ignored --nocapture`
    fn r1cs_bundle_roundtrip() {
        // K=256 → n_log=8 → m=22 with BLAKE3 K_LOG=14 (smallest Ligerito target).
        let setup = Blake3Setup::new(256);
        let (blocks, _, _) = honest_chain(256, 0xDEAD_5170);
        let mut ch = FsChallenger::new(b"flock-proofio-lig");
        let (proof, commitment, _claim) = setup.prove_fast(&blocks, &mut ch);

        let bundle = R1csProofBundleLigerito {
            commitment: commitment.clone(),
            proof: proof.clone(),
        };
        let bytes = bundle.to_bytes();
        assert_eq!(&bytes[0..5], &MAGIC);
        assert_eq!(bytes[5], VERSION);
        assert_eq!(bytes[6], FLAVOR_R1CS_LIGERITO);

        // The composite fast encode (bincode prefix + flat pcs_open) must be
        // byte-identical to the incumbent single-shot bincode of the bundle.
        let mut reference = Vec::new();
        write_header(&mut reference, FLAVOR_R1CS_LIGERITO);
        bincode::serialize_into(&mut reference, &bundle).expect("reference serialize");
        assert_eq!(
            bytes, reference,
            "fast to_bytes diverged from single-shot bincode"
        );

        // The pre-encoded-prefix path must also reproduce the same bytes.
        stash_pre_encoded_prefix(
            &bundle.commitment,
            &bundle.proof.zerocheck,
            &bundle.proof.lincheck,
        );
        assert_eq!(
            bundle.to_bytes(),
            reference,
            "stashed-prefix to_bytes diverged"
        );

        let bundle2 = R1csProofBundleLigerito::from_bytes(&bytes).expect("must round-trip");
        assert_eq!(bundle2.commitment.root, commitment.root);

        let mut chv = FsChallenger::new(b"flock-proofio-lig");
        setup
            .verify(&bundle2.commitment, &bundle2.proof, &mut chv)
            .expect("verify round-tripped Ligerito R1cs proof");

        // Byte-flipping inside the payload should make verification reject.
        // The flip can either fail deserialization OR succeed-then-fail-at-
        // verify; either is acceptable evidence the proof was consumed.
        let flip_at = HEADER_LEN + (bytes.len() - HEADER_LEN) / 2;
        let mut mutated = bytes.clone();
        mutated[flip_at] ^= 0xFF;
        match R1csProofBundleLigerito::from_bytes(&mutated) {
            Err(_) => {}
            Ok(bundle3) => {
                let mut chv = FsChallenger::new(b"flock-proofio-lig");
                let res = setup.verify(&bundle3.commitment, &bundle3.proof, &mut chv);
                assert!(res.is_err(), "verify must reject byte-mutated proof");
            }
        }

        // File roundtrip.
        let path = std::env::temp_dir().join("flock-proofio-roundtrip.bin");
        write_bytes_to_file(&path, &bytes).expect("write");
        let read_back = read_bytes_from_file(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        let bundle4 = R1csProofBundleLigerito::from_bytes(&read_back).expect("file round-trip");
        let mut chv = FsChallenger::new(b"flock-proofio-lig");
        setup
            .verify(&bundle4.commitment, &bundle4.proof, &mut chv)
            .expect("verify after file round-trip");

        eprintln!(
            "Ligerito R1csProofBundle: {} bytes ({:.1} KB)",
            bytes.len(),
            bytes.len() as f64 / 1024.0
        );
    }

    /// Ligerito chain bundle roundtrip. Requires m ≥ 21 — n=256 blocks.
    #[test]
    #[ignore] // Heavier — run with `cargo test chain_bundle_roundtrip -- --ignored --nocapture`
    fn chain_bundle_roundtrip_and_verify() {
        let setup = Blake3Setup::new(256);
        let (blocks, cv_0, cv_last) = honest_chain(256, 0xC0FFEE);
        let mut ch = FsChallenger::new(b"flock-proofio-test");
        let (proof, commitment) = setup.prove_chain(&blocks, &mut ch);

        let bundle = ChainProofBundleLigerito {
            hash_kind: HashKind::Blake3,
            commitment: commitment.clone(),
            proof: proof.clone(),
            cv_0_phys: cv_to_phys_bits(&cv_0),
            cv_last_phys: cv_to_phys_bits(&cv_last),
        };
        let bytes = bundle.to_bytes();
        assert_eq!(bytes[6], FLAVOR_CHAIN_LIGERITO);

        let bundle2 = ChainProofBundleLigerito::from_bytes(&bytes).expect("chain round-trip");
        assert_eq!(bundle2.cv_0_phys, bundle.cv_0_phys);
        assert_eq!(bundle2.cv_last_phys, bundle.cv_last_phys);

        let mut chv = FsChallenger::new(b"flock-proofio-test");
        setup
            .verify_chain(
                &bundle2.commitment,
                &bundle2.proof,
                &cv_0,
                &cv_last,
                &mut chv,
            )
            .expect("verify round-tripped chain proof");
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(b"NOPE!");
        bytes[5] = VERSION;
        bytes[6] = FLAVOR_R1CS_LIGERITO;
        let res = R1csProofBundleLigerito::from_bytes(&bytes);
        assert!(matches!(res, Err(DeserializeError::BadMagic)));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(&MAGIC);
        bytes[5] = VERSION.wrapping_add(1);
        bytes[6] = FLAVOR_R1CS_LIGERITO;
        let res = R1csProofBundleLigerito::from_bytes(&bytes);
        assert!(matches!(res, Err(DeserializeError::UnsupportedVersion(_))));
    }

    #[test]
    fn rejects_flavor_mismatch() {
        // R1CS-flavored header — try to read as Chain. Header validation
        // fails before any payload deserialization, so zero payload is fine.
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(&MAGIC);
        bytes[5] = VERSION;
        bytes[6] = FLAVOR_R1CS_LIGERITO;
        let res = ChainProofBundleLigerito::from_bytes(&bytes);
        assert!(matches!(
            res,
            Err(DeserializeError::FlavorMismatch {
                expected: FLAVOR_CHAIN_LIGERITO,
                found: FLAVOR_R1CS_LIGERITO
            })
        ));
    }

    #[test]
    fn rejects_legacy_basefold_flavor() {
        // Flavor bytes 0/1 were the legacy BaseFold bundles — now unknown.
        for legacy in [0u8, 1u8] {
            let mut bytes = vec![0u8; HEADER_LEN + 10];
            bytes[0..5].copy_from_slice(&MAGIC);
            bytes[5] = VERSION;
            bytes[6] = legacy;
            let res = R1csProofBundleLigerito::from_bytes(&bytes);
            assert!(matches!(res, Err(DeserializeError::UnknownFlavor(f)) if f == legacy));
        }
    }

    #[test]
    fn rejects_truncated() {
        let res = R1csProofBundleLigerito::from_bytes(&[0u8; 3]);
        assert!(matches!(res, Err(DeserializeError::Truncated)));
    }
}
