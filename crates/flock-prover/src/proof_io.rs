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
        let mut out = Vec::with_capacity(HEADER_LEN + fast_ser::bundle_size(self));
        write_header(&mut out, FLAVOR_R1CS_LIGERITO);
        fast_ser::write_bundle(&mut out, self);
        debug_assert_eq!(out.len(), out.capacity(), "fast_ser size estimate must be exact");
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_R1CS_LIGERITO)?;
        Ok(bincode::deserialize(payload)?)
    }
}

/// Hand-rolled encoder for [`R1csProofBundleLigerito`], byte-identical to
/// `bincode::serialize` of the same value (bincode 1.3 legacy options:
/// fixint encoding, little-endian; `usize` as u64 LE, unit enum variants as
/// u32 LE index, `Vec` as u64 LE length + elements, arrays/tuples/structs
/// as plain field concatenation, no framing).
///
/// Every type reachable from the bundle is POD under that encoding, so the
/// large `Vec<F128>` / `Vec<Hash>` / `Vec<u64>` payloads are bulk-copied
/// instead of dispatched element-by-element through serde, and the output
/// vector is allocated once at its exact final size. Byte identity is
/// locked by `tests::fast_encoder_matches_bincode` (synthetic structures,
/// all enum variants, empty/non-empty vectors) and by the real-proof
/// assertion in `tests::r1cs_bundle_roundtrip`.
///
/// `from_bytes` stays on `bincode::deserialize`: decode runs outside the
/// prover's timed window.
mod fast_ser {
    use super::R1csProofBundleLigerito;
    use flock_core::field::F128;
    use flock_core::hash::HashKind;
    use flock_core::lincheck::LincheckProof;
    use flock_core::merkle::Hash;
    use flock_core::pcs::ligerito::{
        FinalProof, LigeritoProfile, LigeritoProof, RecursiveProof,
    };
    use flock_core::pcs::{Commitment, RingSwitchProof};
    use flock_core::zerocheck::ZerocheckProof;

    /// Bytes per `Vec` length prefix / `usize` / `u64`.
    const LEN: usize = 8;
    /// Bytes per serialized `F128` (`lo` u64 LE ++ `hi` u64 LE).
    const ELEM: usize = 16;
    /// Bytes per `Hash = [u8; 32]` (serialized as a 32-tuple: raw bytes).
    const HASH: usize = 32;

    // The bulk copies below reinterpret `&[F128]` as raw bytes. That is
    // bincode-identical only if F128 is exactly two u64 with no padding.
    const _: () = assert!(core::mem::size_of::<F128>() == 16);

    // -----------------------------------------------------------------
    // Primitive writers
    // -----------------------------------------------------------------

    #[inline]
    fn put_u64(out: &mut Vec<u8>, v: u64) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn put_usize(out: &mut Vec<u8>, n: usize) {
        put_u64(out, n as u64);
    }

    /// Unit enum variant: bincode encodes the serde variant index as u32 LE.
    #[inline]
    fn put_variant(out: &mut Vec<u8>, idx: u32) {
        out.extend_from_slice(&idx.to_le_bytes());
    }

    #[inline]
    fn put_f128(out: &mut Vec<u8>, v: F128) {
        put_u64(out, v.lo);
        put_u64(out, v.hi);
    }

    fn put_f128_slice(out: &mut Vec<u8>, xs: &[F128]) {
        put_usize(out, xs.len());
        #[cfg(target_endian = "little")]
        {
            // SAFETY: F128 is repr(C) { lo: u64, hi: u64 }, 16 bytes with no
            // padding (compile-time assert above). On a little-endian target
            // its in-memory bytes are exactly bincode's fixint output
            // (lo LE ++ hi LE) for each element.
            let bytes = unsafe {
                core::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), xs.len() * ELEM)
            };
            out.extend_from_slice(bytes);
        }
        #[cfg(not(target_endian = "little"))]
        for &x in xs {
            put_f128(out, x);
        }
    }

    fn put_u64_slice(out: &mut Vec<u8>, xs: &[u64]) {
        put_usize(out, xs.len());
        #[cfg(target_endian = "little")]
        {
            // SAFETY: u64 LE in memory == bincode fixint LE output.
            let bytes = unsafe {
                core::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), xs.len() * LEN)
            };
            out.extend_from_slice(bytes);
        }
        #[cfg(not(target_endian = "little"))]
        for &x in xs {
            put_u64(out, x);
        }
    }

    fn put_hash_slice(out: &mut Vec<u8>, hs: &[Hash]) {
        put_usize(out, hs.len());
        // [u8; 32] serializes as its 32 raw bytes; a slice of arrays is
        // guaranteed contiguous, so the flattened view is the exact payload.
        out.extend_from_slice(hs.as_flattened());
    }

    /// `Vec<(F128, F128)>`: tuples are element concatenation. These vectors
    /// are tiny (one entry per sumcheck round), so per-element writes are
    /// fine — and `(F128, F128)` is repr(Rust), so no layout guarantee for a
    /// bulk copy anyway.
    fn put_f128_pair_slice(out: &mut Vec<u8>, xs: &[(F128, F128)]) {
        put_usize(out, xs.len());
        for &(a, b) in xs {
            put_f128(out, a);
            put_f128(out, b);
        }
    }

    // -----------------------------------------------------------------
    // Struct writers (fields in declaration order, exactly as serde derive)
    // -----------------------------------------------------------------

    fn put_commitment(out: &mut Vec<u8>, c: &Commitment) {
        out.extend_from_slice(&c.root);
        put_usize(out, c.params.m);
        put_usize(out, c.params.log_inv_rate);
        put_usize(out, c.params.log_batch_size);
        put_variant(
            out,
            match c.params.profile {
                LigeritoProfile::Fast => 0,
                LigeritoProfile::Slim => 1,
                LigeritoProfile::Secure => 2,
            },
        );
        put_variant(
            out,
            match c.params.merkle_hash {
                HashKind::Sha256 => 0,
                HashKind::Blake3 => 1,
            },
        );
    }

    fn put_zerocheck(out: &mut Vec<u8>, z: &ZerocheckProof) {
        put_f128_slice(out, &z.round1_ab);
        put_f128_slice(out, &z.round1_c);
        put_f128_pair_slice(out, &z.multilinear_rounds);
        put_f128(out, z.final_a_eval);
        put_f128(out, z.final_b_eval);
        put_f128(out, z.final_c_eval);
    }

    fn put_lincheck(out: &mut Vec<u8>, l: &LincheckProof) {
        put_f128_pair_slice(out, &l.rounds);
        put_f128_slice(out, &l.z_partial);
    }

    fn put_recursive(out: &mut Vec<u8>, p: &RecursiveProof) {
        put_usize(out, p.opened_rows.len());
        for row in &p.opened_rows {
            put_f128_slice(out, row);
        }
        put_hash_slice(out, &p.merkle_proof);
    }

    fn put_final(out: &mut Vec<u8>, p: &FinalProof) {
        put_f128_slice(out, &p.yr);
        put_usize(out, p.opened_rows.len());
        for row in &p.opened_rows {
            put_f128_slice(out, row);
        }
        put_hash_slice(out, &p.merkle_proof);
    }

    fn put_ligerito(out: &mut Vec<u8>, p: &LigeritoProof) {
        out.extend_from_slice(&p.initial_root);
        put_recursive(out, &p.initial_proof);
        put_hash_slice(out, &p.recursive_roots);
        put_usize(out, p.recursive_proofs.len());
        for rp in &p.recursive_proofs {
            put_recursive(out, rp);
        }
        put_final(out, &p.final_proof);
        put_usize(out, p.sumcheck_transcript.len());
        for m in &p.sumcheck_transcript {
            put_f128(out, m.u_0);
            put_f128(out, m.u_2);
        }
        put_u64_slice(out, &p.grinding_nonces);
        put_f128_slice(out, &p.ood_values);
        put_u64_slice(out, &p.fold_grinding_nonces);
    }

    fn put_ring_switches(out: &mut Vec<u8>, rs: &[RingSwitchProof]) {
        put_usize(out, rs.len());
        for r in rs {
            put_f128_slice(out, &r.s_hat_v);
        }
    }

    pub(super) fn write_bundle(out: &mut Vec<u8>, b: &R1csProofBundleLigerito) {
        put_commitment(out, &b.commitment);
        put_zerocheck(out, &b.proof.zerocheck);
        put_lincheck(out, &b.proof.lincheck);
        put_ring_switches(out, &b.proof.pcs_open.ring_switches);
        put_ligerito(out, &b.proof.pcs_open.ligerito);
    }

    // -----------------------------------------------------------------
    // Exact payload size (cheap length walk, no serialization)
    // -----------------------------------------------------------------

    fn recursive_size(p: &RecursiveProof) -> usize {
        LEN + p.opened_rows.iter().map(|r| LEN + r.len() * ELEM).sum::<usize>()
            + LEN
            + p.merkle_proof.len() * HASH
    }

    pub(super) fn bundle_size(b: &R1csProofBundleLigerito) -> usize {
        // Commitment: root + PcsParams { 3×usize, 2×u32-variant }.
        let commitment = HASH + 3 * LEN + 4 + 4;
        let z = &b.proof.zerocheck;
        let zerocheck = LEN + z.round1_ab.len() * ELEM
            + LEN + z.round1_c.len() * ELEM
            + LEN + z.multilinear_rounds.len() * 2 * ELEM
            + 3 * ELEM;
        let l = &b.proof.lincheck;
        let lincheck = LEN + l.rounds.len() * 2 * ELEM + LEN + l.z_partial.len() * ELEM;
        let ring_switches = LEN
            + b.proof
                .pcs_open
                .ring_switches
                .iter()
                .map(|r| LEN + r.s_hat_v.len() * ELEM)
                .sum::<usize>();
        let p = &b.proof.pcs_open.ligerito;
        let f = &p.final_proof;
        let final_proof = LEN + f.yr.len() * ELEM
            + LEN + f.opened_rows.iter().map(|r| LEN + r.len() * ELEM).sum::<usize>()
            + LEN + f.merkle_proof.len() * HASH;
        let ligerito = HASH
            + recursive_size(&p.initial_proof)
            + LEN + p.recursive_roots.len() * HASH
            + LEN + p.recursive_proofs.iter().map(recursive_size).sum::<usize>()
            + final_proof
            + LEN + p.sumcheck_transcript.len() * 2 * ELEM
            + LEN + p.grinding_nonces.len() * LEN
            + LEN + p.ood_values.len() * ELEM
            + LEN + p.fold_grinding_nonces.len() * LEN;
        commitment + zerocheck + lincheck + ring_switches + ligerito
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
    use flock_core::field::F128;
    use flock_core::pcs::PcsParams;
    use flock_core::pcs::ligerito::{
        FinalProof, LigeritoProfile, LigeritoProof, RecursiveProof, SumcheckMessage,
    };
    use flock_core::pcs::ring_switch::RingSwitchProof;

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

        // The hand-rolled fast encoder must be byte-identical to bincode on
        // a real proof.
        let mut reference = Vec::new();
        write_header(&mut reference, FLAVOR_R1CS_LIGERITO);
        bincode::serialize_into(&mut reference, &bundle).expect("bincode reference");
        assert_eq!(bytes, reference, "fast encoder must match bincode::serialize");

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

    /// Build a synthetic (not verifier-valid) but structurally complete
    /// bundle: every field populated from the rng, with the given enum
    /// variants and a mix of empty and non-empty vectors. Byte identity of
    /// the encoder does not depend on proof validity, so this exercises
    /// every encoding path far more broadly than one honest proof.
    fn synthetic_bundle(
        rng: &mut Rng,
        profile: LigeritoProfile,
        merkle_hash: flock_core::hash::HashKind,
        with_optional_vecs: bool,
    ) -> R1csProofBundleLigerito {
        let f128 = |rng: &mut Rng| F128 {
            lo: rng.nx(),
            hi: rng.nx(),
        };
        let f128_vec = |rng: &mut Rng, n: usize| -> Vec<F128> {
            (0..n).map(|_| F128 {
                lo: rng.nx(),
                hi: rng.nx(),
            })
            .collect()
        };
        let hash = |rng: &mut Rng| -> flock_core::merkle::Hash {
            let mut h = [0u8; 32];
            for chunk in h.chunks_mut(8) {
                chunk.copy_from_slice(&rng.nx().to_le_bytes());
            }
            h
        };
        let pair_vec = |rng: &mut Rng, n: usize| -> Vec<(F128, F128)> {
            (0..n).map(|_| {
                (
                    F128 {
                        lo: rng.nx(),
                        hi: rng.nx(),
                    },
                    F128 {
                        lo: rng.nx(),
                        hi: rng.nx(),
                    },
                )
            })
            .collect()
        };
        let recursive = |rng: &mut Rng, rows: usize, width: usize, proof_len: usize| {
            RecursiveProof {
                opened_rows: (0..rows)
                    .map(|_| {
                        (0..width).map(|_| F128 {
                            lo: rng.nx(),
                            hi: rng.nx(),
                        })
                        .collect()
                    })
                    .collect(),
                merkle_proof: (0..proof_len)
                    .map(|_| {
                        let mut h = [0u8; 32];
                        for chunk in h.chunks_mut(8) {
                            chunk.copy_from_slice(&rng.nx().to_le_bytes());
                        }
                        h
                    })
                    .collect(),
            }
        };
        R1csProofBundleLigerito {
            commitment: Commitment {
                root: hash(rng),
                params: PcsParams {
                    m: 22,
                    log_inv_rate: 1,
                    log_batch_size: 5,
                    profile,
                    merkle_hash,
                },
            },
            proof: flock_core::proof::R1csProofLigerito {
                zerocheck: flock_core::zerocheck::ZerocheckProof {
                    round1_ab: f128_vec(rng, 17),
                    round1_c: f128_vec(rng, 17),
                    multilinear_rounds: pair_vec(rng, 13),
                    final_a_eval: f128(rng),
                    final_b_eval: f128(rng),
                    final_c_eval: f128(rng),
                },
                lincheck: flock_core::lincheck::LincheckProof {
                    rounds: pair_vec(rng, 11),
                    z_partial: f128_vec(rng, 8),
                },
                pcs_open: flock_core::pcs::BatchOpeningProofLigerito {
                    ring_switches: vec![
                        RingSwitchProof {
                            s_hat_v: f128_vec(rng, 16),
                        },
                        RingSwitchProof {
                            s_hat_v: Vec::new(),
                        },
                        RingSwitchProof {
                            s_hat_v: f128_vec(rng, 3),
                        },
                    ],
                    ligerito: LigeritoProof {
                        initial_root: hash(rng),
                        initial_proof: recursive(rng, 7, 32, 19),
                        recursive_roots: (0..3).map(|_| hash(rng)).collect(),
                        recursive_proofs: vec![
                            recursive(rng, 5, 16, 11),
                            recursive(rng, 0, 0, 0), // empty level
                            recursive(rng, 2, 1, 3),
                        ],
                        final_proof: FinalProof {
                            yr: f128_vec(rng, 64),
                            opened_rows: vec![f128_vec(rng, 4), Vec::new(), f128_vec(rng, 2)],
                            merkle_proof: (0..5).map(|_| hash(rng)).collect(),
                        },
                        sumcheck_transcript: (0..9)
                            .map(|_| SumcheckMessage {
                                u_0: f128(rng),
                                u_2: f128(rng),
                            })
                            .collect(),
                        grinding_nonces: if with_optional_vecs {
                            (0..4).map(|_| rng.nx()).collect()
                        } else {
                            Vec::new()
                        },
                        ood_values: if with_optional_vecs {
                            f128_vec(rng, 6)
                        } else {
                            Vec::new()
                        },
                        fold_grinding_nonces: if with_optional_vecs {
                            (0..2).map(|_| rng.nx()).collect()
                        } else {
                            Vec::new()
                        },
                    },
                },
            },
        }
    }

    /// The fast encoder must produce exactly `header ++ bincode::serialize`
    /// for arbitrary bundle contents — across all enum variants and with
    /// empty and non-empty optional vectors — and `from_bytes` must
    /// round-trip it.
    #[test]
    fn fast_encoder_matches_bincode() {
        use flock_core::hash::HashKind as MerkleHashKind;
        let mut rng = Rng::new(0xFA57_5E12);
        let cases = [
            (LigeritoProfile::Fast, MerkleHashKind::Blake3, true),
            (LigeritoProfile::Slim, MerkleHashKind::Sha256, false),
            (LigeritoProfile::Secure, MerkleHashKind::Blake3, false),
            (LigeritoProfile::Fast, MerkleHashKind::Sha256, true),
        ];
        for (profile, merkle_hash, with_optional) in cases {
            let bundle = synthetic_bundle(&mut rng, profile, merkle_hash, with_optional);

            let bytes = bundle.to_bytes();
            let mut reference = Vec::new();
            write_header(&mut reference, FLAVOR_R1CS_LIGERITO);
            bincode::serialize_into(&mut reference, &bundle).expect("bincode reference");
            assert_eq!(
                bytes, reference,
                "fast encoder must match bincode ({profile:?}, {merkle_hash:?}, optional={with_optional})"
            );

            // Round-trip: decode and re-encode; identical bytes imply the
            // decoded bundle equals the original field-for-field.
            let decoded = R1csProofBundleLigerito::from_bytes(&bytes).expect("round-trip");
            assert_eq!(decoded.to_bytes(), bytes, "round-trip must be lossless");
        }
    }
}
