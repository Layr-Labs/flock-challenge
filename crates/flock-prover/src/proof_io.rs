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

use flock_core::field::F128;
use flock_core::merkle::{Hash, HashKind as MerkleHashKind};
use flock_core::pcs::Commitment;
use flock_core::pcs::ligerito::{LigeritoProfile, RecursiveProof};

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

/// The ranked R1CS bundle is bounded at 500 kB by the trusted harness.
///
/// This is only a capacity hint: larger bundles still grow normally. Starting
/// at the reviewed bound avoids the nine geometric growths made by the old
/// 1,031-byte allocation for the ranked 436–438 kB payload.
const R1CS_BUNDLE_CAPACITY_HINT: usize = 500_000;

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
        let mut out = Vec::with_capacity(R1CS_BUNDLE_CAPACITY_HINT);
        write_header(&mut out, FLAVOR_R1CS_LIGERITO);
        write_r1cs_payload(&mut out, self);
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_R1CS_LIGERITO)?;
        Ok(bincode::deserialize(payload)?)
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

#[inline(always)]
fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[inline(always)]
fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[inline(always)]
fn write_len(out: &mut Vec<u8>, len: usize) {
    write_u64(out, len as u64);
}

#[inline(always)]
fn write_f128(out: &mut Vec<u8>, value: F128) {
    write_u64(out, value.lo);
    write_u64(out, value.hi);
}

#[inline]
fn write_f128_slice(out: &mut Vec<u8>, values: &[F128]) {
    write_len(out, values.len());
    #[cfg(target_endian = "little")]
    {
        const {
            assert!(core::mem::size_of::<F128>() == 16);
        }
        // SAFETY: `F128` is `repr(C, align(16))` with exactly two initialized
        // `u64` fields and size 16, hence no padding. On a little-endian
        // target its object bytes are exactly bincode's fixed-int encoding.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        out.extend_from_slice(bytes);
    }
    #[cfg(target_endian = "big")]
    for &value in values {
        write_f128(out, value);
    }
}

#[inline(always)]
fn write_hash(out: &mut Vec<u8>, value: &Hash) {
    out.extend_from_slice(value);
}

#[inline]
fn write_hash_slice(out: &mut Vec<u8>, values: &[Hash]) {
    write_len(out, values.len());
    // SAFETY: `Hash` is `[u8; 32]`; arrays and slices are contiguous and
    // `[u8; 32]` has no padding. Bincode encodes the array elements without
    // an inner length prefix, so the flattened bytes are identical.
    let bytes = unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    };
    out.extend_from_slice(bytes);
}

#[inline]
fn write_f128_pairs(out: &mut Vec<u8>, values: &[(F128, F128)]) {
    write_len(out, values.len());
    for &(first, second) in values {
        write_f128(out, first);
        write_f128(out, second);
    }
}

#[inline]
fn write_opened_rows(out: &mut Vec<u8>, rows: &[Vec<F128>]) {
    write_len(out, rows.len());
    for row in rows {
        write_f128_slice(out, row);
    }
}

#[inline]
fn write_recursive_proof(out: &mut Vec<u8>, proof: &RecursiveProof) {
    write_opened_rows(out, &proof.opened_rows);
    write_hash_slice(out, &proof.merkle_proof);
}

/// Encode the v4 R1CS payload exactly like bincode 1.3's default
/// fixed-integer, little-endian serializer.
///
/// Keeping this encoder next to `VERSION` makes the schema coupling explicit.
/// The heavy proof-IO round-trip test compares these bytes with bincode.
fn write_r1cs_payload(out: &mut Vec<u8>, bundle: &R1csProofBundleLigerito) {
    let commitment = &bundle.commitment;
    write_hash(out, &commitment.root);
    write_u64(out, commitment.params.m as u64);
    write_u64(out, commitment.params.log_inv_rate as u64);
    write_u64(out, commitment.params.log_batch_size as u64);
    write_u32(
        out,
        match commitment.params.profile {
            LigeritoProfile::Fast => 0,
            LigeritoProfile::Slim => 1,
            LigeritoProfile::Secure => 2,
        },
    );
    write_u32(
        out,
        match commitment.params.merkle_hash {
            MerkleHashKind::Sha256 => 0,
            MerkleHashKind::Blake3 => 1,
        },
    );

    let zerocheck = &bundle.proof.zerocheck;
    write_f128_slice(out, &zerocheck.round1_ab);
    write_f128_slice(out, &zerocheck.round1_c);
    write_f128_pairs(out, &zerocheck.multilinear_rounds);
    write_f128(out, zerocheck.final_a_eval);
    write_f128(out, zerocheck.final_b_eval);
    write_f128(out, zerocheck.final_c_eval);

    let lincheck = &bundle.proof.lincheck;
    write_f128_pairs(out, &lincheck.rounds);
    write_f128_slice(out, &lincheck.z_partial);

    let opening = &bundle.proof.pcs_open;
    write_len(out, opening.ring_switches.len());
    for ring_switch in &opening.ring_switches {
        write_f128_slice(out, &ring_switch.s_hat_v);
    }

    let proof = &opening.ligerito;
    write_hash(out, &proof.initial_root);
    write_recursive_proof(out, &proof.initial_proof);
    write_hash_slice(out, &proof.recursive_roots);
    write_len(out, proof.recursive_proofs.len());
    for recursive in &proof.recursive_proofs {
        write_recursive_proof(out, recursive);
    }
    write_f128_slice(out, &proof.final_proof.yr);
    write_opened_rows(out, &proof.final_proof.opened_rows);
    write_hash_slice(out, &proof.final_proof.merkle_proof);
    write_len(out, proof.sumcheck_transcript.len());
    for message in &proof.sumcheck_transcript {
        write_f128(out, message.u_0);
        write_f128(out, message.u_2);
    }
    write_len(out, proof.grinding_nonces.len());
    for &nonce in &proof.grinding_nonces {
        write_u64(out, nonce);
    }
    write_f128_slice(out, &proof.ood_values);
    write_len(out, proof.fold_grinding_nonces.len());
    for &nonce in &proof.fold_grinding_nonces {
        write_u64(out, nonce);
    }
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
        let mut bincode_bytes = Vec::with_capacity(bytes.len());
        write_header(&mut bincode_bytes, FLAVOR_R1CS_LIGERITO);
        bincode::serialize_into(&mut bincode_bytes, &bundle)
            .expect("bincode serialize R1csProofBundleLigerito");
        assert_eq!(bytes, bincode_bytes, "direct encoder must match bincode");
        assert_eq!(&bytes[0..5], &MAGIC);
        assert_eq!(bytes[5], VERSION);
        assert_eq!(bytes[6], FLAVOR_R1CS_LIGERITO);

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
