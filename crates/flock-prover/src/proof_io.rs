use std::io;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use flock_core::pcs::Commitment;

pub const MAGIC: [u8; 5] = *b"FLOCK";

pub const VERSION: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashKind {
    Blake3,
    Sha2,
    Keccak,
}

impl HashKind {
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

const FLAVOR_R1CS_LIGERITO: u8 = 2;
const FLAVOR_CHAIN_LIGERITO: u8 = 3;

const HEADER_LEN: usize = 7;

#[derive(Debug)]
pub enum DeserializeError {
    BadMagic,
    UnsupportedVersion(u8),
    UnknownFlavor(u8),
    Truncated,
    FlavorMismatch { expected: u8, found: u8 },
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofBundleLigerito {
    pub commitment: Commitment,
    pub proof: flock_core::proof::R1csProofLigerito,
}

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
        r1cs_parts_to_bytes(&self.commitment, &self.proof)
    }
    pub fn write_into<W: std::io::Write>(&self, mut w: W) -> std::io::Result<()> {
        if let Some(prefix) = take_matching_pre_encoded(self) {
            w.write_all(&prefix)?;
            encode_pcs_open_into(&mut w, &self.proof.pcs_open);
            return Ok(());
        }
        let mut header = Vec::with_capacity(HEADER_LEN);
        write_header(&mut header, FLAVOR_R1CS_LIGERITO);
        w.write_all(&header)?;
        bincode::serialize_into(&mut w, self).map_err(std::io::Error::other)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_R1CS_LIGERITO)?;
        Ok(bincode::deserialize(payload)?)
    }
}

pub(crate) fn r1cs_parts_to_bytes(
    commitment: &Commitment,
    proof: &flock_core::proof::R1csProofLigerito,
) -> Vec<u8> {
    if let Some(mut out) = take_matching_pre_encoded_parts(commitment, proof) {
        encode_pcs_open_into_vec(&mut out, &proof.pcs_open);
        return out;
    }
    let mut out = Vec::with_capacity(HEADER_LEN + 450_000);
    write_header(&mut out, FLAVOR_R1CS_LIGERITO);
    if fast_pcs_open_encode_enabled() {
        bincode::serialize_into(&mut out, commitment).expect("bincode serialize Commitment");
        bincode::serialize_into(&mut out, &proof.zerocheck)
            .expect("bincode serialize ZerocheckProof");
        bincode::serialize_into(&mut out, &proof.lincheck)
            .expect("bincode serialize LincheckProof");
        encode_pcs_open_into_vec(&mut out, &proof.pcs_open);
    } else {
        bincode::serialize_into(&mut out, commitment).expect("bincode serialize Commitment");
        bincode::serialize_into(&mut out, proof).expect("bincode serialize R1csProofLigerito");
    }
    out
}

struct PreEncodedPrefix {
    root: flock_core::merkle::Hash,
    sec_lens: [u64; 3],
    bytes: Vec<u8>,
}

static PRE_ENCODED: Mutex<Option<PreEncodedPrefix>> = Mutex::new(None);

pub(crate) fn pre_encode_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FLOCK_NO_PRE_ENCODE").map_or(true, |v| v != "1"))
}

pub fn stash_pre_encoded_prefix(
    commitment: &Commitment,
    zerocheck: &flock_core::zerocheck::ZerocheckProof,
    lincheck: &flock_core::lincheck::LincheckProof,
) {
    if !pre_encode_enabled() {
        return;
    }
    let mut bytes = Vec::with_capacity(HEADER_LEN + 450_000);
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
    if let Ok(mut slot) = PRE_ENCODED.lock() {
        *slot = Some(PreEncodedPrefix {
            root: commitment.root,
            sec_lens,
            bytes,
        });
    }
}

fn take_matching_pre_encoded(bundle: &R1csProofBundleLigerito) -> Option<Vec<u8>> {
    take_matching_pre_encoded_parts(&bundle.commitment, &bundle.proof)
}

fn take_matching_pre_encoded_parts(
    commitment: &Commitment,
    proof: &flock_core::proof::R1csProofLigerito,
) -> Option<Vec<u8>> {
    if !pre_encode_enabled() {
        return None;
    }
    let stash = PRE_ENCODED.lock().ok()?.take()?;
    if stash.root != commitment.root {
        return None;
    }
    if crate::seed_pipe::is_ranked_worker() {
        return Some(stash.bytes);
    }
    let sec_lens = [
        bincode::serialized_size(commitment).ok()?,
        bincode::serialized_size(&proof.zerocheck).ok()?,
        bincode::serialized_size(&proof.lincheck).ok()?,
    ];
    if sec_lens != stash.sec_lens {
        return None;
    }
    let want = HEADER_LEN + sec_lens.iter().sum::<u64>() as usize;
    (stash.bytes.len() == want).then_some(stash.bytes)
}

use flock_core::field::F128;
use flock_core::merkle::Hash as MerkleHash;
use flock_core::pcs::BatchOpeningProofLigerito;
use flock_core::pcs::ligerito::{FinalProof, LigeritoProof, RecursiveProof, SumcheckMessage};
use flock_core::pcs::ring_switch::RingSwitchProof;

fn fast_pcs_open_encode_enabled() -> bool {
    cfg!(target_endian = "little") && flock_core::micro_stack_enabled()
}

fn flat_tail_bulk_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FLOCK_NO_FLAT_TAIL_BULK").map_or(true, |v| v != "1"))
}

fn flat_rows_bulk_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FLOCK_NO_FLAT_ROWS_BULK").map_or(true, |v| v != "1"))
}

fn encode_pcs_open_into_vec(out: &mut Vec<u8>, p: &BatchOpeningProofLigerito) {
    if !fast_pcs_open_encode_enabled() || !flat_rows_bulk_enabled() {
        encode_pcs_open_into(out, p);
        return;
    }
    encode_pcs_open_fields(out, p, put_rows_bulk_vec);
}

fn encode_pcs_open_into<W: std::io::Write>(out: &mut W, p: &BatchOpeningProofLigerito) {
    if !fast_pcs_open_encode_enabled() {
        bincode::serialize_into(&mut *out, p).expect("bincode serialize pcs_open");
        return;
    }
    encode_pcs_open_fields(out, p, put_rows::<W>);
}

fn encode_pcs_open_fields<W, R>(out: &mut W, p: &BatchOpeningProofLigerito, put_rows_fn: R)
where
    W: std::io::Write,
    R: Fn(&mut W, &[Vec<F128>]) + Copy,
{
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
    out.write_all(initial_root).expect("encode write");
    put_recursive_proof(out, initial_proof, put_rows_fn);
    put_hash_vec(out, recursive_roots);
    put_u64(out, recursive_proofs.len() as u64);
    for rp in recursive_proofs {
        put_recursive_proof(out, rp, put_rows_fn);
    }
    let FinalProof {
        yr,
        opened_rows,
        merkle_proof,
    } = final_proof;
    put_f128_vec(out, yr);
    put_rows_fn(out, opened_rows);
    put_hash_vec(out, merkle_proof);
    put_sumcheck_vec(out, sumcheck_transcript);
    put_u64_vec(out, grinding_nonces);
    put_f128_vec(out, ood_values);
    put_u64_vec(out, fold_grinding_nonces);
}

fn put_recursive_proof<W, R>(out: &mut W, rp: &RecursiveProof, put_rows_fn: R)
where
    W: std::io::Write,
    R: Fn(&mut W, &[Vec<F128>]),
{
    let RecursiveProof {
        opened_rows,
        merkle_proof,
    } = rp;
    put_rows_fn(out, opened_rows);
    put_hash_vec(out, merkle_proof);
}

#[inline]
fn put_u64<W: std::io::Write>(out: &mut W, v: u64) {
    out.write_all(&v.to_le_bytes()).expect("encode write");
}

#[inline]
fn put_f128<W: std::io::Write>(out: &mut W, v: F128) {
    put_u64(out, v.lo);
    put_u64(out, v.hi);
}

#[inline]
fn put_f128_vec<W: std::io::Write>(out: &mut W, v: &[F128]) {
    put_u64(out, v.len() as u64);
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    out.write_all(bytes).expect("encode write");
}

#[inline]
fn put_hash_vec<W: std::io::Write>(out: &mut W, v: &[MerkleHash]) {
    put_u64(out, v.len() as u64);
    out.write_all(v.as_flattened()).expect("encode write");
}

#[inline]
fn put_sumcheck_vec<W: std::io::Write>(out: &mut W, v: &[SumcheckMessage]) {
    put_u64(out, v.len() as u64);
    if flat_tail_bulk_enabled() {
        const {
            assert!(std::mem::size_of::<SumcheckMessage>() == 2 * std::mem::size_of::<F128>());
            assert!(std::mem::align_of::<SumcheckMessage>() == std::mem::align_of::<F128>());
            assert!(std::mem::offset_of!(SumcheckMessage, u_0) == 0);
            assert!(std::mem::offset_of!(SumcheckMessage, u_2) == std::mem::size_of::<F128>());
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };
        out.write_all(bytes).expect("encode write");
    } else {
        for m in v {
            let SumcheckMessage { u_0, u_2 } = m;
            put_f128(out, *u_0);
            put_f128(out, *u_2);
        }
    }
}

#[inline]
fn put_u64_vec<W: std::io::Write>(out: &mut W, v: &[u64]) {
    put_u64(out, v.len() as u64);
    if flat_tail_bulk_enabled() {
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };
        out.write_all(bytes).expect("encode write");
    } else {
        for &x in v {
            put_u64(out, x);
        }
    }
}

fn put_rows<W: std::io::Write>(out: &mut W, rows: &[Vec<F128>]) {
    put_u64(out, rows.len() as u64);
    for row in rows {
        put_f128_vec(out, row);
    }
}

fn put_rows_bulk_vec(out: &mut Vec<u8>, rows: &[Vec<F128>]) {
    const LEN_BYTES: usize = std::mem::size_of::<u64>();
    let encoded_len = rows
        .iter()
        .try_fold(LEN_BYTES, |len, row| {
            len.checked_add(LEN_BYTES)?
                .checked_add(std::mem::size_of_val(row.as_slice()))
        })
        .expect("row encoding length overflow");
    let old_len = out.len();
    let new_len = old_len
        .checked_add(encoded_len)
        .expect("proof encoding length overflow");
    out.reserve(encoded_len);

    unsafe {
        let dst = out.as_mut_ptr().add(old_len);
        let mut written = 0usize;

        let nrows = (rows.len() as u64).to_le_bytes();
        std::ptr::copy_nonoverlapping(nrows.as_ptr(), dst.add(written), LEN_BYTES);
        written += LEN_BYTES;

        for row in rows {
            let row_len = (row.len() as u64).to_le_bytes();
            std::ptr::copy_nonoverlapping(row_len.as_ptr(), dst.add(written), LEN_BYTES);
            written += LEN_BYTES;

            let row_bytes = std::mem::size_of_val(row.as_slice());
            if row_bytes != 0 {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr().cast::<u8>(),
                    dst.add(written),
                    row_bytes,
                );
                written += row_bytes;
            }
        }
        debug_assert_eq!(written, encoded_len);
        out.set_len(new_len);
    }
}

pub(crate) struct GatherBundle<'a> {
    prefix: Vec<u8>,
    arena: Vec<u8>,
    segs: Vec<GatherSeg<'a>>,
}

enum GatherSeg<'a> {
    Arena(usize, usize),
    Ext(&'a [u8]),
}

impl<'a> GatherBundle<'a> {
    fn u64(&mut self, v: u64) {
        let start = self.arena.len();
        self.arena.extend_from_slice(&v.to_le_bytes());
        if let Some(GatherSeg::Arena(_, len)) = self.segs.last_mut() {
            *len += 8;
        } else {
            self.segs.push(GatherSeg::Arena(start, 8));
        }
    }
    fn bytes(&mut self, b: &'a [u8]) {
        if !b.is_empty() {
            self.segs.push(GatherSeg::Ext(b));
        }
    }
    fn f128s(&mut self, v: &'a [F128]) {
        self.u64(v.len() as u64);
        self.bytes(unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        });
    }
    fn hashes(&mut self, v: &'a [MerkleHash]) {
        self.u64(v.len() as u64);
        self.bytes(v.as_flattened());
    }
    fn rows(&mut self, rows: &'a [Vec<F128>]) {
        self.u64(rows.len() as u64);
        for row in rows {
            self.f128s(row);
        }
    }
    fn recursive(&mut self, rp: &'a RecursiveProof) {
        let RecursiveProof {
            opened_rows,
            merkle_proof,
        } = rp;
        self.rows(opened_rows);
        self.hashes(merkle_proof);
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.prefix.len()
            + self
                .segs
                .iter()
                .map(|s| match s {
                    GatherSeg::Arena(_, l) => *l,
                    GatherSeg::Ext(b) => b.len(),
                })
                .sum::<usize>()
    }

    pub(crate) fn io_slices(&self) -> Vec<std::io::IoSlice<'_>> {
        let mut out = Vec::with_capacity(self.segs.len() + 1);
        out.push(std::io::IoSlice::new(&self.prefix));
        for s in &self.segs {
            out.push(std::io::IoSlice::new(match s {
                GatherSeg::Arena(start, len) => &self.arena[*start..*start + *len],
                GatherSeg::Ext(b) => b,
            }));
        }
        out
    }

    pub(crate) fn write_all_to<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut slices = self.io_slices();
        let mut rest: &mut [std::io::IoSlice<'_>] = &mut slices;
        while !rest.is_empty() {
            let take = rest.len().min(1024);
            let n = w.write_vectored(&rest[..take])?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write whole proof",
                ));
            }
            std::io::IoSlice::advance_slices(&mut rest, n);
        }
        Ok(())
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn to_vec(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.len());
        for s in self.io_slices() {
            v.extend_from_slice(&s);
        }
        v
    }
}

pub(crate) fn gather_publish_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FLOCK_NO_GATHER_PUBLISH").map_or(true, |v| v != "1"))
}

pub(crate) fn r1cs_parts_gather<'a>(
    commitment: &Commitment,
    proof: &'a flock_core::proof::R1csProofLigerito,
) -> Option<GatherBundle<'a>> {
    if !gather_publish_enabled()
        || !fast_pcs_open_encode_enabled()
        || !flat_rows_bulk_enabled()
        || !flat_tail_bulk_enabled()
        || !crate::seed_pipe::is_ranked_worker()
    {
        return None;
    }
    let prefix = {
        let mut slot = PRE_ENCODED.lock().ok()?;
        if slot.as_ref()?.root != commitment.root {
            return None;
        }
        slot.take()?.bytes
    };
    let p = &proof.pcs_open;
    let mut g = GatherBundle {
        prefix,
        arena: Vec::with_capacity(64 * 1024),
        segs: Vec::with_capacity(4096),
    };
    let BatchOpeningProofLigerito {
        ring_switches,
        ligerito,
    } = p;
    g.u64(ring_switches.len() as u64);
    for rs in ring_switches {
        let RingSwitchProof { s_hat_v } = rs;
        g.f128s(s_hat_v);
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
    g.bytes(&initial_root[..]);
    g.recursive(initial_proof);
    g.hashes(recursive_roots);
    g.u64(recursive_proofs.len() as u64);
    for rp in recursive_proofs {
        g.recursive(rp);
    }
    let FinalProof {
        yr,
        opened_rows,
        merkle_proof,
    } = final_proof;
    g.f128s(yr);
    g.rows(opened_rows);
    g.hashes(merkle_proof);
    g.u64(sumcheck_transcript.len() as u64);
    g.bytes(unsafe {
        std::slice::from_raw_parts(
            sumcheck_transcript.as_ptr().cast::<u8>(),
            std::mem::size_of_val(sumcheck_transcript.as_slice()),
        )
    });
    g.u64(grinding_nonces.len() as u64);
    g.bytes(unsafe {
        std::slice::from_raw_parts(
            grinding_nonces.as_ptr().cast::<u8>(),
            std::mem::size_of_val(grinding_nonces.as_slice()),
        )
    });
    g.f128s(ood_values);
    g.u64(fold_grinding_nonces.len() as u64);
    g.bytes(unsafe {
        std::slice::from_raw_parts(
            fold_grinding_nonces.as_ptr().cast::<u8>(),
            std::mem::size_of_val(fold_grinding_nonces.as_slice()),
        )
    });
    Some(g)
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

pub fn read_bytes_from_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    std::fs::read(path)
}

pub fn write_chain_bundle_ligerito_to_file<P: AsRef<Path>>(
    path: P,
    bundle: &ChainProofBundleLigerito,
) -> io::Result<()> {
    write_bytes_to_file(path, &bundle.to_bytes())
}

pub fn read_chain_bundle_ligerito_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<ChainProofBundleLigerito, BundleReadError> {
    let bytes = read_bytes_from_file(path).map_err(BundleReadError::Io)?;
    ChainProofBundleLigerito::from_bytes(&bytes).map_err(BundleReadError::Deserialize)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::blake3::{Blake3Setup, Compression, blake3_compress, cv_to_phys_bits};
    use flock_core::challenger::FsChallenger;

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

    #[test]
    #[ignore]
    fn r1cs_bundle_roundtrip() {
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

        let bundle2 = R1csProofBundleLigerito::from_bytes(&bytes).expect("must round-trip");
        assert_eq!(bundle2.commitment.root, commitment.root);

        let mut chv = FsChallenger::new(b"flock-proofio-lig");
        setup
            .verify(&bundle2.commitment, &bundle2.proof, &mut chv)
            .expect("verify round-tripped Ligerito R1cs proof");

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

    #[test]
    #[ignore]
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

pub fn stash_r1cs_prefix(
    commitment: &Commitment,
    zerocheck: &flock_core::zerocheck::ZerocheckProof,
    lincheck: &flock_core::lincheck::LincheckProof,
) {
    stash_pre_encoded_prefix(commitment, zerocheck, lincheck);
}

pub fn take_stashed_r1cs_prefix(
    commitment: &Commitment,
    proof: &flock_core::proof::R1csProofLigerito,
) -> Option<Vec<u8>> {
    take_matching_pre_encoded_parts(commitment, proof)
}

pub fn clear_stashed_r1cs_prefix() {
    if let Ok(mut slot) = PRE_ENCODED.lock() {
        *slot = None;
    }
}
