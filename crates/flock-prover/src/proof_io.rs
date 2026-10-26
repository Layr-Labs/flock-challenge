use std::io;
use std::path::Path;

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
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_R1CS_LIGERITO);
        bincode::serialize_into(&mut out, self).expect("bincode serialize R1csProofBundleLigerito");
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
