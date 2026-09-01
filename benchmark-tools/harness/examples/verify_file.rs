//! Verify-only oracle: check a worker-produced proof file against a seed.
//! Mirrors `verify_proof` in src/main.rs (the trusted driver's logic), but
//! takes an existing proof file instead of spawning a worker.
//!
//! Run: verify_file <proof-file> <seed> <log2>
//! Exits 0 with "verified ok: <bytes>" on success; nonzero with the error.
//! Used to validate proofs produced by a qemu-emulated AVX-512 worker that
//! the native trusted verifier cannot spawn (exec'd children are native).

use std::fs;

use bincode::Options;
use flock_benchmark_common::{DOMAIN, generate_compressions};
use flock_prover::challenger::FsChallenger;
use flock_prover::merkle::HashKind;
use flock_prover::pcs;
use flock_prover::proof_io::R1csProofBundleLigerito;
use flock_prover::r1cs_hashes::blake3::Blake3Setup;

const MAX_PROOF_BYTES: u64 = 500_000;
const BENCHMARK_HASH: HashKind = HashKind::Blake3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let proof_path = args.next().expect("usage: verify_file <proof> <seed> <log2>");
    let seed: u64 = args.next().expect("seed").parse().expect("seed u64");
    let log2: u32 = args.next().expect("log2").parse().expect("log2 u32");

    let bytes = fs::read(&proof_path)?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_PROOF_BYTES {
        return Err(format!("proof size {} outside allowed range", bytes.len()).into());
    }

    let bundle = deserialize_bundle(&bytes)?;
    let mut setup = Blake3Setup::new(1usize << log2);
    setup.pcs_params.merkle_hash = BENCHMARK_HASH;
    let blocks = generate_compressions(log2, seed);
    let witness = setup.generate_witness_packed(&blocks);
    let (expected, _) = pcs::commit(&witness, &setup.pcs_params);

    if bundle.commitment.root != expected.root
        || bundle.commitment.params.m != setup.pcs_params.m
        || bundle.commitment.params.log_inv_rate != setup.pcs_params.log_inv_rate
        || bundle.commitment.params.log_batch_size != setup.pcs_params.log_batch_size
        || bundle.commitment.params.profile != setup.pcs_params.profile
        || bundle.commitment.params.merkle_hash != BENCHMARK_HASH
    {
        return Err("proof commitment does not match the trusted BLAKE3 witness".into());
    }

    let mut challenger = FsChallenger::with_hash(DOMAIN, BENCHMARK_HASH);
    setup
        .verify(&bundle.commitment, &bundle.proof, &mut challenger)
        .map_err(|error| format!("trusted verifier rejected proof: {error:?}"))?;
    println!("verified ok: {} bytes", bytes.len());
    Ok(())
}

fn deserialize_bundle(bytes: &[u8]) -> Result<R1csProofBundleLigerito, Box<dyn std::error::Error>> {
    const HEADER_LEN: usize = 7;
    if bytes.len() < HEADER_LEN
        || bytes[..5] != flock_prover::proof_io::MAGIC
        || bytes[5] != flock_prover::proof_io::VERSION
        || bytes[6] != 2
    {
        return Err("invalid FLOCK R1CS proof header".into());
    }
    Ok(bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_PROOF_BYTES)
        .reject_trailing_bytes()
        .deserialize(&bytes[HEADER_LEN..])?)
}
