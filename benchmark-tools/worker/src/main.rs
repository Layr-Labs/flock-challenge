use std::hint::black_box;
use std::io::{self, BufRead};

use flock_benchmark_common::{DOMAIN, generate_compressions};
use flock_prover::challenger::FsChallenger;
use flock_prover::proof_io::R1csProofBundleLigerito;
use flock_prover::r1cs_hashes::blake3::Blake3Setup;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let log2_size: u32 = args.next().ok_or("missing LOG2")?.parse()?;
    let ready_path = args.next().ok_or("missing READY_PATH")?;
    let proof_path = args.next().ok_or("missing PROOF_PATH")?;
    if args.next().is_some() || !(8..=20).contains(&log2_size) {
        return Err("usage: flock-benchmark-worker LOG2 READY_PATH PROOF_PATH".into());
    }

    let _ = flock_prover::init_perf_thread_pool();
    let setup = Blake3Setup::new(1usize << log2_size);

    // This matches the upstream bench's untimed, same-process warm-up. The
    // measured seed is not sent to this process until after readiness.
    {
        let blocks = generate_compressions(log2_size, 0x00C0_FFEE_BEEF_D15C);
        let mut challenger = FsChallenger::new(DOMAIN);
        black_box(setup.prove_fast(&blocks, &mut challenger));
    }

    std::fs::write(ready_path, b"ready\n")?;
    let mut request = String::new();
    if io::stdin().lock().read_line(&mut request)? == 0 {
        return Err("missing seed on stdin".into());
    }

    let seed: u64 = request.trim().parse()?;
    let blocks = generate_compressions(log2_size, seed);
    let mut challenger = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = setup.prove_fast(&blocks, &mut challenger);
    let bundle = R1csProofBundleLigerito { commitment, proof };
    let temporary_proof_path = format!("{proof_path}.tmp");
    std::fs::write(&temporary_proof_path, bundle.to_bytes())?;
    std::fs::rename(temporary_proof_path, proof_path)?;
    Ok(())
}
