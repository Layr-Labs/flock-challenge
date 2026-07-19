//! Trusted driver, verifier, timer, and score writer for the BLAKE3 challenge.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use bincode::Options;
use flock_benchmark_common::{DOMAIN, generate_compressions};
use flock_prover::challenger::FsChallenger;
use flock_prover::pcs;
use flock_prover::proof_io::R1csProofBundleLigerito;
use flock_prover::r1cs_hashes::blake3::Blake3Setup;
use serde::Serialize;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const RUN_TIMEOUT: Duration = Duration::from_secs(900);
const POLL_INTERVAL: Duration = Duration::from_micros(100);
// A canonical 2^16 proof is at most about 409 kB for the fixed m=30 profile.
// Leave room for serialization variation while rejecting oversized input.
const MAX_PROOF_BYTES: u64 = 500_000;

struct Config {
    worker: PathBuf,
    scratch: PathBuf,
    score: PathBuf,
    summary: PathBuf,
    log2_size: u32,
    threads: usize,
    runs: usize,
    sandbox_profile: Option<PathBuf>,
}

struct Trial {
    seconds: f64,
    proof_bytes: usize,
}

#[derive(Serialize)]
struct ScoreFile {
    score: f64,
    metrics: ScoreMetrics,
}

#[derive(Serialize)]
struct ScoreMetrics {
    trial_seconds: Vec<f64>,
    batch_size: usize,
    threads: usize,
    proof_bytes: usize,
    verified: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    std::fs::create_dir_all(&config.scratch)?;

    let mut trials = Vec::with_capacity(config.runs);
    for run in 1..=config.runs {
        trials.push(run_trial(&config, run)?);
    }

    let best = trials
        .iter()
        .min_by(|left, right| left.seconds.total_cmp(&right.seconds))
        .ok_or("no benchmark trials")?;
    if !best.seconds.is_finite() || best.seconds <= 0.0 {
        return Err("benchmark duration must be finite and positive".into());
    }

    let batch_size = 1usize << config.log2_size;
    let throughput = batch_size as f64 / best.seconds;
    write_results(&config, &trials, best, throughput, batch_size)?;
    println!("score={throughput:.3} compressions_per_second");
    Ok(())
}

fn run_trial(config: &Config, run: usize) -> Result<Trial, Box<dyn std::error::Error>> {
    let ready = config.scratch.join(format!("run-{run}.ready"));
    let proof = config.scratch.join(format!("run-{run}.proof"));
    let _ = std::fs::remove_file(&ready);
    let _ = std::fs::remove_file(&proof);

    let mut command = worker_command(config, &ready, &proof);
    command
        .env_clear()
        .env("RAYON_NUM_THREADS", config.threads.to_string())
        .env("TMPDIR", &config.scratch)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let mut child = command.spawn()?;
    wait_for_ready(&mut child, &ready)?;
    let seed = match fresh_seed() {
        Ok(seed) => seed,
        Err(error) => {
            stop(&mut child);
            return Err(error);
        }
    };

    let start = Instant::now();
    let Some(mut stdin) = child.stdin.take() else {
        stop(&mut child);
        return Err("worker stdin unavailable".into());
    };
    if let Err(error) = writeln!(stdin, "{seed}") {
        stop(&mut child);
        return Err(error.into());
    }
    let status = wait_for_exit(&mut child, RUN_TIMEOUT)?;
    let seconds = start.elapsed().as_secs_f64();
    if !status.success() {
        return Err(format!("run {run}: worker exited with {status}").into());
    }

    let proof_bytes = verify_proof(config.log2_size, seed, &proof)?;
    let _ = std::fs::remove_file(ready);
    let _ = std::fs::remove_file(proof);
    Ok(Trial {
        seconds,
        proof_bytes,
    })
}

fn worker_command(config: &Config, ready: &Path, proof: &Path) -> Command {
    let mut command = if let Some(profile) = &config.sandbox_profile {
        let mut sandbox = Command::new("/usr/bin/sandbox-exec");
        sandbox.arg("-f").arg(profile).arg(&config.worker);
        sandbox
    } else {
        Command::new(&config.worker)
    };
    command
        .arg(config.log2_size.to_string())
        .arg(ready)
        .arg(proof);
    command
}

fn wait_for_ready(child: &mut Child, ready: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        if ready.is_file() {
            return Ok(());
        }
        if let Some(status) = child.try_wait()? {
            return Err(format!("worker exited before readiness with {status}").into());
        }
        if Instant::now() >= deadline {
            stop(child);
            return Err("worker readiness timed out".into());
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn wait_for_exit(
    child: &mut Child,
    timeout: Duration,
) -> Result<ExitStatus, Box<dyn std::error::Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            stop(child);
            return Err("worker timed out".into());
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn stop(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn fresh_seed() -> Result<u64, Box<dyn std::error::Error>> {
    let mut bytes = [0u8; 8];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn verify_proof(
    log2_size: u32,
    seed: u64,
    path: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() == 0 || metadata.len() > MAX_PROOF_BYTES {
        return Err(format!("proof size {} is outside the allowed range", metadata.len()).into());
    }

    let bytes = std::fs::read(path)?;
    let bundle = deserialize_bundle(&bytes)?;
    let setup = Blake3Setup::new(1usize << log2_size);
    let blocks = generate_compressions(log2_size, seed);
    let witness = setup.generate_witness_packed(&blocks);
    let (expected, _) = pcs::commit(&witness, &setup.pcs_params);

    if bundle.commitment.root != expected.root
        || bundle.commitment.params.m != setup.pcs_params.m
        || bundle.commitment.params.log_inv_rate != setup.pcs_params.log_inv_rate
        || bundle.commitment.params.log_batch_size != setup.pcs_params.log_batch_size
        || bundle.commitment.params.profile != setup.pcs_params.profile
    {
        return Err("proof commitment does not match the trusted BLAKE3 witness".into());
    }

    let mut challenger = FsChallenger::new(DOMAIN);
    setup
        .verify(&bundle.commitment, &bundle.proof, &mut challenger)
        .map_err(|error| format!("trusted verifier rejected proof: {error:?}"))?;
    Ok(bytes.len())
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

fn write_results(
    config: &Config,
    trials: &[Trial],
    best: &Trial,
    throughput: f64,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let best_seconds = best.seconds;
    let proof_bytes = best.proof_bytes;
    let score = ScoreFile {
        score: throughput,
        metrics: ScoreMetrics {
            trial_seconds: trials.iter().map(|trial| trial.seconds).collect(),
            batch_size,
            threads: config.threads,
            proof_bytes,
            verified: true,
        },
    };
    let mut score_file = File::create(&config.score)?;
    serde_json::to_writer_pretty(&mut score_file, &score)?;
    writeln!(score_file)?;

    let trial_lines = trials
        .iter()
        .enumerate()
        .map(|(index, trial)| format!("- Run {}: `{:.9} s`", index + 1, trial.seconds))
        .collect::<Vec<_>>()
        .join("\n");
    let summary = format!(
        concat!(
            "# Flock BLAKE3 benchmark\n\n",
            "- Batch: `{batch_size}`\n- Threads: `{threads}`\n",
            "- Score: **{throughput:.3} compressions/second**\n",
            "- Best: `{best:.9} s`\n- Proof: `{proof_bytes}` bytes\n",
            "- Trusted verification: `passed`\n\n## Trials\n\n{trial_lines}\n"
        ),
        batch_size = batch_size,
        threads = config.threads,
        throughput = throughput,
        best = best_seconds,
        proof_bytes = proof_bytes,
        trial_lines = trial_lines,
    );
    std::fs::write(&config.summary, summary)?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let worker = args.next().ok_or("missing WORKER")?.into();
    let scratch = args.next().ok_or("missing SCRATCH")?.into();
    let score = args.next().ok_or("missing SCORE")?.into();
    let summary = args.next().ok_or("missing SUMMARY")?.into();
    let log2_size: u32 = args.next().ok_or("missing LOG2")?.parse()?;
    let threads: usize = args.next().ok_or("missing THREADS")?.parse()?;
    let runs: usize = args.next().ok_or("missing RUNS")?.parse()?;
    let sandbox_profile = args.next().map(PathBuf::from);
    if args.next().is_some() || !(8..=20).contains(&log2_size) || threads == 0 || runs == 0 {
        return Err(concat!(
            "usage: flock_benchmark_harness WORKER SCRATCH SCORE SUMMARY ",
            "LOG2 THREADS RUNS [SANDBOX_PROFILE]"
        )
        .into());
    }
    Ok(Config {
        worker,
        scratch,
        score,
        summary,
        log2_size,
        threads,
        runs,
        sandbox_profile,
    })
}
