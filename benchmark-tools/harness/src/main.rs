//! Trusted driver, verifier, timer, and score writer for the BLAKE3 challenge.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
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
const P10_PERCENTILE: f64 = 0.10;
const SCORE_PERCENTILE: f64 = 0.50;
const P90_PERCENTILE: f64 = 0.90;
// Sampled 2^18 proofs are about 436-438 kB. Keep the reviewed 500 kB bound.
const MAX_PROOF_BYTES: u64 = 500_000;

struct Config {
    worker: PathBuf,
    scratch: PathBuf,
    score: PathBuf,
    summary: PathBuf,
    log2_size: u32,
    threads: usize,
    warmup_runs: usize,
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
    warmup_trial_seconds: Vec<f64>,
    trial_seconds: Vec<f64>,
    p10_seconds: f64,
    median_seconds: f64,
    aggregate_compressions_per_second: f64,
    p90_p10_latency_ratio: f64,
    batch_size: usize,
    warmup_runs: usize,
    measured_runs: usize,
    threads: usize,
    proof_bytes: usize,
    verified: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let batch_size = 1usize << config.log2_size;

    let total_runs = config
        .warmup_runs
        .checked_add(config.runs)
        .ok_or("trial count overflow")?;
    let mut warmup_trials = Vec::with_capacity(config.warmup_runs);
    let mut measured_trials = Vec::with_capacity(config.runs);
    for run in 1..=total_runs {
        let trial = run_trial(&config, run)?;
        if run <= config.warmup_runs {
            log_trial("warmup", run, config.warmup_runs, &trial, batch_size)?;
            warmup_trials.push(trial);
        } else {
            log_trial(
                "measured",
                run - config.warmup_runs,
                config.runs,
                &trial,
                batch_size,
            )?;
            measured_trials.push(trial);
        }
    }

    let median_seconds = percentile_seconds(&measured_trials, SCORE_PERCENTILE)?;
    let throughput = batch_size as f64 / median_seconds;
    write_results(
        &config,
        &warmup_trials,
        &measured_trials,
        throughput,
        batch_size,
    )?;
    println!("score={throughput:.3} compressions_per_second");
    Ok(())
}

fn log_trial(
    phase: &str,
    index: usize,
    total: usize,
    trial: &Trial,
    batch_size: usize,
) -> std::io::Result<()> {
    let throughput = batch_size as f64 / trial.seconds;
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "{phase}_trial={index}/{total} trial_score={throughput:.3} \
         compressions_per_second seconds={:.9} verified=true included_in_score={}",
        trial.seconds,
        phase == "measured",
    )?;
    stdout.flush()
}

fn percentile_seconds(
    trials: &[Trial],
    percentile: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    if trials.is_empty() || !(0.0..=1.0).contains(&percentile) {
        return Err("invalid percentile input".into());
    }
    let mut seconds = trials.iter().map(|trial| trial.seconds).collect::<Vec<_>>();
    if seconds
        .iter()
        .any(|seconds| !seconds.is_finite() || *seconds <= 0.0)
    {
        return Err("trial duration must be finite and positive".into());
    }
    seconds.sort_by(f64::total_cmp);
    let rank = (seconds.len() - 1) as f64 * percentile;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;
    Ok(seconds[lower] + (seconds[upper] - seconds[lower]) * fraction)
}

fn run_trial(config: &Config, run: usize) -> Result<Trial, Box<dyn std::error::Error>> {
    reset_scratch(&config.scratch)?;
    let ready = config.scratch.join(format!("run-{run}.ready"));
    let proof = config.scratch.join(format!("run-{run}.proof"));

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
    let captured_proof = match wait_for_proof(&mut child, &proof, RUN_TIMEOUT) {
        Ok(bytes) => bytes,
        Err(error) => {
            stop(&mut child);
            return Err(error);
        }
    };
    let seconds = start.elapsed().as_secs_f64();

    // Proof availability, rather than process teardown, is the scored boundary.
    // Reap the untrusted worker before verifying the immutable trusted copy.
    stop(&mut child);
    let proof_bytes = verify_proof(config.log2_size, seed, &captured_proof)?;
    reset_scratch(&config.scratch)?;
    Ok(Trial {
        seconds,
        proof_bytes,
    })
}

fn reset_scratch(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if path.as_os_str().is_empty() || path == Path::new("/") {
        return Err("refusing unsafe scratch path".into());
    }
    if path.exists() {
        std::fs::remove_dir_all(path)?;
    }
    std::fs::create_dir_all(path)?;
    Ok(())
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

fn wait_for_proof(
    child: &mut Child,
    proof: &Path,
    timeout: Duration,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(bytes) = capture_proof_if_ready(proof)? {
            return Ok(bytes);
        }
        if let Some(status) = child.try_wait()? {
            // The rename and process exit may become observable in either
            // order, so make one final capture attempt before rejecting.
            return capture_proof_if_ready(proof)?.ok_or_else(|| {
                format!("worker exited before publishing a proof with {status}").into()
            });
        }
        if Instant::now() >= deadline {
            return Err("worker proof timed out".into());
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn capture_proof_if_ready(path: &Path) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot open final proof: {error}").into()),
    };

    let initial = file.metadata()?;
    if !initial.file_type().is_file() {
        return Err("final proof path is not a regular file".into());
    }
    if initial.len() == 0 || initial.len() > MAX_PROOF_BYTES {
        return Err(format!("proof size {} is outside the allowed range", initial.len()).into());
    }

    let mut bytes = Vec::with_capacity(initial.len() as usize);
    (&mut file)
        .take(MAX_PROOF_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_PROOF_BYTES {
        return Err(format!(
            "captured proof size {} is outside the allowed range",
            bytes.len()
        )
        .into());
    }

    let final_metadata = file.metadata()?;
    if final_metadata.len() != initial.len() || final_metadata.len() != bytes.len() as u64 {
        return Err("proof file changed while the trusted harness captured it".into());
    }
    Ok(Some(bytes))
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
    bytes: &[u8],
) -> Result<usize, Box<dyn std::error::Error>> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_PROOF_BYTES {
        return Err(format!("proof size {} is outside the allowed range", bytes.len()).into());
    }

    let bundle = deserialize_bundle(bytes)?;
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
    warmup_trials: &[Trial],
    measured_trials: &[Trial],
    throughput: f64,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let p10_seconds = percentile_seconds(measured_trials, P10_PERCENTILE)?;
    let median_seconds = percentile_seconds(measured_trials, SCORE_PERCENTILE)?;
    let p90_seconds = percentile_seconds(measured_trials, P90_PERCENTILE)?;
    let total_seconds = measured_trials
        .iter()
        .map(|trial| trial.seconds)
        .sum::<f64>();
    let aggregate_throughput = batch_size as f64 * measured_trials.len() as f64 / total_seconds;
    let proof_bytes = warmup_trials
        .iter()
        .chain(measured_trials)
        .map(|trial| trial.proof_bytes)
        .max()
        .ok_or("no verified trials")?;
    let score = ScoreFile {
        score: throughput,
        metrics: ScoreMetrics {
            warmup_trial_seconds: warmup_trials.iter().map(|trial| trial.seconds).collect(),
            trial_seconds: measured_trials.iter().map(|trial| trial.seconds).collect(),
            p10_seconds,
            median_seconds,
            aggregate_compressions_per_second: aggregate_throughput,
            p90_p10_latency_ratio: p90_seconds / p10_seconds,
            batch_size,
            warmup_runs: warmup_trials.len(),
            measured_runs: measured_trials.len(),
            threads: config.threads,
            proof_bytes,
            verified: true,
        },
    };
    let mut score_file = File::create(&config.score)?;
    serde_json::to_writer_pretty(&mut score_file, &score)?;
    writeln!(score_file)?;

    let summary = format!(
        concat!(
            "# Flock BLAKE3 benchmark\n\n",
            "- Batch: `{batch_size}`\n- Threads: `{threads}`\n",
            "- Warm-up trials: `{warmup_runs}` (verified, not scored)\n",
            "- Measured trials: `{measured_runs}`\n",
            "- Score: **{throughput:.3} compressions/second**\n",
            "- P10 latency: `{p10_seconds:.9} s`\n",
            "- Median latency: `{median_seconds:.9} s`\n",
            "- Aggregate throughput: `{aggregate_throughput:.3} compressions/second`\n",
            "- P90/P10 latency ratio: `{dispersion:.6}`\n",
            "- Largest proof: `{proof_bytes}` bytes\n",
            "- Trusted verification: `passed`\n"
        ),
        batch_size = batch_size,
        threads = config.threads,
        warmup_runs = warmup_trials.len(),
        measured_runs = measured_trials.len(),
        throughput = throughput,
        p10_seconds = p10_seconds,
        median_seconds = median_seconds,
        aggregate_throughput = aggregate_throughput,
        dispersion = p90_seconds / p10_seconds,
        proof_bytes = proof_bytes,
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
    let warmup_runs: usize = args.next().ok_or("missing WARMUP_RUNS")?.parse()?;
    let runs: usize = args.next().ok_or("missing RUNS")?.parse()?;
    let sandbox_profile = args.next().map(PathBuf::from);
    if args.next().is_some() || !(8..=20).contains(&log2_size) || threads == 0 || runs == 0 {
        return Err(concat!(
            "usage: flock_benchmark_harness WORKER SCRATCH SCORE SUMMARY ",
            "LOG2 THREADS WARMUP_RUNS RUNS [SANDBOX_PROFILE]"
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
        warmup_runs,
        runs,
        sandbox_profile,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use super::{
        MAX_PROOF_BYTES, Trial, capture_proof_if_ready, percentile_seconds, stop, wait_for_proof,
    };

    static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "flock-harness-test-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn trials(seconds: &[f64]) -> Vec<Trial> {
        seconds
            .iter()
            .map(|seconds| Trial {
                seconds: *seconds,
                proof_bytes: 1,
            })
            .collect()
    }

    #[test]
    fn percentile_uses_linear_interpolation() {
        let samples = trials(&[4.0, 1.0, 3.0, 2.0]);
        assert_eq!(percentile_seconds(&samples, 0.10).unwrap(), 1.3);
        assert_eq!(percentile_seconds(&samples, 0.50).unwrap(), 2.5);
        assert_eq!(percentile_seconds(&samples, 0.90).unwrap(), 3.7);
    }

    #[test]
    fn percentile_rejects_invalid_durations() {
        assert!(percentile_seconds(&[], 0.10).is_err());
        assert!(percentile_seconds(&trials(&[0.0]), 0.10).is_err());
        assert!(percentile_seconds(&trials(&[f64::NAN]), 0.10).is_err());
    }

    #[test]
    fn proof_capture_waits_for_publication_then_copies_bytes() {
        let temp = TempDir::new();
        let proof = temp.join("proof");
        assert!(capture_proof_if_ready(&proof).unwrap().is_none());

        std::fs::write(&proof, b"complete proof").unwrap();
        assert_eq!(
            capture_proof_if_ready(&proof).unwrap(),
            Some(b"complete proof".to_vec())
        );
    }

    #[test]
    fn proof_capture_rejects_empty_and_oversized_files() {
        let temp = TempDir::new();
        let proof = temp.join("proof");

        std::fs::write(&proof, []).unwrap();
        assert!(capture_proof_if_ready(&proof).is_err());

        std::fs::File::create(&proof)
            .unwrap()
            .set_len(MAX_PROOF_BYTES + 1)
            .unwrap();
        assert!(capture_proof_if_ready(&proof).is_err());
    }

    #[test]
    fn proof_capture_rejects_symlinks_and_non_files() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new();
        let target = temp.join("target");
        let link = temp.join("link");
        std::fs::write(&target, b"proof").unwrap();
        symlink(&target, &link).unwrap();

        assert!(capture_proof_if_ready(&link).is_err());
        assert!(capture_proof_if_ready(&temp.0).is_err());
    }

    #[test]
    fn proof_publisher_helper() {
        let Some(path) = std::env::var_os("FLOCK_TEST_PROOF_PATH") else {
            return;
        };
        let temporary = PathBuf::from(&path).with_extension("tmp");
        std::fs::write(&temporary, b"published proof").unwrap();
        std::fs::rename(temporary, path).unwrap();
        std::thread::sleep(Duration::from_secs(10));
    }

    #[test]
    fn proof_publication_does_not_wait_for_worker_exit() {
        let temp = TempDir::new();
        let proof = temp.join("proof");
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::proof_publisher_helper"])
            .env("FLOCK_TEST_PROOF_PATH", &proof)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();

        let start = Instant::now();
        let bytes = wait_for_proof(&mut child, &proof, Duration::from_secs(5)).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(bytes, b"published proof");
        assert!(elapsed < Duration::from_secs(2));
        assert!(child.try_wait().unwrap().is_none());
        stop(&mut child);
    }
}
