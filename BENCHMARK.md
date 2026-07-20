# Flock BLAKE3 benchmark

This benchmark measures how quickly Flock can produce a valid proof for a
fixed batch of independent BLAKE3 compression operations on Apple Silicon.
Submissions may optimize almost all prover implementation code. A prebuilt,
checksum-pinned verifier controls the private inputs, timing, correctness
decision, and score.

For a visual version of this document, open
[`docs/blake3-benchmark-flow.html`](docs/blake3-benchmark-flow.html).

## Ranked contract

| Property | Ranked value |
| --- | --- |
| Work per timed proof | `2^18 = 262,144` BLAKE3 compressions |
| Machine warm-up | 20 private, verified trials discarded before scoring |
| Measured trials | 100, each in a fresh worker process |
| Worker warm-up | 1 fixed-seed, untimed proof inside every worker |
| Score | `262,144 / median(measured_trial_seconds)` |
| Unit | verified BLAKE3 compressions per second |
| Direction | higher is better |
| Correctness | every timed proof must pass the prebuilt verifier |
| Proof file limit | 500,000 bytes, enforced by the prebuilt verifier |
| Official runner | Apple M4 Pro, 48 GB unified memory, 10 performance cores |

The benchmark counts **BLAKE3 compression functions**, not complete
whole-message BLAKE3 hashes. The distinction matters when presenting the
score.

## What one operation contains

One generated compression input has this shape:

```text
(
  chaining_value: [u32; 8],
  message_block:  [u32; 16],
  counter:        u64,
  block_len:      64,
  flags:          11,
)
```

The chaining value, 64-byte message block, and counter are generated from a
private 64-bit trial seed. `block_len = 64` and `flags = 11` are fixed. The
same deterministic generator is compiled into both the protected worker
wrapper and the trusted verifier.

Each ranked run therefore performs:

- 20 timed, private, serialized, and verified machine warm-up proofs whose
  durations are recorded but excluded from scoring;
- 100 timed, private, serialized, and verified measured proofs;
- 26,214,400 scored compression operations in total;
- one additional fixed-seed, untimed proof inside each of the 120 workers.

## Scoring

For each trial, the trusted harness measures wall-clock time from immediately
before it writes the private seed to worker stdin until the worker exits
successfully:

```text
rank_seconds = median(measured_trial_seconds)
score = 262,144 / rank_seconds
```

The median uses linear interpolation at rank `(sample_count - 1) × 0.50`; for
100 measured trials this is the mean of the 50th and 51st sorted durations.
Setup, fixed-seed
warm-up, and trusted verification are outside the timed interval. Input
generation from the private seed, witness generation, commitment, proving,
serialization, proof-file writing, and process exit are inside it.

Proof size is a reported secondary metric, not part of the ranking formula.
At the ranked size, sampled valid proofs were approximately 436–438 kB.
500,000 bytes leaves measured headroom while bounding file and deserialization
work.

## Official hardware and directional local results

Official scores run in GitHub Actions on a dedicated self-hosted Mac with:

- Apple M4 Pro;
- 48 GB unified memory;
- 10 performance cores used by the default thread selection;
- arm64 macOS;
- runner label `m4-pro`;
- Rust 1.97.0;
- candidate builds using `-C target-cpu=native`.

The validated stability experiment used macOS 26.4 build 25E246. Across five
independent sessions, the unmodified candidate averaged approximately 483,866
verified compressions/s with 0.539% run-to-run CV. This is a
reference observation, not a guaranteed baseline: system version, thermals,
background load, and compiler output can move absolute throughput.

Local runs on another Apple Silicon Mac are useful for correctness and
directional optimization feedback. Compare performance changes on the same
quiet machine. Only the official M4 Pro runner determines the ranked score.

## Editable surface

Yukon accepts replacements only under the two `editablePaths` declared in
[`benchmark.json`](benchmark.json):

| Editable path | Optimization scope |
| --- | --- |
| `crates/flock-core/src/**` | Field arithmetic, NTTs, PCS, R1CS, Merkle operations, transcript/protocol implementation, architecture kernels, and other core code linked into the candidate. |
| `crates/flock-prover/src/**` | Witness generation, BLAKE3 R1CS implementation, proving orchestration, layouts, memory use, and prover-specific code. |

This intentionally provides a large optimization surface. The following are
outside it and cannot be supplied by a solver:

- `benchmark-tools/worker/**`;
- `benchmark-tools/harness/**`;
- `benchmark-tools/trusted/**`;
- Cargo manifests and `Cargo.lock`;
- `setup.sh` and `benchmark.sh`;
- `.github/workflows/benchmark-blake3-mac.yml`;
- `benchmark.json` and the score path.

The worker wrapper is protected, but it links the editable Flock source into
the candidate binary. The trusted verifier is a different binary containing
its own compiled-in pristine Flock code.

## Why the verifier is prebuilt

Building the verifier from a submission checkout would let editable imports
change the judge. Instead, the author-only
[`benchmark-tools/build-trusted-verifier.sh`](benchmark-tools/build-trusted-verifier.sh)
does the following before release:

1. Checks out the exact reviewed benchmark source commit in a detached
   worktree.
2. Refuses a dirty or unexpected checkout.
3. Builds `flock_benchmark_harness` from inside that worktree with the pinned
   Rust toolchain and conservative `target-cpu=apple-m1` target.
4. Copies the arm64 Mach-O to `benchmark-tools/trusted/`.
5. Writes its SHA-256 to
   [`benchmark-tools/trusted/SHA256SUMS`](benchmark-tools/trusted/SHA256SUMS).

Ranked setup never runs this author tool. `setup.sh` verifies the committed
binary's SHA-256 and code signature before building the candidate.
`benchmark.sh` checks SHA-256 again immediately before invoking it.

The committed verifier was built from reviewed benchmark commit
`7a6585a20adfd5eb38814a1587a3adb9fb7e838c`. Its underlying re-signed Flock
tree matches upstream Flock commit
`85fc0e7cc002e7ca4dffdff805ba89976e9a5293`.

## Harness and worker interaction

The trusted harness repeats the following sequence 120 times:

1. It creates private scratch paths and starts a fresh candidate worker under
   a macOS Seatbelt profile.
2. The worker creates `Blake3Setup`, computes one fixed-seed warm-up proof, and
   writes a ready file.
3. Only after readiness, the harness reads 8 bytes from `/dev/urandom`.
4. The harness starts its external clock and writes the decimal seed plus a
   newline to worker stdin.
5. The worker expands the seed into 262,144 inputs, runs `prove_fast`,
   serializes a `R1csProofBundleLigerito`, atomically writes the proof in
   scratch, and exits.
6. The harness stops timing after successful process exit.
7. The harness verifies the proof using its compiled-in pristine code.
8. It records the duration and proof size, then erases and recreates the entire
   writable scratch directory before the next worker.

After each proof passes verification, the harness immediately logs its phase,
index, duration, and individual `batch_size / trial_seconds` throughput. A
warm-up line is progress information only; only the measured population enters
the final median score.

The first 20 complete trials warm the machine and are excluded from ranking.
The next 100 are the measured population used for the median. Every proof in both
groups must verify; “warm-up” never means “unchecked.”

The seed line is the only request sent to the worker. The proof file is the
only candidate-produced object consumed by the trusted verifier. Worker stdout
and stderr are discarded and cannot report time or score.

Startup has a 5-minute deadline. Seed-dependent proving has a 15-minute
deadline. Timeout and trusted-side errors kill and reap the worker.

## Final verification

For every timed proof, the prebuilt verifier requires all of the following:

1. The worker exited successfully before the deadline.
2. The proof file is nonempty and below the trusted byte limit.
3. The file has the exact FLOCK magic, current version, and R1CS flavor.
4. Fixed-width bincode decoding consumes the complete file with no trailing
   bytes.
5. The verifier independently reconstructs the private compression inputs.
6. It generates the pristine packed witness and commits to it.
7. The submitted commitment root and every PCS parameter match that trusted
   commitment.
8. `Blake3Setup::verify` accepts the proof under the fixed
   `flock-bench-v0` Fiat–Shamir domain.

This commitment reconstruction is essential. It binds an otherwise valid
Flock protocol proof to the exact private benchmark witness.

If any warm-up or measured trial fails, the harness exits nonzero and does not
write a replacement score. `benchmark.sh` deletes stale score files before
execution, so a failed run cannot upload an earlier result as a new score.

## Worker sandbox

On ranked macOS runs, `benchmark.sh` requires `sandbox-exec` and launches only
the candidate worker under a generated Seatbelt profile. The worker:

- receives a cleared environment containing only `RAYON_NUM_THREADS` and
  `TMPDIR`;
- cannot use the network;
- cannot create child processes;
- cannot write outside its private scratch directory.

The trusted harness wipes scratch between workers. Candidate code therefore
cannot move setup or precomputation from the 20 discarded machine warm-ups
into the 100 measured processes through files.

The sandbox prevents the candidate from writing `score.json` or leaving a
descendant to finish work after the timed worker exits. It is not a complete VM
boundary. The self-hosted runner must remain dedicated and should contain no
unrelated secrets or credentials.

## Score file

The verifier uses typed Rust structs and Serde to write repository-root
`score.json` only after all trials pass:

```json
{
  "score": 484700.1738854293,
  "metrics": {
    "warmup_trial_seconds": [0.5351, 0.5344],
    "trial_seconds": [0.5338, 0.5362],
    "p10_seconds": 0.5346941128,
    "median_seconds": 0.5408374375,
    "aggregate_compressions_per_second": 484858.3396136077,
    "p90_p10_latency_ratio": 1.0220526782,
    "batch_size": 262144,
    "warmup_runs": 20,
    "measured_runs": 100,
    "threads": 10,
    "proof_bytes": 436107,
    "verified": true
  }
}
```

`score` is a finite number measured in verified BLAKE3 compressions per
second. `median_seconds` is the scored latency statistic. P10 latency,
aggregate throughput, and p90/p10 dispersion are diagnostics rather than
ranking inputs. `proof_bytes` is the largest accepted proof in the run and is reported
for visibility only. The arrays above are abbreviated; ranked output contains
all 20 warm-up and 100 measured durations.

GitHub Actions uploads the exact root `score.json` as the score artifact and
uploads `benchmark-results/` separately for diagnostics. Yukon reads the score
artifact, compares higher-is-better, and associates the result with the exact
submission commit it dispatched.

## Running locally

On an Apple Silicon Mac:

```bash
./setup.sh
./benchmark.sh
```

`setup.sh` checks macOS/arm64 prerequisites, installs or selects the pinned
Rust toolchain, verifies the trusted binary, fetches locked dependencies with
retries, and builds the candidate offline.

For a quick functional smoke test, lower the batch and trial count:

```bash
BLAKE3_LOG2=8 BLAKE3_WARMUP_RUNS=0 BLAKE3_RUNS=1 ./benchmark.sh
```

Environment overrides are for local diagnostics. The GitHub Actions workflow
sets the ranked values explicitly:

```text
BLAKE3_LOG2=18
BLAKE3_THREADS=auto
BLAKE3_WARMUP_RUNS=20
BLAKE3_RUNS=100
FLOCK_REQUIRE_SANDBOX=1
```

## GitHub Actions and Yukon

The production workflow is
`.github/workflows/benchmark-blake3-mac.yml` and is dispatch-only. It:

1. checks out the exact dispatched SHA with credentials disabled;
2. records runner hardware and software information;
3. runs `./setup.sh`;
4. runs `./benchmark.sh`;
5. publishes the Markdown summary;
6. uploads root `score.json` and a separate diagnostics artifact.

The workflow runs on `[self-hosted, m4-pro]`. Yukon owns submission archive
validation, editable-path enforcement, candidate commit construction, workflow
dispatch, score comparison, and promotion. The GitHub pull request remains the
durable record of the candidate diff, workflow run, and result.
