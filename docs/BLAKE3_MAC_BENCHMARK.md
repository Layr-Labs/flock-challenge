# Flock BLAKE3 Mac benchmark

The benchmark has one untrusted process and one trusted process.

- The **candidate prover** links the solver-editable Flock source, receives a
  fresh private block-set seed, produces one BLAKE3 proof, writes it, and exits.
- The **trusted driver/verifier** is a committed arm64 binary built from reviewed
  source commit `44844f05847e381b094bf04fb19aaec0223ce801`. That
  commit retains the original Flock verifier and imports from upstream commit
  `85fc0e7cc002e7ca4dffdff805ba89976e9a5293`. It owns the private input, timer,
  verification, and score file.

The complete visual review is
[blake3-benchmark-flow.html](blake3-benchmark-flow.html). It includes the
timing boundary, trust boundary, function inventory, failure behavior, and the
GitHub Actions/Yukon handoff.

## Ranked contract

- Runner: dedicated Apple M4 Pro runner labeled `m4-pro`
- Work: 2^18 independent BLAKE3 compressions per proof
- Default Rayon threads: performance-core count
- Machine warm-up: 20 private, timed, verified proofs discarded from scoring
- Measurement: 100 private, timed, verified proofs
- Score: `262,144 / median(measured_seconds)`; higher is better
- Warm-up: one seed-independent `prove_fast` before each trial is ready
- Timed interval: sending the fresh seed through prover exit, including input
  generation and serialization
- Correctness: the fixed trusted code reconstructs the input and witness,
  checks the full PCS commitment, and verifies every proof
- Toolchain: Rust 1.97.0 with `-C target-cpu=native`

The private seed expands deterministically to all 262,144 test blocks. It does
not enter the candidate process until the trusted binary starts the clock.
Trusted verification runs after the timer stops and before a score is written.
Any setup, execution, decoding, commitment, or verification failure exits
nonzero without inventing a score.

## Editable surface

`benchmark.json` lets solvers replace only:

- `crates/flock-core/src`
- `crates/flock-prover/src`

This includes the prover and all performance-sensitive field, NTT, Merkle,
PCS, zerocheck, lincheck, witness, and BLAKE3 code. It also includes Flock's
ordinary verifier source, but that source is never trusted: the official
binary was linked entirely against the immutable original checkout.

The manifests, dependencies, prover wrapper, input generator, harness source,
committed verifier binary, checksum, shell scripts, workflow, and `score.json`
path are not editable.

## Trusted binary

Ranked setup verifies `benchmark-tools/trusted/SHA256SUMS` and the macOS code
signature, then builds only the candidate prover. `benchmark.sh` verifies the
checksum again immediately before execution.

Current SHA-256:
`c61a34759f69bd5863fb252ece81845440be5f9ed2c38f7f970c2ac224ca5127`.

The binary is reviewable through `benchmark-tools/harness/src/main.rs`. Benchmark
authors regenerate it—not solvers—with:

```sh
./benchmark-tools/build-trusted-verifier.sh
```

That script has only three jobs: create/check the literal reviewed-source
worktree, build its declared harness target from inside the locked workspace,
and replace the committed binary plus SHA-256. It uses Rust 1.97.0 and
conservative `target-cpu=apple-m1`. The subshell leaves the caller in its
original directory, and the ranked workflow never runs this author-only script.

## Setup behavior

`setup.sh` follows the hardened bootstrap pattern from `quantum_ecc_add`:

- fail early unless the host is Apple Silicon macOS;
- require Git, checksum/signing tools, and Seatbelt for ranked runs;
- locate and execute-check Xcode's Clang linker, requesting the Command Line
  Tools installer interactively when they are absent;
- install Rustup over TLS when it is absent;
- install exact Rust 1.97.0 only when that toolchain is missing;
- retry the locked Cargo fetch with explicit network retry/timeouts;
- compile the candidate with `--locked --offline` after the cache is populated.

The script is idempotent. Once prerequisites and Cargo artifacts exist, reruns
perform integrity checks and an up-to-date build without reinstalling them.

## Local smoke test

```sh
./setup.sh
BLAKE3_LOG2=8 BLAKE3_THREADS=1 BLAKE3_WARMUP_RUNS=0 BLAKE3_RUNS=2 ./benchmark.sh
```

The ranked workflow sets `FLOCK_REQUIRE_SANDBOX=1`. Local runs warn and proceed
without Seatbelt when `sandbox-exec` is unavailable.

## GitHub Actions and Yukon

The workflow follows Yukon's `github-actions-benchmark-author-guide.md`:

- `workflow_dispatch` is the only trigger;
- checkout uses the exact `${{ github.sha }}` from a clean checkout;
- the Setup and Benchmark steps match `benchmark.json`;
- the exact root `score.json` is uploaded even though diagnostics are separate;
- failures do not produce a trusted score.

Yukon constructs the candidate commit from the current baseline by replacing
only `editablePaths`, dispatches this workflow, reads the score artifact, and
promotes only the exact scored commit. Do not manually merge submission PRs.

Install the matching Yukon GitHub App before import:

- development: <https://github.com/apps/yukon-eigen/installations/new>
- production: <https://github.com/apps/yukon-autoresearch/installations/new>

The self-hosted runner executes untrusted native code. Keep it dedicated,
ephemeral where possible, free of unrelated credentials, and restricted by the
Seatbelt profile in `benchmark.sh`.

`RAYON_NUM_THREADS` configures the default prover but is not a hard CPU
quota—the editable source could use a different thread pool. Ranked fairness
therefore comes from running every candidate alone on the same dedicated Mac.
