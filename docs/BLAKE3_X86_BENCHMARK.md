# Flock BLAKE3 x86_64 benchmark

The benchmark has one untrusted process and one trusted process.

- The **candidate prover** links the solver-editable Flock source, receives a
  fresh private block-set seed, produces one BLAKE3 proof, writes it, and exits.
- The **trusted driver/verifier** is a committed x86_64 ELF binary built from
  reviewed source commit `7b3c050dcc07ab9945899947b4c3fcf974fd21b8`. That
  commit retains the original Flock verifier and imports from upstream commit
  `85fc0e7cc002e7ca4dffdff805ba89976e9a5293`. It owns the private input, timer,
  verification, and score file.

The complete visual review is
[blake3-benchmark-flow.html](blake3-benchmark-flow.html). It includes the
timing boundary, trust boundary, function inventory, failure behavior, and the
GitHub Actions/Hilbert handoff.

## Ranked contract

- Runner: dedicated c7i.4xlarge (Intel Sapphire Rapids), 16 vCPU and 32 GB
  memory, Ubuntu 24.04 LTS x86_64, labeled `x86-16c-32gb`
- Work: 2^18 independent BLAKE3 compressions per proof
- Default Rayon threads: every vCPU reported by `nproc`
- Machine warm-up: 20 private, timed, verified proofs discarded from scoring
- Measurement: 100 private, timed, verified proofs
- Score: `262,144 / median(measured_seconds)`; higher is better
- Warm-up: one seed-independent `prove_fast` before each trial is ready
- Timed interval: sending the fresh seed through safe capture of the published
  proof bytes, including input generation, serialization, file publication,
  and the trusted bounded read
- Correctness: the fixed trusted code reconstructs the input and witness,
  checks the full PCS commitment, and verifies every proof
- Toolchain: Rust 1.97.0 with `-C target-cpu=native`

The private seed expands deterministically to all 262,144 test blocks. It does
not enter the candidate process until the trusted binary starts the clock.
The protected worker wrapper publishes only by writing a temporary proof and
renaming it onto the final path; it does not rely on solver-editable file I/O.
After opening the final path without following symlinks, requiring a regular
file, and making a bounded copy whose length remains stable, the harness stops
the timer and kills/reaps the worker. Trusted decoding and verification then
run on that immutable copy before a score is written. Any setup, execution,
capture, decoding, commitment, or verification failure exits nonzero without
inventing a score.

## Measured baseline

The unmodified prover scores a mean of **381,080 verified BLAKE3 compressions
per second** on the official runner, with a run-to-run coefficient of variation
of 0.600% across 5 full ranked sessions. Treat differences smaller than about
1% as noise.

Most of a trial is not scored. A trial costs roughly 6.6 seconds of wall time,
of which about 2.2 seconds is trusted verification that runs after the timer
stops; the remainder is worker startup, the fixed-seed warm-up proof, teardown,
and the scratch wipe. A complete 120-trial session takes about 10.4 minutes;
the workflow's 25-minute timeout also covers setup and the candidate build.

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

Ranked setup verifies `benchmark-tools/trusted/SHA256SUMS` before it builds
anything, and `benchmark.sh` verifies the checksum again immediately before
execution. The checksum is the only authenticity control: Linux ELF binaries
carry no equivalent of an ad-hoc code signature.

Current SHA-256:
`5ad0acfa59a6f3415061b0536d401075b7e7c71da5ec1e5d3d8784bd81d68798`.

The binary is reviewable through `benchmark-tools/harness/src/main.rs`. Benchmark
authors regenerate it—not solvers—with:

```sh
./benchmark-tools/build-trusted-verifier.sh
```

That script has only three jobs: create/check the literal reviewed-source
worktree, build its declared harness target from inside the locked workspace,
and replace the committed binary plus SHA-256. It uses Rust 1.97.0 and the
conservative floor `target-cpu=x86-64-v3` with `+pclmulqdq,+aes`. Those two
features are requested explicitly because the AVX2 baseline does not imply
them, and without carry-less multiply every GF(2^128) multiply falls back to a
bit-by-bit loop, which costs the verifier about 43 seconds per trial instead of
2.2. The subshell leaves the caller in its original directory, and the ranked
workflow never runs this author-only script.

## Setup behavior

`setup.sh` follows the hardened bootstrap pattern from `quantum_ecc_add`:

- fail early unless the host is Linux on x86_64;
- require Git, `sha256sum`, `curl`, and bubblewrap for ranked runs;
- locate and execute-check a C compiler from `CC`, `cc`, `gcc`, or `clang`,
  which Cargo and the `cc` crate need;
- install Rustup over TLS when it is absent;
- install exact Rust 1.97.0 only when that toolchain is missing;
- retry the locked Cargo fetch with explicit network retry/timeouts;
- compile the candidate with `--locked --offline` after the cache is populated.

The script is idempotent. Once prerequisites and Cargo artifacts exist, reruns
perform integrity checks and an up-to-date build without reinstalling them.

## Local smoke test

```sh
# First run only:
./setup.sh

# Rebuild the current prover, then run:
BLAKE3_LOG2=8 BLAKE3_THREADS=1 BLAKE3_WARMUP_RUNS=0 BLAKE3_RUNS=2 ./benchmark.sh
```

`benchmark.sh` performs a locked, offline candidate rebuild before invoking the
trusted verifier. Cargo reuses unchanged artifacts, and compilation is never
inside a trial timer. Run `setup.sh` again only when toolchain, dependency, or
machine prerequisites need repair.

The ranked workflow sets `FLOCK_REQUIRE_SANDBOX=1`. Local runs warn and proceed
unsandboxed when `bwrap` is unavailable.

## GitHub Actions and Hilbert

The workflow follows Hilbert's `github-actions-benchmark-author-guide.md`:

- `workflow_dispatch` is the only trigger;
- checkout uses the exact `${{ github.sha }}` from a clean checkout;
- the Setup and Benchmark steps match `benchmark.json`;
- the exact root `score.json` is uploaded even though diagnostics are separate;
- failures do not produce a trusted score.

Hilbert constructs the candidate commit from the current baseline by replacing
only `editablePaths`, dispatches this workflow, reads the score artifact, and
promotes only the exact scored commit. Do not manually merge submission PRs.

Solver pull requests also run `.github/workflows/lint.yml`, which checks
formatting, Clippy, and shell and JSON syntax on GitHub-hosted runners. It
never touches the benchmark instance, which stays free for ranked jobs.

Install the matching Hilbert GitHub App before import:

- development: <https://github.com/apps/yukon-eigen/installations/new>
- production: <https://github.com/apps/yukon-autoresearch/installations/new>

The self-hosted runner executes untrusted native code. Keep it dedicated,
ephemeral where possible, free of unrelated credentials, and restricted by the
bubblewrap sandbox the harness builds around every worker.

`RAYON_NUM_THREADS` configures the default prover but is not a hard CPU
quota—the editable source could use a different thread pool. Ranked fairness
therefore comes from running every candidate alone on the same dedicated
instance.
