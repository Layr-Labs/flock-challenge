# ITER3 EVIDENCE LEDGER — angel lane (timed-window census + preflight repair)

Direction: the ready/seed boundary and the sandbox layer, not the kernel
internals. Two prior directions (mechanism census, build-toolchain) never
examined how the ranked worker's timed window is actually reached on this
box.

## A. benchmark.sh has NEVER run a worker on this box: bwrap dies first

Every `./benchmark.sh` invocation ends in `Error: "worker exited before
readiness with exit status: 1"` (reproduced twice this iteration, log2=16).
Root cause captured by running the harness's exact bwrap policy manually
(harness worker_command, benchmark-tools/harness/src/main.rs:216-240):

    bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted

`--unshare-net` cannot configure its in-namespace loopback (no CAP_NET_ADMIN
in the user namespace on this host), so bwrap exits 1 BEFORE exec'ing the
worker. The harness's `wait_for_ready` then reports exactly the misleading
line above. This is environmental; it cannot be fixed from the candidate
lane and it is invisible on the ranked runner.

Consequence: any candidate verdict produced by `./benchmark.sh` on this box
is a sandbox verdict, not a candidate verdict. Preflights MUST invoke the
trusted verifier directly without the SANDBOX_SCRATCH argument:

    ./benchmark-tools/trusted/flock_benchmark_verifier \
      ./target/challenge-candidate/challenge/flock-benchmark-worker \
      /tmp/wt-h /tmp/wt-h.json /tmp/wt-h.md 16 16 1 2

(The candidate binary lives at target/challenge-candidate/challenge/…, built
by benchmark.sh's own cargo invocation; a plain `cargo build` writes to
target/challenge and is not what the verifier should run.)

## B. Lane mutation this iteration: eval_sk_at_vks memoized

crates/flock-core/src/pcs/ligerito.rs: `eval_sk_at_vks(log_n)` previously
allocated a fresh Vec per call (finding ITER1#4). Now: signature
`Vec<F128>` -> `&'static [F128]`, 33-slot `OnceLock<Vec<F128>>` array
indexed by log_n, fresh-compute fallback for log_n >= 33. Grep confirms no
call site mutates the returned table (`mut sks_vks` occurs only inside the
function body; 16 call sites consume by shared borrow). Lane builds clean on
pinned 1.97.0 (`Finished challenge profile in 10.09s`).

## C. Verified at log2=16 via the trusted verifier (direct, no sandbox)

warmup_trial=1/1 verified=true; measured_trial=1/2 verified=true
(score 59475.302 cps), measured_trial=2/2 verified=true; final
score=59434.379 compressions_per_second (score file /tmp/wt-h.json).
Pre-edit baseline from ITER2: 59268.094 cps. Delta +0.28% is noise-level —
the memoization alone cannot clear the +1% gate; it is a hygiene edit that
removes per-call heap traffic from the open phase and composes with other
packs.

## D. log2=18 preflight remains dead on this box

Trusted verifier (1 warmup + 1 measured, threads=16, direct, no sandbox)
hung with zero output for 30s and was killed. Consistent with ITER2
findings 9-11 (size-17/18 fallback bugs in crown's own code); re-testing
17/18 locally is wasted budget. Only log2<=16 is a usable verified
preflight.

## E. The ready/seed boundary is already exhausted by the crown itself

crates/flock-prover/src/r1cs_hashes/blake3.rs:3996-4115: the crown's
prove_fast performs 11 untimed EXTRA_WARMUP proves (45s budget), hoists the
generator verification ahead of them, then arms a seed_pipe thread that
reads stdin, proves, and publishes the proof file directly (with a
try_adopt fallback for the main thread). Any "move seed-independent work
before ready" seam is therefore already taken by the promoted source; only
in-loop work deletions inside the timed prove remain available.
