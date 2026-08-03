// r273 no-op probe — lane-warm submission candidate.
//
// Deliberately inert by construction:
//  * `-p flock-benchmark-worker` builds flock-core only as a lib dependency,
//    so this bin target is never compiled by benchmark.sh's candidate build.
//  * Even if it were, the body is gated on FLOCK_R273_PROBE, which nothing in
//    the worker or the trusted verifier ever sets — bit-exact dead code.
//  * std-only: no new dependencies, no Cargo.lock delta, offline --locked safe.
//
// Purpose: keep the submission lane occupied with a verified-safe row while the
// static-B byte-move candidate awaits its two gating disclosures
// (kernels/aarch64.rs fused body + aarch64_bstatic_gen.rs:351 table coverage)
// in the next iteration's first reads.
fn main() {
    if std::env::var_os("FLOCK_R273_PROBE").is_some() {
        eprintln!("r273 no-op probe: never reached on the ranked path");
    }
}
