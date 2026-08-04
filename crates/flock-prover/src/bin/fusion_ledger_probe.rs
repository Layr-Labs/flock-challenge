//! Fused-layout byte ledger for the ranked inner-commit lincheck fold.
//!
//! Model-only measurement. Re-derives the transpose-write / fold-read byte
//! counts at the scored ranked geometry (m=32, log_batch_size=6, 32768
//! stripes) from the pitch constants measured by `pitch_shrink_probe`
//! (transpose writes pitch64*32768 = 481.97 MiB; the padded fold re-reads
//! pitch8*32768 = 481.75 MiB), and reports what the transpose+fold fusion —
//! folding each stripe into a thread-local accumulator inside the transpose
//! task (common.rs:446-463) instead of materializing z_lincheck — removes
//! from the DRAM ledger.
//!
//! Links no prover code; pure geometry. This is a BYTE ledger for the
//! submission note, not a timing: this host is x86-64 and cannot predict the
//! Apple M-series score (AGENTS.md §1).

fn main() {
    let stripes: u64 = 32768;
    let write_mib: f64 = 481.97; // transpose write, pitch64 * stripes (measured)
    let read_mib: f64 = 481.75; // padded fold re-read, pitch8 * stripes (measured)
    let mib: f64 = (1u64 << 20) as f64;

    let write_bytes = write_mib * mib;
    let read_bytes = read_mib * mib;
    let current = write_bytes + read_bytes;

    // Candidate fused layout: fold consumes the stripe while it is
    // register/cache-resident inside the transpose task; only the per-stripe
    // k-entry F128 accumulator survives to the combine pass.
    let k: u64 = 32;
    let combine_bytes = stripes * k * 16;
    let fused = write_bytes + combine_bytes as f64;
    let saved = read_bytes - combine_bytes as f64;

    let ledger_gib = 15.0 * (1u64 << 30) as f64; // ~15 GiB timed-region ledger (AGENTS.md §7)

    println!("FUSED-LAYOUT LEDGER m=32 (model on x86 host — NOT a timing)");
    println!("stripes={stripes}");
    println!("transpose_write={write_bytes:.0} B ({write_mib:.2} MiB)  pitch64={:.1} B/row", write_bytes / stripes as f64);
    println!("fold_read={read_bytes:.0} B ({read_mib:.2} MiB)  pitch8={:.1} B/row", read_bytes / stripes as f64);
    println!("current_double_touch={current:.0} B ({:.2} MiB)", current / mib);
    println!("fused_layout={fused:.0} B ({:.2} MiB)  [k={k} F128/stripe, combine={combine_bytes} B]", fused / mib);
    println!("bytes_removed={saved:.0} B ({:.2} MiB)", saved / mib);
    println!("savings={:.3}% of the ~15 GiB ledger", saved / ledger_gib * 100.0);
    println!("fold_byte_ratio_to_transpose={:.4}", read_bytes / write_bytes);
}
