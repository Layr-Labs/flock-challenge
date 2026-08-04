//! warmup_cache_probe — locate and decode the persisted GPU warmup-latch cache.
//!
//! The warmup latch (gpu_commit.rs:1851 `warmup_and_decide`) decides GPU vs CPU
//! once per process, but the code carries a versioned cache magic
//! (`WARMUP_CACHE_MAGIC_V3 = 0x464C_4B5F_574C_4333` — "FLK_WLC3", gpu_commit.rs
//! ~4508) and a "canonical reprime kill switch" that returns to the incumbent
//! V2 cache — implying the latch verdict is *persisted across processes*.
//!
//! If a stale cached "CPU" verdict survives between submissions, then the whole
//! 0.26 s floor family (b9bbbd8/05aa734/5782387/7ddfb7e — four unrelated kernel
//! families all landing in 0.258–0.279) could be one poisoned cache entry, not
//! four independent kernel failures. This probe scans the usual cache roots for
//! the WLC3/WLC2 magic byte patterns and reports which file carries the verdict
//! and its raw bytes. Pure std; no flock-core deps. Driver-side only.
//!
//! Usage: cargo run --release --bin warmup_cache_probe -- [root...]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// "FLK_WLC2" / "FLK_WLC3" in little-endian byte order (as a u64 would sit in a file).
const WLC2_LE: [u8; 8] = [0x32, 0x43, 0x4C, 0x57, 0x5F, 0x4B, 0x4C, 0x46];
const WLC3_LE: [u8; 8] = [0x33, 0x43, 0x4C, 0x57, 0x5F, 0x4B, 0x4C, 0x46];
// Big-endian spellings too, for safety.
const WLC2_BE: [u8; 8] = [0x46, 0x4C, 0x4B, 0x5F, 0x57, 0x4C, 0x43, 0x32];
const WLC3_BE: [u8; 8] = [0x46, 0x4C, 0x4B, 0x5F, 0x57, 0x4C, 0x43, 0x33];

fn contains_magic(buf: &[u8]) -> Option<&'static str> {
    for (name, pat) in [
        ("WLC3-LE", &WLC3_LE[..]),
        ("WLC3-BE", &WLC3_BE[..]),
        ("WLC2-LE", &WLC2_LE[..]),
        ("WLC2-BE", &WLC2_BE[..]),
    ] {
        if buf.windows(8).any(|w| w == pat) {
            return Some(name);
        }
    }
    None
}

fn default_roots() -> Vec<PathBuf> {
    // Deliberately bounded: scanning $HOME pulled in the 14 GiB cargo `target`
    // tree and multi-GB registries, hanging the first local run (r292).
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"] {
        if let Ok(v) = env::var(var) {
            roots.push(PathBuf::from(&v));
        }
    }
    roots.push(PathBuf::from("/tmp"));
    roots.push(PathBuf::from("/var/tmp"));
    if let Ok(cwd) = env::current_dir() {
        roots.push(cwd);
    }
    roots
}

const MAX_FILES: usize = 60_000;
static FILES_SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn walk(dir: &Path, depth: usize, out: &mut Vec<(PathBuf, String)>) -> bool {
    if depth > 3 {
        return true;
    }
    if FILES_SEEN.load(std::sync::atomic::Ordering::Relaxed) > MAX_FILES {
        return false;
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return false,
    };
    for ent in entries.flatten() {
        let p = ent.path();
        let Ok(meta) = ent.metadata() else { continue };
        if meta.is_dir() {
            // Skip obvious huge/noise dirs.
            let name = ent.file_name().to_string_lossy().to_string();
            if matches!(
                name.as_str(),
                "target" | ".git" | "node_modules" | "Library" | "snap" | "proc" | "sys"
                    | "registry" | ".rustup" | "go" | ".cargo"
            ) {
                continue;
            }
            if !walk(&p, depth + 1, out) {
                return false;
            }
        } else if FILES_SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) <= MAX_FILES
            && meta.len() > 0
            && meta.len() < 64 << 20
        {
            // Only files plausibly holding a small cache record.
            let Ok(bytes) = fs::read(&p) else { continue };
            if let Some(kind) = contains_magic(&bytes) {
                out.push((p.clone(), kind.to_string()));
            }
        }
    }
    true
}

fn main() {
    let mut roots: Vec<PathBuf> = env::args()
        .skip(1)
        .map(PathBuf::from)
        .collect();
    if roots.is_empty() {
        roots = default_roots();
    }
    println!("[warmup-cache-probe] roots: {:?}", roots);
    let mut hits: Vec<(PathBuf, String)> = Vec::new();
    for r in &roots {
        if r.is_dir() {
            if !walk(r, 0, &mut hits) {
                println!("[warmup-cache-probe] scan aborted: file budget {} exceeded", MAX_FILES);
                break;
            }
        } else if r.is_file() {
            if let Ok(bytes) = fs::read(r) {
                if let Some(kind) = contains_magic(&bytes) {
                    hits.push((r.clone(), kind.to_string()));
                }
            }
        }
    }
    hits.sort();
    hits.dedup();
    if hits.is_empty() {
        println!("[warmup-cache-probe] NO cache file found in scanned roots");
    }
    for (p, kind) in &hits {
        let bytes = fs::read(p).unwrap_or_default();
        println!(
            "[warmup-cache-probe] HIT {} in {} ({} bytes)",
            kind,
            p.display(),
            bytes.len()
        );
        let n = bytes.len().min(64);
        println!("[warmup-cache-probe]   head: {:02x?}", &bytes[..n]);
        // Human-readable tail: verdict strings often sit near the end.
        if bytes.len() > 64 {
            let t = &bytes[bytes.len() - 64..];
            let s: String = t.iter().map(|&b| if b.is_ascii_graphic() || b == b' ' { b as char } else { '.' }).collect();
            println!("[warmup-cache-probe]   tail: {}", s);
        }
    }
    println!("[warmup-cache-probe] done");
}
