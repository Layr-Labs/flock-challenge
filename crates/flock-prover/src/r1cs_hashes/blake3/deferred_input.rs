//! Reconstruct deferred A/B inputs from a completed row-major witness.
use super::K;
use crate::seed_pipe::CompressionQuadSoa;

/// Any value restores the original input generator. Resolve once per prove
/// handle, never inside the per-quad loop.
pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_AB_INPUT_FROM_Z").is_none())
}

/// Decode four complete blocks; input fields are independent of G trace bits.
#[inline(always)]
pub(super) fn from_quad_words(words: &[u32]) -> CompressionQuadSoa {
    const WORDS_PER_BLOCK: usize = K / 32;
    assert_eq!(words.len(), 4 * WORDS_PER_BLOCK);
    let mut quad = CompressionQuadSoa {
        cv: [[0; 4]; 8],
        message: [[0; 4]; 16],
        counter_lo: [0; 4],
        counter_hi: [0; 4],
        block_len: [0; 4],
        flags: [0; 4],
    };
    for lane in 0..4 {
        let block = &words[lane * WORDS_PER_BLOCK..(lane + 1) * WORDS_PER_BLOCK];
        for j in 0..8 {
            quad.cv[j][lane] = block[j];
        }
        // The message and the four control words begin at bit513, one bit
        // into word16. Read both adjacent words, including flags bit31 at
        // bit1152; the first G-trace bit at1153 is shifted out.
        let input = |j: usize| (block[16 + j] >> 1) | (block[17 + j] << 31);
        for j in 0..16 {
            quad.message[j][lane] = input(j);
        }
        quad.counter_lo[lane] = input(16);
        quad.counter_hi[lane] = input(17);
        quad.block_len[lane] = input(18);
        quad.flags[lane] = input(19);
    }
    quad
}

/// Compare every decoded field against the independent input vector.
/// Kept separate from admission so malformed inputs are directly testable.
pub(super) fn matches_inputs(words: &[u32], blocks: &[super::Compression]) -> bool {
    use rayon::prelude::*;
    const QUAD_WORDS: usize = 4 * (K / 32);
    if !blocks.len().is_multiple_of(4)
        || blocks.len().checked_mul(K / 32) != Some(words.len())
    {
        return false;
    }
    blocks.par_chunks_exact(4).zip(words.par_chunks_exact(QUAD_WORDS)).all(|(blocks, words)| {
        let got = from_quad_words(words);
        (0..4).all(|lane| {
            let (cv, message, counter, block_len, flags) = &blocks[lane];
            (0..8).all(|j| got.cv[j][lane] == cv[j])
                && (0..16).all(|j| got.message[j][lane] == message[j])
                && (u64::from(got.counter_lo[lane]) | (u64::from(got.counter_hi[lane]) << 32)) == *counter
                && got.block_len[lane] == *block_len
                && got.flags[lane] == *flags
        })
    })
}

// 0: no independent warm-up check yet; 1: all decoded inputs matched;
// 2: a mismatch occurred. A mismatch never enables the optimization.
static MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Run only after the completed warm-up witness drain, on ordinary input
/// blocks rather than the seed pipe's lazy backing. The witness's lifetime
/// is still owned by the caller and this comparison borrows it read-only.
pub(super) fn admit(words: &[u32], blocks: &[super::Compression]) {
    use std::sync::atomic::Ordering;
    if !enabled() || !cfg!(target_endian = "little") || MODE.load(Ordering::Acquire) != 0 {
        return;
    }
    let verdict = if matches_inputs(words, blocks) { 1 } else { 2 };
    let _ = MODE.compare_exchange(0, verdict, Ordering::AcqRel, Ordering::Acquire);
}

pub(super) fn admitted() -> bool {
    enabled() && MODE.load(std::sync::atomic::Ordering::Acquire) == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_input_validation_is_complete_and_rejects_malformed() {
        let mut blocks = Vec::new();
        let mut words = Vec::new();
        for i in 0..8 {
            let block = crate::seed_pipe::gen_block_scalar(0x7d3a_0042, i);
            let bits = super::super::build_block_witness(
                &block.0, &block.1, block.2, block.3, block.4,
            );
            let mut packed = vec![0u32; K / 32];
            for (bit, &set) in bits.iter().enumerate() {
                packed[bit / 32] |= u32::from(set) << (bit % 32);
            }
            words.extend_from_slice(&packed);
            blocks.push(block);
        }
        assert!(matches_inputs(&words, &blocks));
        assert!(!matches_inputs(&words[..words.len() - 1], &blocks));
        assert!(!matches_inputs(&words, &blocks[..7]));
        // Every one of the 896 input bits per block must be checked.
        for block in [0usize, 7] {
            for bit in (0..256).chain(513..1153) {
                let at = block * (K / 32) + bit / 32;
                words[at] ^= 1 << (bit % 32);
                assert!(!matches_inputs(&words, &blocks), "block={block} bit={bit}");
                words[at] ^= 1 << (bit % 32);
            }
        }
        // Non-input trace bits must not affect the decoder's admission.
        words[36] ^= 1 << 1;
        assert!(matches_inputs(&words, &blocks));
    }
}
