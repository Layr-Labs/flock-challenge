use rayon::prelude::*;
use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

pub use crate::hash::HashKind;

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
))]
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
))]
const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
#[path = "merkle/aarch64.rs"]
mod sha256x4;

#[cfg(all(target_arch = "x86_64", target_feature = "sha"))]
#[path = "merkle/x86_64.rs"]
mod sha256x4;

#[cfg(feature = "hash-count")]
pub mod hash_count {
    use super::HashKind;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static LEAF_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static LEAF_COMPRESSIONS: AtomicU64 = AtomicU64::new(0);
    pub static PAIR_CALLS: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub fn sha256_blocks(len: usize) -> u64 {
        ((len + 9).div_ceil(64)) as u64
    }

    #[inline]
    pub fn blake3_blocks(len: usize) -> u64 {
        let blocks = (len.div_ceil(64)).max(1) as u64;
        let chunks = (len.div_ceil(1024)).max(1) as u64;
        blocks + (chunks - 1)
    }

    #[inline]
    pub fn blocks(kind: HashKind, len: usize) -> u64 {
        match kind {
            HashKind::Sha256 => sha256_blocks(len),
            HashKind::Blake3 => blake3_blocks(len),
        }
    }

    pub fn reset() {
        LEAF_CALLS.store(0, Relaxed);
        LEAF_COMPRESSIONS.store(0, Relaxed);
        PAIR_CALLS.store(0, Relaxed);
    }

    pub fn snapshot() -> (u64, u64, u64) {
        (
            LEAF_CALLS.load(Relaxed),
            LEAF_COMPRESSIONS.load(Relaxed),
            PAIR_CALLS.load(Relaxed),
        )
    }
}

#[inline]
fn blake3_leaf_cv(data: &[u8]) -> Hash {
    use blake3::hazmat::HasherExt;
    blake3::Hasher::new().update(data).finalize_non_root()
}

#[inline]
fn blake3_parent_cv(left: &Hash, right: &Hash) -> Hash {
    blake3::hazmat::merge_subtrees_non_root(left, right, blake3::hazmat::Mode::Hash)
}

#[inline]
pub fn hash_leaf(data: &[u8], kind: HashKind) -> Hash {
    #[cfg(feature = "hash-count")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        hash_count::LEAF_CALLS.fetch_add(1, Relaxed);
        hash_count::LEAF_COMPRESSIONS.fetch_add(hash_count::blocks(kind, data.len()), Relaxed);
    }
    match kind {
        HashKind::Sha256 => Sha256::digest(data).into(),
        HashKind::Blake3 => blake3_leaf_cv(data),
    }
}

#[inline]
pub fn hash_pair(left: &Hash, right: &Hash, kind: HashKind) -> Hash {
    #[cfg(feature = "hash-count")]
    hash_count::PAIR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match kind {
        HashKind::Sha256 => {
            let mut h = Sha256::new();
            h.update(left);
            h.update(right);
            h.finalize().into()
        }
        HashKind::Blake3 => blake3_parent_cv(left, right),
    }
}

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
))]
#[inline]
fn sha256_hash4(inputs: [&[u8]; 4], outs: &mut [Hash]) {
    sha256x4::hash4_equal_len(inputs, outs);
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
)))]
#[inline]
fn sha256_hash4(inputs: [&[u8]; 4], outs: &mut [Hash]) {
    for (out, input) in outs.iter_mut().zip(inputs) {
        *out = Sha256::digest(input).into();
    }
}

const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const BLAKE3_CHUNK_START: u8 = 1;
const BLAKE3_CHUNK_END: u8 = 2;
const BLAKE3_PARENT: u8 = 4;

fn blake3_platform() -> blake3::platform::Platform {
    use std::sync::OnceLock;
    static PLATFORM: OnceLock<blake3::platform::Platform> = OnceLock::new();
    *PLATFORM.get_or_init(blake3::platform::Platform::detect)
}

const BLAKE3_BATCH: usize = 16;

#[inline]
fn blake3_hash_many<const N: usize>(
    data: &[u8],
    out: &mut [Hash],
    flags: u8,
    flags_start: u8,
    flags_end: u8,
) {
    debug_assert_eq!(data.len(), out.len() * N);
    let plat = blake3_platform();
    for (outs, msgs) in out
        .chunks_mut(BLAKE3_BATCH)
        .zip(data.chunks(BLAKE3_BATCH * N))
    {
        let n = outs.len();

        let first: &[u8; N] = msgs[..N].try_into().unwrap();
        let mut inputs: [&[u8; N]; BLAKE3_BATCH] = [first; BLAKE3_BATCH];
        for (i, slot) in inputs[..n].iter_mut().enumerate() {
            *slot = msgs[i * N..(i + 1) * N].try_into().unwrap();
        }

        let out_bytes: &mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(outs.as_mut_ptr() as *mut u8, n * 32) };
        plat.hash_many(
            &inputs[..n],
            &BLAKE3_IV,
            0,
            blake3::IncrementCounter::No,
            flags,
            flags_start,
            flags_end,
            out_bytes,
        );
    }
}

fn blake3_hash_many_leaves(data: &[u8], leaf_size: usize, out: &mut [Hash]) -> bool {
    macro_rules! dispatch {
        ($($n:literal),+ $(,)?) => {
            match leaf_size {
                $($n => {
                    blake3_hash_many::<$n>(
                        data, out, 0, BLAKE3_CHUNK_START, BLAKE3_CHUNK_END,
                    );
                    true
                })+
                _ => false,
            }
        };
    }

    dispatch!(64, 128, 256, 512, 1024)
}

#[inline]
fn blake3_leaf_size_is_batchable(leaf_size: usize) -> bool {
    matches!(leaf_size, 64 | 128 | 256 | 512 | 1024)
}

#[inline]
pub fn blake3_leaf_chunk_flags(block_idx: usize, num_blocks: usize) -> u8 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        crate::gf2_kernels::gf2_kern_blake3_chunk_flags(block_idx as u64, num_blocks as u64) as u8
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let start = if block_idx == 0 {
            BLAKE3_CHUNK_START
        } else {
            0
        };
        let end = if block_idx + 1 == num_blocks {
            BLAKE3_CHUNK_END
        } else {
            0
        };
        start | end
    }
}

#[inline]
fn blake3_hash_many_parents(data: &[u8], out: &mut [Hash]) {
    blake3_hash_many::<64>(data, out, BLAKE3_PARENT, 0, 0);
}

fn hash_leaves(data: &[u8], leaf_size: usize, out: &mut [Hash], kind: HashKind) {
    #[cfg(feature = "hash-count")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        hash_count::LEAF_CALLS.fetch_add(out.len() as u64, Relaxed);
        hash_count::LEAF_COMPRESSIONS.fetch_add(
            out.len() as u64 * hash_count::blocks(kind, leaf_size),
            Relaxed,
        );
    }
    match kind {
        HashKind::Blake3 if blake3_leaf_size_is_batchable(leaf_size) => {
            out.par_chunks_mut(BLAKE3_GROUP)
                .zip(data.par_chunks(BLAKE3_GROUP * leaf_size))
                .for_each(|(outs, leaves)| {
                    blake3_hash_many_leaves(leaves, leaf_size, outs);
                });
        }
        HashKind::Blake3 => out
            .par_iter_mut()
            .zip(data.par_chunks(leaf_size))
            .for_each(|(o, leaf)| *o = blake3_leaf_cv(leaf)),
        HashKind::Sha256 => {
            out.par_chunks_mut(4)
                .zip(data.par_chunks(4 * leaf_size))
                .for_each(|(outs, leaves)| {
                    if outs.len() == 4 {
                        sha256_hash4(
                            [
                                &leaves[..leaf_size],
                                &leaves[leaf_size..2 * leaf_size],
                                &leaves[2 * leaf_size..3 * leaf_size],
                                &leaves[3 * leaf_size..],
                            ],
                            outs,
                        );
                    } else {
                        for (out, leaf) in outs.iter_mut().zip(leaves.chunks(leaf_size)) {
                            *out = Sha256::digest(leaf).into();
                        }
                    }
                });
        }
    }
}

const BLAKE3_GROUP: usize = 1024;

fn hash_pairs_level(read: &[Hash], write: &mut [Hash], kind: HashKind) {
    #[cfg(feature = "hash-count")]
    hash_count::PAIR_CALLS.fetch_add(write.len() as u64, std::sync::atomic::Ordering::Relaxed);

    let read_bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(read.as_ptr() as *const u8, read.len() * 32) };
    const SERIAL_LEVEL_NODES: usize = 1024;
    let serial = write.len() <= SERIAL_LEVEL_NODES;

    match kind {
        HashKind::Blake3 => {
            if serial {
                blake3_hash_many_parents(read_bytes, write);
            } else {
                write
                    .par_chunks_mut(BLAKE3_GROUP)
                    .zip(read_bytes.par_chunks(BLAKE3_GROUP * 64))
                    .for_each(|(outs, children)| blake3_hash_many_parents(children, outs));
            }
        }
        HashKind::Sha256 => {
            let hash_quad = |outs: &mut [Hash], children: &[u8]| {
                if outs.len() == 4 {
                    sha256_hash4(
                        [
                            &children[..64],
                            &children[64..128],
                            &children[128..192],
                            &children[192..256],
                        ],
                        outs,
                    );
                } else {
                    for (i, out) in outs.iter_mut().enumerate() {
                        let l: &Hash = children[i * 64..i * 64 + 32].try_into().unwrap();
                        let r: &Hash = children[i * 64 + 32..i * 64 + 64].try_into().unwrap();
                        let mut h = Sha256::new();
                        h.update(l);
                        h.update(r);
                        *out = h.finalize().into();
                    }
                }
            };
            if serial {
                for (outs, children) in write.chunks_mut(4).zip(read_bytes.chunks(256)) {
                    hash_quad(outs, children);
                }
            } else {
                write
                    .par_chunks_mut(4)
                    .zip(read_bytes.par_chunks(256))
                    .for_each(|(outs, children)| hash_quad(outs, children));
            }
        }
    }
}

pub fn merkle_root(data: &[u8], num_leaves: usize, kind: HashKind) -> Hash {
    let tree = merkle_tree(data, num_leaves, kind);
    tree[tree.len() - 1]
}

pub fn merkle_tree(data: &[u8], num_leaves: usize, kind: HashKind) -> Vec<Hash> {
    assert!(
        num_leaves.is_power_of_two() && num_leaves > 0,
        "num_leaves must be power of 2"
    );
    assert_eq!(
        data.len() % num_leaves,
        0,
        "data length must be a multiple of num_leaves"
    );

    let leaf_size = data.len() / num_leaves;
    if kind == HashKind::Blake3
        && let Some(gpu_tree) = crate::gpu_commit::try_gpu_merkle_blake3(data, num_leaves)
    {
        return gpu_tree;
    }
    let total_nodes = 2 * num_leaves - 1;

    let mut tree: Vec<Hash> = crate::scratch::take_hash(total_nodes);

    hash_leaves(data, leaf_size, &mut tree[..num_leaves], kind);

    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;

        let (read, rest) = tree[read_start..].split_at_mut(read_len);
        let write = &mut rest[..next_len];

        hash_pairs_level(read, write, kind);

        read_start += read_len;
        read_len = next_len;
    }

    tree
}

#[inline]
pub fn hash_leaves_serial(data: &[u8], leaf_size: usize, out: &mut [Hash], kind: HashKind) {
    debug_assert_eq!(data.len(), out.len() * leaf_size);
    #[cfg(feature = "hash-count")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        hash_count::LEAF_CALLS.fetch_add(out.len() as u64, Relaxed);
        hash_count::LEAF_COMPRESSIONS.fetch_add(
            out.len() as u64 * hash_count::blocks(kind, leaf_size),
            Relaxed,
        );
    }
    match kind {
        HashKind::Blake3 if blake3_leaf_size_is_batchable(leaf_size) => {
            blake3_hash_many_leaves(data, leaf_size, out);
        }
        HashKind::Blake3 => {
            for (o, leaf) in out.iter_mut().zip(data.chunks(leaf_size)) {
                *o = blake3_leaf_cv(leaf);
            }
        }
        HashKind::Sha256 => {
            for (outs, leaves) in out.chunks_mut(4).zip(data.chunks(4 * leaf_size)) {
                if outs.len() == 4 {
                    sha256_hash4(
                        [
                            &leaves[..leaf_size],
                            &leaves[leaf_size..2 * leaf_size],
                            &leaves[2 * leaf_size..3 * leaf_size],
                            &leaves[3 * leaf_size..],
                        ],
                        outs,
                    );
                } else {
                    for (out, leaf) in outs.iter_mut().zip(leaves.chunks(leaf_size)) {
                        *out = Sha256::digest(leaf).into();
                    }
                }
            }
        }
    }
}

#[inline]
pub fn hash_pairs_level_serial(read: &[Hash], write: &mut [Hash], kind: HashKind) {
    debug_assert_eq!(read.len(), write.len() * 2);
    #[cfg(feature = "hash-count")]
    hash_count::PAIR_CALLS.fetch_add(write.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let read_bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(read.as_ptr() as *const u8, read.len() * 32) };
    match kind {
        HashKind::Blake3 => {
            blake3_hash_many_parents(read_bytes, write);
        }
        HashKind::Sha256 => {
            for (outs, children) in write.chunks_mut(4).zip(read_bytes.chunks(256)) {
                if outs.len() == 4 {
                    sha256_hash4(
                        [
                            &children[..64],
                            &children[64..128],
                            &children[128..192],
                            &children[192..256],
                        ],
                        outs,
                    );
                } else {
                    for (i, out) in outs.iter_mut().enumerate() {
                        let l: &Hash = children[i * 64..i * 64 + 32].try_into().unwrap();
                        let r: &Hash = children[i * 64 + 32..i * 64 + 64].try_into().unwrap();
                        let mut h = Sha256::new();
                        h.update(l);
                        h.update(r);
                        *out = h.finalize().into();
                    }
                }
            }
        }
    }
}

pub fn build_upper_levels(
    tree: &mut [Hash],
    num_leaves: usize,
    first_unfinished_parent_level: usize,
    kind: HashKind,
) {
    debug_assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    debug_assert_eq!(tree.len(), 2 * num_leaves - 1);
    let mut read_len = num_leaves >> first_unfinished_parent_level;
    let mut read_start = 2 * num_leaves - 2 * read_len;
    while read_len > 1 {
        let next_len = read_len >> 1;
        let (read, rest) = tree[read_start..].split_at_mut(read_len);
        let write = &mut rest[..next_len];
        hash_pairs_level(read, write, kind);
        read_start += read_len;
        read_len = next_len;
    }
}

pub fn merkle_tree_sequential(data: &[u8], num_leaves: usize, kind: HashKind) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(data.len() % num_leaves, 0);

    let leaf_size = data.len() / num_leaves;
    let total_nodes = 2 * num_leaves - 1;
    let mut tree: Vec<Hash> = crate::alloc_uninit_vec(total_nodes);

    for (i, leaf) in data.chunks(leaf_size).enumerate() {
        tree[i] = hash_leaf(leaf, kind);
    }
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        for i in 0..next_len {
            let left = tree[read_start + 2 * i];
            let right = tree[read_start + 2 * i + 1];
            tree[read_start + read_len + i] = hash_pair(&left, &right, kind);
        }
        read_start += read_len;
        read_len = next_len;
    }
    tree
}

pub fn merkle_proof(tree: &[Hash], num_leaves: usize, index: usize) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert!(index < num_leaves);
    assert_eq!(tree.len(), 2 * num_leaves - 1);

    let log_n = num_leaves.trailing_zeros() as usize;
    let mut proof = Vec::with_capacity(log_n);

    let mut level_start = 0usize;
    let mut level_len = num_leaves;
    let mut idx = index;
    while level_len > 1 {
        let sibling_idx = idx ^ 1;
        proof.push(tree[level_start + sibling_idx]);
        level_start += level_len;
        level_len >>= 1;
        idx >>= 1;
    }
    proof
}

pub fn verify_merkle_proof(
    root: &Hash,
    leaf_hash: &Hash,
    index: usize,
    proof: &[Hash],
    kind: HashKind,
) -> bool {
    let mut acc = *leaf_hash;
    let mut idx = index;
    for sibling in proof {
        let (left, right) = if idx & 1 == 0 {
            (acc, *sibling)
        } else {
            (*sibling, acc)
        };
        acc = hash_pair(&left, &right, kind);
        idx >>= 1;
    }
    &acc == root
}

pub fn merkle_multi_proof(tree: &[Hash], num_leaves: usize, positions: &[usize]) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(tree.len(), 2 * num_leaves - 1);

    if positions.is_empty() || num_leaves == 1 {
        return Vec::new();
    }

    let mut active: Vec<usize> = positions.to_vec();
    active.sort_unstable();
    active.dedup();
    debug_assert!(active.iter().all(|&p| p < num_leaves));

    let mut proof = Vec::new();
    let mut level_start = 0usize;
    let mut level_len = num_leaves;

    while level_len > 1 {
        let mut next = Vec::with_capacity(active.len());
        let mut i = 0;
        while i < active.len() {
            let p = active[i];
            let sib_active = i + 1 < active.len() && active[i + 1] == (p ^ 1);
            if sib_active {
                i += 2;
            } else {
                proof.push(tree[level_start + (p ^ 1)]);
                i += 1;
            }
            next.push(p >> 1);
        }

        active = next;
        level_start += level_len;
        level_len >>= 1;
    }

    proof
}

pub fn verify_merkle_multi_proof(
    root: &Hash,
    num_leaves: usize,
    sorted_unique_positions: &[usize],
    leaf_hashes: &[Hash],
    proof: &[Hash],
    kind: HashKind,
) -> bool {
    if !num_leaves.is_power_of_two() || num_leaves == 0 {
        return false;
    }
    if sorted_unique_positions.len() != leaf_hashes.len() {
        return false;
    }
    if sorted_unique_positions.is_empty() {
        return proof.is_empty();
    }

    for (i, &p) in sorted_unique_positions.iter().enumerate() {
        if p >= num_leaves {
            return false;
        }
        if i > 0 && sorted_unique_positions[i - 1] >= p {
            return false;
        }
    }

    if num_leaves == 1 {
        return proof.is_empty() && leaf_hashes[0] == *root;
    }

    let mut active: Vec<(usize, Hash)> = sorted_unique_positions
        .iter()
        .copied()
        .zip(leaf_hashes.iter().copied())
        .collect();
    let mut proof_iter = proof.iter().copied();
    let mut level_len = num_leaves;

    while level_len > 1 {
        let mut next = Vec::with_capacity(active.len());
        let mut i = 0;
        while i < active.len() {
            let (p, h) = active[i];
            let sib_active = i + 1 < active.len() && active[i + 1].0 == (p ^ 1);
            let (left, right) = if sib_active {
                let (_, h_sib) = active[i + 1];

                debug_assert_eq!(p & 1, 0);
                i += 2;
                (h, h_sib)
            } else {
                let sib = match proof_iter.next() {
                    Some(s) => s,
                    None => return false,
                };
                i += 1;
                if p & 1 == 0 { (h, sib) } else { (sib, h) }
            };
            next.push((p >> 1, hash_pair(&left, &right, kind)));
        }
        active = next;
        level_len >>= 1;
    }

    if proof_iter.next().is_some() {
        return false;
    }
    active.len() == 1 && active[0].1 == *root
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

    #[test]
    fn two_leaves_matches_hand_computation() {
        let data: Vec<u8> = (0..16).collect();
        for kind in KINDS {
            let tree = merkle_tree(&data, 2, kind);
            assert_eq!(tree.len(), 3);

            let h0 = hash_leaf(&data[0..8], kind);
            let h1 = hash_leaf(&data[8..16], kind);
            let root = hash_pair(&h0, &h1, kind);

            assert_eq!(tree[0], h0, "{kind}");
            assert_eq!(tree[1], h1, "{kind}");
            assert_eq!(tree[2], root, "{kind}");
        }
    }

    #[test]
    fn primitives_match_reference_implementations() {
        use blake3::hazmat::HasherExt;
        let data: Vec<u8> = (0..=255u8).cycle().take(3000).collect();

        assert_eq!(
            hash_leaf(&data, HashKind::Sha256),
            <[u8; 32]>::from(Sha256::digest(&data))
        );
        let (l, r) = ([7u8; 32], [9u8; 32]);
        let cat: Vec<u8> = l.iter().chain(r.iter()).copied().collect();
        assert_eq!(
            hash_pair(&l, &r, HashKind::Sha256),
            hash_leaf(&cat, HashKind::Sha256),
            "sha256 pair hash is the digest of the concatenation"
        );

        assert_eq!(
            hash_leaf(&data, HashKind::Blake3),
            blake3::Hasher::new().update(&data).finalize_non_root()
        );
        assert_eq!(
            hash_pair(&l, &r, HashKind::Blake3),
            blake3::hazmat::merge_subtrees_non_root(&l, &r, blake3::hazmat::Mode::Hash)
        );

        assert_ne!(
            hash_leaf(&data, HashKind::Blake3),
            *blake3::hash(&data).as_bytes(),
            "leaf CVs must be non-root"
        );
    }

    #[test]
    fn blake3_separates_leaf_and_parent_domains() {
        let (l, r) = ([7u8; 32], [9u8; 32]);
        let cat: Vec<u8> = l.iter().chain(r.iter()).copied().collect();
        assert_ne!(
            hash_pair(&l, &r, HashKind::Blake3),
            hash_leaf(&cat, HashKind::Blake3),
            "PARENT flag must separate the two domains"
        );

        assert_eq!(
            hash_pair(&l, &r, HashKind::Sha256),
            hash_leaf(&cat, HashKind::Sha256),
        );
    }

    #[test]
    fn blake3_batched_matches_scalar_spec() {
        let counts = [1usize, 5, 63, 64, 65, 200];

        for n in counts {
            let children: Vec<u8> = (0..=255u8).cycle().take(n * 64).collect();
            let mut batched = vec![[0u8; 32]; n];
            blake3_hash_many_parents(&children, &mut batched);
            for i in 0..n {
                let l: &Hash = children[i * 64..i * 64 + 32].try_into().unwrap();
                let r: &Hash = children[i * 64 + 32..i * 64 + 64].try_into().unwrap();
                assert_eq!(batched[i], blake3_parent_cv(l, r), "parent {i} of {n}");
            }
        }

        for leaf_size in [64usize, 128, 256, 512, 1024] {
            for n in counts {
                let data: Vec<u8> = (0..=255u8).cycle().take(n * leaf_size).collect();
                let mut batched = vec![[0u8; 32]; n];
                assert!(
                    blake3_hash_many_leaves(&data, leaf_size, &mut batched),
                    "size {leaf_size} should take the batched path"
                );
                for i in 0..n {
                    assert_eq!(
                        batched[i],
                        blake3_leaf_cv(&data[i * leaf_size..(i + 1) * leaf_size]),
                        "leaf {i} of {n} at size {leaf_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn blake3_batch_dispatch_agrees() {
        for leaf_size in [
            1usize, 16, 32, 48, 63, 64, 65, 100, 128, 192, 256, 512, 1000, 1024, 1088, 2048,
        ] {
            let data = vec![0u8; leaf_size];
            let mut out = [[0u8; 32]; 1];
            let dispatched = blake3_hash_many_leaves(&data, leaf_size, &mut out);
            assert_eq!(
                dispatched,
                blake3_leaf_size_is_batchable(leaf_size),
                "predicate and dispatch disagree at leaf_size={leaf_size}"
            );
        }
    }

    #[test]
    fn the_two_kinds_produce_different_roots() {
        let data = random_data(64, 32, 11);
        assert_ne!(
            merkle_root(&data, 64, HashKind::Sha256),
            merkle_root(&data, 64, HashKind::Blake3)
        );
    }

    #[test]
    fn one_leaf_root_is_the_leaf_hash() {
        let data: Vec<u8> = (0..32).collect();
        for kind in KINDS {
            assert_eq!(
                merkle_root(&data, 1, kind),
                hash_leaf(&data, kind),
                "{kind}"
            );
        }
    }

    #[test]
    fn parallel_matches_sequential() {
        let n_leaves = 1024;
        let leaf_size = 64;
        let mut data = vec![0u8; n_leaves * leaf_size];

        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E3779B9)) & 0xff) as u8;
        }
        for kind in KINDS {
            let par = merkle_tree(&data, n_leaves, kind);
            let seq = merkle_tree_sequential(&data, n_leaves, kind);
            assert_eq!(par, seq, "{kind}");
        }
    }

    #[test]
    fn parallel_matches_sequential_tail_shapes() {
        for (n_leaves, leaf_size) in [
            (64, 1024),
            (64, 1025),
            (64, 2048),
            (64, 100),
            (64, 60),
            (64, 56),
            (2, 48),
            (16, 1),
        ] {
            let mut data = vec![0u8; n_leaves * leaf_size];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(0x6C8E944D)) & 0xff) as u8;
            }
            for kind in KINDS {
                let par = merkle_tree(&data, n_leaves, kind);
                let seq = merkle_tree_sequential(&data, n_leaves, kind);
                assert_eq!(par, seq, "{kind} n_leaves={n_leaves} leaf_size={leaf_size}");
            }
        }
    }

    #[test]
    fn root_changes_when_any_leaf_changes() {
        let n_leaves = 64;
        let leaf_size = 32;
        let mut data = vec![0u8; n_leaves * leaf_size];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31);
        }
        for kind in KINDS {
            let r0 = merkle_root(&data, n_leaves, kind);

            data[n_leaves * leaf_size - 1] ^= 0x01;
            let r1 = merkle_root(&data, n_leaves, kind);
            assert_ne!(r0, r1, "{kind}: single-bit change should change the root");
            data[n_leaves * leaf_size - 1] ^= 0x01;
        }
    }

    #[test]
    fn power_of_two_assertion() {
        let data = vec![0u8; 64];

        for kind in KINDS {
            let _ = merkle_root(&data, 1, kind);
            let _ = merkle_root(&data, 2, kind);
            let _ = merkle_root(&data, 4, kind);
            let _ = merkle_root(&data, 8, kind);
        }
    }

    #[test]
    #[should_panic(expected = "num_leaves must be power of 2")]
    fn rejects_non_power_of_two() {
        let data = vec![0u8; 30];
        let _ = merkle_root(&data, 3, HashKind::Sha256);
    }

    #[test]
    fn merkle_proof_roundtrips_at_every_leaf() {
        let n_leaves = 16;
        let leaf_size = 8;
        let mut data = vec![0u8; n_leaves * leaf_size];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E3779B9)) & 0xff) as u8;
        }
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            for i in 0..n_leaves {
                let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
                let proof = merkle_proof(&tree, n_leaves, i);
                assert_eq!(proof.len(), 4);
                assert!(
                    verify_merkle_proof(&root, &leaf_hash, i, &proof, kind),
                    "{kind}: verify failed at i={i}"
                );
            }
        }
    }

    #[test]
    fn merkle_proof_rejects_the_other_hash() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 77);
        for kind in KINDS {
            let other = match kind {
                HashKind::Sha256 => HashKind::Blake3,
                HashKind::Blake3 => HashKind::Sha256,
            };
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();
            let leaf_hash = hash_leaf(&data[0..leaf_size], kind);
            let proof = merkle_proof(&tree, n_leaves, 0);

            assert!(verify_merkle_proof(&root, &leaf_hash, 0, &proof, kind));
            assert!(
                !verify_merkle_proof(&root, &leaf_hash, 0, &proof, other),
                "{kind} proof must not verify as {other}"
            );
        }
    }

    #[test]
    fn merkle_proof_rejects_wrong_index() {
        let n_leaves = 8;
        let leaf_size = 16;
        let data: Vec<u8> = (0..(n_leaves * leaf_size) as u8).collect();
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let leaf_hash = hash_leaf(&data[0..leaf_size], kind);
            let proof = merkle_proof(&tree, n_leaves, 0);

            assert!(
                !verify_merkle_proof(&root, &leaf_hash, 1, &proof, kind),
                "{kind}"
            );
        }
    }

    #[test]
    fn merkle_proof_rejects_tampered_path() {
        let n_leaves = 8;
        let leaf_size = 16;
        let data: Vec<u8> = (0..(n_leaves * leaf_size) as u8).collect();
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let leaf_hash = hash_leaf(&data[0..leaf_size], kind);
            let mut proof = merkle_proof(&tree, n_leaves, 0);

            proof[0][0] ^= 1;
            assert!(
                !verify_merkle_proof(&root, &leaf_hash, 0, &proof, kind),
                "{kind}"
            );
        }
    }

    fn random_data(n_leaves: usize, leaf_size: usize, seed: u64) -> Vec<u8> {
        let mut data = vec![0u8; n_leaves * leaf_size];
        let mut z = seed;
        for b in data.iter_mut() {
            z = z.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *b = ((z >> 33) & 0xff) as u8;
        }
        data
    }

    #[test]
    fn multi_proof_single_position_matches_single_proof() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 42);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            for i in 0..n_leaves {
                let multi = merkle_multi_proof(&tree, n_leaves, &[i]);
                let single = merkle_proof(&tree, n_leaves, i);
                assert_eq!(
                    multi, single,
                    "{kind}: multi-proof of [{i}] must equal single proof"
                );

                let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
                assert!(verify_merkle_multi_proof(
                    &root,
                    n_leaves,
                    &[i],
                    &[leaf_hash],
                    &multi,
                    kind
                ));
            }
        }
    }

    #[test]
    fn multi_proof_sibling_pair_emits_no_hashes_at_leaf_level() {
        let n_leaves = 8;
        let leaf_size = 4;
        let data = random_data(n_leaves, leaf_size, 7);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let multi = merkle_multi_proof(&tree, n_leaves, &[0, 1]);
            assert_eq!(
                multi.len(),
                2,
                "sibling pair at leaves saves the leaf-level hash"
            );

            let leaves: Vec<Hash> = [0usize, 1]
                .iter()
                .map(|&i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind))
                .collect();
            assert!(verify_merkle_multi_proof(
                &root,
                n_leaves,
                &[0, 1],
                &leaves,
                &multi,
                kind
            ));
        }
    }

    #[test]
    fn multi_proof_full_query_set_is_root_only() {
        let n_leaves = 16;
        let leaf_size = 8;
        let data = random_data(n_leaves, leaf_size, 99);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions: Vec<usize> = (0..n_leaves).collect();
            let multi = merkle_multi_proof(&tree, n_leaves, &positions);
            assert!(
                multi.is_empty(),
                "full-set multi-proof should have zero hashes"
            );

            let leaves: Vec<Hash> = (0..n_leaves)
                .map(|i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind))
                .collect();
            assert!(verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
        }
    }

    #[test]
    fn multi_proof_random_subsets_roundtrip() {
        let n_leaves = 64;
        let leaf_size = 16;
        let data = random_data(n_leaves, leaf_size, 2024);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let all_leaves: Vec<Hash> = (0..n_leaves)
                .map(|i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind))
                .collect();

            let subsets: &[&[usize]] = &[
                &[0],
                &[63],
                &[0, 63],
                &[3, 17, 41],
                &[10, 11, 12, 13],
                &[0, 1, 2, 3, 60, 61, 62, 63],
                &[5, 5, 5, 17, 17],
                &[0, 8, 16, 24, 32, 40, 48, 56],
            ];
            for positions in subsets {
                let multi = merkle_multi_proof(&tree, n_leaves, positions);

                let mut sorted: Vec<usize> = positions.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                let leaves: Vec<Hash> = sorted.iter().map(|&p| all_leaves[p]).collect();

                assert!(
                    verify_merkle_multi_proof(&root, n_leaves, &sorted, &leaves, &multi, kind),
                    "{kind}: roundtrip failed for positions={positions:?}"
                );

                let log_n = n_leaves.trailing_zeros() as usize;
                assert!(
                    multi.len() <= sorted.len() * log_n,
                    "multi-proof can't exceed sum of independent paths"
                );
            }
        }
    }

    #[test]
    fn multi_proof_rejects_wrong_leaf() {
        let (n_leaves, leaf_size) = (32, 8);
        let data = random_data(n_leaves, leaf_size, 1);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions = vec![3usize, 7, 19, 28];
            let multi = merkle_multi_proof(&tree, n_leaves, &positions);
            let mut leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            assert!(verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
            leaves[1][0] ^= 1;
            assert!(!verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
        }
    }

    #[test]
    fn multi_proof_rejects_tampered_proof_hash() {
        let (n_leaves, leaf_size) = (32, 8);
        let data = random_data(n_leaves, leaf_size, 2);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions = vec![1usize, 14, 27];
            let mut multi = merkle_multi_proof(&tree, n_leaves, &positions);
            let leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            assert!(verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
            multi[0][0] ^= 1;
            assert!(!verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
        }
    }

    #[test]
    fn multi_proof_rejects_extra_or_missing_hashes() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 3);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions = vec![2usize, 11];
            let multi = merkle_multi_proof(&tree, n_leaves, &positions);
            let leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            let mut extra = multi.clone();
            extra.push([0xaa; 32]);
            assert!(!verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &extra, kind
            ));

            let mut short = multi.clone();
            short.pop();
            assert!(!verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &short, kind
            ));
        }
    }

    #[test]
    fn multi_proof_rejects_unsorted_positions() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 5);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions = vec![2usize, 11];
            let multi = merkle_multi_proof(&tree, n_leaves, &positions);
            let leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            let unsorted = vec![11usize, 2];
            let unsorted_leaves = vec![leaves[1], leaves[0]];
            assert!(!verify_merkle_multi_proof(
                &root,
                n_leaves,
                &unsorted,
                &unsorted_leaves,
                &multi,
                kind
            ));
        }
    }

    #[test]
    fn multi_proof_beats_independent_paths_at_scale() {
        let n_leaves = 1024;
        let leaf_size = 8;
        let data = random_data(n_leaves, leaf_size, 4096);
        let log_n = n_leaves.trailing_zeros() as usize;
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions_raw: Vec<usize> = (0..100)
                .map(|i| {
                    let mut z = (i as u64).wrapping_mul(0xDEAD_BEEF_F0F0_F0F0);
                    z ^= z >> 27;
                    (z as usize) & (n_leaves - 1)
                })
                .collect();
            let multi = merkle_multi_proof(&tree, n_leaves, &positions_raw);

            let mut positions = positions_raw.clone();
            positions.sort_unstable();
            positions.dedup();
            let leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            assert!(verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
            assert!(
                multi.len() < positions.len() * log_n,
                "multi-proof should beat independent paths: got {} vs {} × {}",
                multi.len(),
                positions.len(),
                log_n
            );
        }
    }
}
