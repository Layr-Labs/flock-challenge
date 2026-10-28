use crate::field::F128;
use crate::merkle::{self, Hash, HashKind};
use crate::ntt::AdditiveNttF128;
use crate::pcs::pack::LOG_PACKING;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PcsParams {
    pub m: usize,
    pub log_inv_rate: usize,
    pub log_batch_size: usize,
    #[serde(default)]
    pub profile: crate::pcs::ligerito::LigeritoProfile,
    #[serde(default)]
    pub merkle_hash: HashKind,
}

impl PcsParams {
    pub fn log_msg_len(&self) -> usize {
        self.m - LOG_PACKING
    }
    pub fn log_dim(&self) -> usize {
        self.log_msg_len() - self.log_batch_size
    }
    pub fn k_code(&self) -> usize {
        self.log_dim() + self.log_inv_rate
    }
    pub fn n_positions(&self) -> usize {
        1usize << self.k_code()
    }
    pub fn num_ntts(&self) -> usize {
        1usize << self.log_batch_size
    }
    pub fn codeword_len_f128(&self) -> usize {
        self.n_positions() * self.num_ntts()
    }
    pub fn log_leaf_f128_count(&self) -> usize {
        self.log_batch_size
    }
    pub fn n_leaves(&self) -> usize {
        self.codeword_len_f128() >> self.log_leaf_f128_count()
    }
    pub fn leaf_size_bytes(&self) -> usize {
        16usize << self.log_leaf_f128_count()
    }

    pub fn ligerito_prover_config(&self) -> Result<crate::pcs::ligerito::ProverConfig, String> {
        let mut cfg = crate::pcs::ligerito::prover_config_for(
            self.log_msg_len(),
            self.log_batch_size,
            self.profile,
        )?;
        cfg.merkle_hash = self.merkle_hash;
        Ok(cfg)
    }

    pub fn ligerito_verifier_config(&self) -> Result<crate::pcs::ligerito::VerifierConfig, String> {
        let mut cfg = crate::pcs::ligerito::verifier_config_for(
            self.log_msg_len(),
            self.log_batch_size,
            self.profile,
        )?;
        cfg.merkle_hash = self.merkle_hash;
        Ok(cfg)
    }

    fn validate(&self) {
        assert!(
            self.m >= LOG_PACKING + self.log_batch_size,
            "m={} too small (need m ≥ LOG_PACKING + log_batch_size = {})",
            self.m,
            LOG_PACKING + self.log_batch_size,
        );
        assert!(
            self.log_inv_rate >= 1,
            "log_inv_rate must be ≥ 1 for a non-trivial RS code",
        );
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitment {
    pub root: Hash,
    pub params: PcsParams,
}

pub struct ProverData {
    pub codeword: Vec<F128>,
    pub merkle_tree: Vec<Hash>,
}

impl Drop for ProverData {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.codeword));
        crate::scratch::give_hash(std::mem::take(&mut self.merkle_tree));
    }
}

pub fn commit(z_packed: &[F128], params: &PcsParams) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), 1usize << params.log_msg_len());

    let num_ntts = params.num_ntts();
    let n_positions = params.n_positions();
    let codeword_len = n_positions * num_ntts;

    let codeword = crate::scratch::take_f128(codeword_len);
    commit_into(z_packed, params, codeword)
}

pub fn commit_into(
    z_packed: &[F128],
    params: &PcsParams,
    codeword: Vec<F128>,
) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), 1usize << params.log_msg_len());
    let codeword_len = params.n_positions() * params.num_ntts();
    assert_eq!(
        codeword.len(),
        codeword_len,
        "commit_into: prebuilt codeword buffer has wrong length"
    );

    finalize_commit(codeword, z_packed, params)
}

pub(crate) fn replicate_message_fill(codeword: &mut [F128], msg: &[F128]) {
    use rayon::prelude::*;
    let msg_len = msg.len();
    debug_assert!(codeword.len().is_multiple_of(msg_len));
    const COPY_CHUNK: usize = 1 << 16;
    if msg_len >= COPY_CHUNK {
        let log_msg_len = msg_len.trailing_zeros() as u64;
        codeword
            .par_chunks_mut(COPY_CHUNK)
            .enumerate()
            .for_each(|(i, dst)| {
                let src_off = unsafe {
                    crate::gf2_kernels::gf2_kern_ntt_replicated_msg_idx(
                        (i * COPY_CHUNK) as u64,
                        log_msg_len,
                    )
                } as usize;
                dst.copy_from_slice(&msg[src_off..src_off + dst.len()]);
            });
    } else {
        for rep in codeword.chunks_mut(msg_len) {
            rep.copy_from_slice(msg);
        }
    }
}

fn finalize_commit(
    mut codeword: Vec<F128>,
    z_packed: &[F128],
    params: &PcsParams,
) -> (Commitment, ProverData) {
    let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
    let t_ntt = std::time::Instant::now();
    let ntt = AdditiveNttF128::standard(params.k_code());
    if crate::gpu_commit::gpu_commit_enabled() {
        replicate_message_fill(&mut codeword, z_packed);
        if params.merkle_hash == HashKind::Blake3 {
            if let Some(merkle_tree) = crate::gpu_commit::try_gpu_commit_ntt_and_blake3_merkle(
                &ntt,
                &mut codeword,
                params.num_ntts(),
                params.log_inv_rate,
                params.n_leaves(),
            ) {
                let root = *merkle_tree.last().expect("merkle tree non-empty");
                if timing {
                    eprintln!(
                        "[commit-timing] fused gpu ntt+merkle: {:.2} ms",
                        t_ntt.elapsed().as_secs_f64() * 1e3
                    );
                }
                return (
                    Commitment {
                        root,
                        params: params.clone(),
                    },
                    ProverData {
                        codeword,
                        merkle_tree,
                    },
                );
            }
        }
        if !crate::gpu_commit::try_gpu_interleaved_ntt_from_layer(
            &ntt,
            &mut codeword,
            params.num_ntts(),
            params.log_inv_rate,
        ) {
            ntt.forward_transform_interleaved_from_layer(
                &mut codeword,
                params.num_ntts(),
                params.log_inv_rate,
            );
        }
        if timing {
            eprintln!(
                "[commit-timing] ntt: {:.2} ms",
                t_ntt.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_merkle = std::time::Instant::now();
        let codeword_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                codeword.as_ptr() as *const u8,
                codeword.len() * core::mem::size_of::<F128>(),
            )
        };
        let merkle_tree =
            merkle::merkle_tree(codeword_bytes, params.n_leaves(), params.merkle_hash);
        let root = *merkle_tree.last().expect("merkle tree non-empty");
        if timing {
            eprintln!(
                "[commit-timing] merkle: {:.2} ms",
                t_merkle.elapsed().as_secs_f64() * 1e3
            );
        }
        (
            Commitment {
                root,
                params: params.clone(),
            },
            ProverData {
                codeword,
                merkle_tree,
            },
        )
    } else {
        let n_leaves = params.n_leaves();
        let num_ntts = params.num_ntts();
        let leaf_bytes = num_ntts * core::mem::size_of::<F128>();
        let hash_kind = params.merkle_hash;
        let mut merkle_tree: Vec<merkle::Hash> = crate::scratch::take_hash(2 * n_leaves - 1);
        let tree_base = merkle_tree.as_mut_ptr() as usize;
        let folded_depth = std::sync::atomic::AtomicUsize::new(usize::MAX);

        ntt.forward_transform_interleaved_from_replicated_msg_on_sub_done(
            &mut codeword,
            z_packed,
            num_ntts,
            params.log_inv_rate,
            &|range, sub_data| {
                debug_assert!(range.len().is_power_of_two() && range.start % range.len() == 0);
                let sub_bytes: &[u8] = unsafe {
                    core::slice::from_raw_parts(
                        sub_data.as_ptr() as *const u8,
                        sub_data.len() * core::mem::size_of::<F128>(),
                    )
                };
                let leaves_out: &mut [merkle::Hash] = unsafe {
                    core::slice::from_raw_parts_mut(
                        (tree_base as *mut merkle::Hash).add(range.start),
                        range.len(),
                    )
                };
                merkle::hash_leaves_serial(sub_bytes, leaf_bytes, leaves_out, hash_kind);

                let depth = range.len().trailing_zeros() as usize;
                folded_depth.fetch_min(depth, std::sync::atomic::Ordering::Relaxed);

                let mut prev_start = range.start;
                let mut prev_len = range.len();
                let mut prev_level_start = 0usize;
                let mut level_start = n_leaves;
                let mut level_len = n_leaves >> 1;
                for _ in 0..depth {
                    let parent_offset = (prev_start - prev_level_start) >> 1;
                    let parent_start = level_start + parent_offset;
                    let parent_len = prev_len >> 1;
                    let (children, parents) = unsafe {
                        let base = tree_base as *mut merkle::Hash;
                        (
                            core::slice::from_raw_parts(base.add(prev_start), prev_len),
                            core::slice::from_raw_parts_mut(base.add(parent_start), parent_len),
                        )
                    };
                    merkle::hash_pairs_level_serial(children, parents, hash_kind);
                    prev_level_start = level_start;
                    prev_start = parent_start;
                    prev_len = parent_len;
                    level_start += level_len;
                    level_len >>= 1;
                }
            },
        );
        if timing {
            eprintln!(
                "[commit-timing] ntt (incl. L2 merkle subtrees): {:.2} ms",
                t_ntt.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_merkle = std::time::Instant::now();
        let depth_done = folded_depth.load(std::sync::atomic::Ordering::Relaxed);
        merkle::build_upper_levels(&mut merkle_tree, n_leaves, depth_done, hash_kind);
        let root = *merkle_tree.last().expect("merkle tree non-empty");
        if timing {
            eprintln!(
                "[commit-timing] merkle: {:.2} ms",
                t_merkle.elapsed().as_secs_f64() * 1e3
            );
        }
        (
            Commitment {
                root,
                params: params.clone(),
            },
            ProverData {
                codeword,
                merkle_tree,
            },
        )
    }
}

#[cfg(target_os = "macos")]
fn set_background_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x09, 0);
    }
}
#[cfg(not(target_os = "macos"))]
fn set_background_qos() {}

pub fn prefault_codeword_during<R>(
    params: &PcsParams,
    generate: impl FnOnce() -> R,
) -> (Option<Vec<F128>>, R) {
    if rayon::current_num_threads() <= 1 || std::env::var_os("FLOCK_NO_PREFAULT").is_some() {
        return (None, generate());
    }
    let codeword_len = params.n_positions() * params.num_ntts();
    if let Some(buf) = crate::scratch::try_take_f128(codeword_len) {
        return (Some(buf), generate());
    }
    std::thread::scope(|s| {
        let h = s.spawn(move || {
            set_background_qos();
            let mut buf: Vec<F128> = crate::alloc_uninit_f128_vec(codeword_len);
            unsafe {
                std::ptr::write_bytes(buf.as_mut_ptr(), 0u8, codeword_len);
            }
            buf
        });
        let r = generate();
        (Some(h.join().unwrap()), r)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
    }

    #[test]
    fn ligerito_configs_inherit_the_params_merkle_hash() {
        let mut params = default_params(22);
        params.log_batch_size = 6;

        assert_eq!(params.merkle_hash, HashKind::Sha256);
        assert_eq!(
            params.ligerito_prover_config().unwrap().merkle_hash,
            HashKind::Sha256
        );

        params.merkle_hash = HashKind::Blake3;
        assert_eq!(
            params.ligerito_prover_config().unwrap().merkle_hash,
            HashKind::Blake3,
            "prover config must follow PcsParams, not the embedded TOML"
        );
        assert_eq!(
            params.ligerito_verifier_config().unwrap().merkle_hash,
            HashKind::Blake3,
            "verifier config must follow PcsParams, not the embedded TOML"
        );
    }

    fn default_params(m: usize) -> PcsParams {
        PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: 1,
            profile: Default::default(),
            merkle_hash: Default::default(),
        }
    }

    #[test]
    fn commit_matches_full_ntt_oracle() {
        use crate::ntt::AdditiveNttF128;
        let mut rng = Rng::new(0xFEED);
        for (m, log_inv_rate, log_batch_size) in [(10, 1, 1), (12, 1, 2), (12, 2, 1), (14, 2, 3)] {
            let params = PcsParams {
                m,
                log_inv_rate,
                log_batch_size,
                profile: Default::default(),
                merkle_hash: Default::default(),
            };
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);

            let (commitment, pd) = commit(&z_packed, &params);

            let mut oracle = vec![F128::ZERO; params.codeword_len_f128()];
            oracle[..z_packed.len()].copy_from_slice(&z_packed);
            let ntt = AdditiveNttF128::standard(params.k_code());
            ntt.forward_transform_interleaved(&mut oracle, params.num_ntts());

            assert_eq!(
                pd.codeword, oracle,
                "codeword mismatch at m={m} r={log_inv_rate}"
            );
            let oracle_bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(oracle.as_ptr() as *const u8, oracle.len() * 16)
            };
            let oracle_root =
                *crate::merkle::merkle_tree(oracle_bytes, params.n_leaves(), params.merkle_hash)
                    .last()
                    .unwrap();
            assert_eq!(
                commitment.root, oracle_root,
                "root mismatch at m={m} r={log_inv_rate}"
            );
        }
    }

    #[test]
    fn commit_runs_and_produces_root() {
        let mut rng = Rng::new(42);
        for m in [8usize, 10, 12] {
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);
            let params = default_params(m);
            let (commitment, prover_data) = commit(&z_packed, &params);
            assert_eq!(prover_data.codeword.len(), params.codeword_len_f128());
            assert_eq!(
                prover_data.merkle_tree.last().copied().unwrap(),
                commitment.root
            );
            assert_eq!(z_packed.len(), 1 << params.log_msg_len());
        }
    }

    #[test]
    fn commit_is_deterministic() {
        let mut rng = Rng::new(7);
        let m = 10;
        let z = rng.bits(1 << m);
        let z_packed = super::super::pack::pack_witness(&z, m);
        let params = default_params(m);
        let (c1, _) = commit(&z_packed, &params);
        let (c2, _) = commit(&z_packed, &params);
        assert_eq!(c1.root, c2.root);
    }

    #[test]
    fn commit_root_sensitive_to_witness() {
        let mut rng = Rng::new(99);
        let m = 10;
        let mut z = rng.bits(1 << m);
        let params = default_params(m);
        let (c1, _) = commit(&super::super::pack::pack_witness(&z, m), &params);
        z[7] ^= true;
        let (c2, _) = commit(&super::super::pack::pack_witness(&z, m), &params);
        assert_ne!(c1.root, c2.root);
    }

    #[test]
    fn rs_encoding_is_linear() {
        let mut rng = Rng::new(123);
        let m = 9;
        let params = default_params(m);
        let z1 = rng.bits(1 << m);
        let z2 = rng.bits(1 << m);
        let z_xor: Vec<bool> = z1.iter().zip(&z2).map(|(a, b)| a ^ b).collect();
        let pack = |z: &[bool]| super::super::pack::pack_witness(z, m);
        let (_, pd1) = commit(&pack(&z1), &params);
        let (_, pd2) = commit(&pack(&z2), &params);
        let (_, pd_x) = commit(&pack(&z_xor), &params);
        for (i, (&c1, &c2)) in pd1.codeword.iter().zip(&pd2.codeword).enumerate() {
            assert_eq!(c1 + c2, pd_x.codeword[i], "linearity fails at i={i}");
        }
    }

    #[test]
    fn codeword_doubles_message_length() {
        let mut rng = Rng::new(2);
        let m = 10;
        let params = default_params(m);
        let z = rng.bits(1 << m);
        let z_packed = super::super::pack::pack_witness(&z, m);
        let (_, pd) = commit(&z_packed, &params);
        assert_eq!(pd.codeword.len(), 2 * z_packed.len());
    }
}
