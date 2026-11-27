use crate::epool::SyncPtr;
use crate::field::F128;
use crate::ntt::AdditiveNttF128;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

const SUB: usize = 8192 * 8;

pub(super) struct Early {
    mat: Vec<F128>,
    ntt: AdditiveNttF128,
    helper: &'static rayon::ThreadPool,
    ready: AtomicUsize,
    claimed: AtomicUsize,
    complete: AtomicU8,
}

pub(super) struct Lease<'a> {
    mat: SyncPtr<F128>,
    prefix: SyncPtr<F128>,
    ntt: &'a AdditiveNttF128,
    ready: &'a AtomicUsize,
    claimed: &'a AtomicUsize,
    complete: &'a AtomicU8,
}

pub(super) struct Prepared {
    pub(super) mat: Vec<F128>,
    pub(super) complete: u8,
    pub(super) helper: &'static rayon::ThreadPool,
}

impl Early {
    pub(super) fn new(ranked: bool) -> Option<Self> {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !ranked
            // On only after the ordinary warm L1 overlap race has joined.
            || crate::gpu_commit::l1_overlap_latch_state() != 1
            || !*ENABLED.get_or_init(|| {
                std::env::var_os("FLOCK_NO_FOLD8_L1_EARLY_CPU").as_deref()
                    != Some(std::ffi::OsStr::new("1"))
                    && std::env::var_os("FLOCK_NO_RECURSIVE_FROM_MESSAGE_L1").is_none()
                    && super::recursive_from_message_env_clear()
                    && super::zfold_gpu_enabled()
            })
        {
            return None;
        }
        let helper = AdditiveNttF128::early_l1_helper()?;
        Some(Self {
            mat: crate::scratch::take_f128(1 << 21),
            ntt: AdditiveNttF128::standard(18),
            helper,
            ready: AtomicUsize::new(0),
            claimed: AtomicUsize::new(0),
            complete: AtomicU8::new(0),
        })
    }

    pub(super) fn lease(&mut self, prefix: SyncPtr<F128>) -> Lease<'_> {
        Lease {
            mat: SyncPtr(self.mat.as_mut_ptr()),
            prefix,
            ntt: &self.ntt,
            ready: &self.ready,
            claimed: &self.claimed,
            complete: &self.complete,
        }
    }

    pub(super) fn finish(self) -> Prepared {
        Prepared {
            mat: self.mat,
            complete: self.complete.into_inner(),
            helper: self.helper,
        }
    }
}

impl Lease<'_> {
    pub(super) fn block_done(&self, block: usize) {
        if block < 128 {
            // Every writer has ended its exclusive borrow before publishing.
            // The AcqRel chain publishes all first 128 blocks to the tail.
            self.ready.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub(super) fn try_tail(&self) {
        if self.ready.load(Ordering::Acquire) != 128 {
            return;
        }
        let task = self.claimed.fetch_add(1, Ordering::Relaxed);
        let mask = match task {
            0 => 1,
            1 => 2,
            2 => 12,
            _ => return,
        };
        // SAFETY: the Acquire above covers every write to the first four
        // octants. The tail tasks claim separate output ranges. The Fold8
        // join precedes any other borrow of the codeword matrix.
        unsafe {
            let prefix = std::slice::from_raw_parts(self.prefix.ptr(), 4 * SUB);
            let output = self.mat.ptr().add(task * SUB);
            self.ntt.early_l1_zero_root_subtrees(prefix, output, task);
        }
        self.complete.fetch_or(mask, Ordering::Release);
    }
}
