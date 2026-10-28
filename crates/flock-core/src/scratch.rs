use crate::field::F128;
use std::sync::Mutex;

static POOL: Mutex<Vec<Vec<F128>>> = Mutex::new(Vec::new());
static U8_POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static HASH_POOL: Mutex<Vec<Vec<[u8; 32]>>> = Mutex::new(Vec::new());

const MAX_POOLED: usize = 24;

pub fn take_f128(n: usize) -> Vec<F128> {
    if let Some(v) = try_take_f128(n) {
        return v;
    }
    crate::alloc_uninit_vec(n)
}

pub(crate) fn try_take_f128(n: usize) -> Option<Vec<F128>> {
    let mut pool = POOL.lock().unwrap();
    let mut best: Option<usize> = None;
    for (i, v) in pool.iter().enumerate() {
        if v.capacity() >= n && best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
            best = Some(i);
        }
    }
    if let Some(i) = best {
        let mut v = pool.swap_remove(i);
        drop(pool);
        v.clear();
        unsafe { v.set_len(n) };
        return Some(v);
    }
    None
}

pub fn give_f128(v: Vec<F128>) {
    const MAX_POOLED_F128_ELEMS: usize = 200 * (1 << 20);
    if v.capacity() == 0 {
        return;
    }
    let mut pool = POOL.lock().unwrap();
    pool.push(v);
    let mut total_elems: usize = pool.iter().map(|b| b.capacity()).sum();
    while pool.len() > MAX_POOLED || (pool.len() > 1 && total_elems > MAX_POOLED_F128_ELEMS) {
        let smallest = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| b.capacity())
            .map(|(i, _)| i)
            .expect("pool non-empty");
        let removed = pool.swap_remove(smallest);
        total_elems -= removed.capacity();
    }
}

pub fn take_u8(n: usize) -> Vec<u8> {
    let mut pool = U8_POOL.lock().unwrap();
    let mut best: Option<usize> = None;
    for (i, v) in pool.iter().enumerate() {
        if v.capacity() >= n && best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
            best = Some(i);
        }
    }
    if let Some(i) = best {
        let mut v = pool.swap_remove(i);
        drop(pool);
        v.clear();
        unsafe { v.set_len(n) };
        return v;
    }
    drop(pool);
    crate::alloc_uninit_vec(n)
}

pub fn give_u8(v: Vec<u8>) {
    const MAX_POOLED_U8: usize = 2;
    const MAX_POOLED_U8_BYTES: usize = 512 * (1 << 20);
    if v.capacity() == 0 {
        return;
    }
    let mut pool = U8_POOL.lock().unwrap();
    pool.push(v);
    let mut total_bytes: usize = pool.iter().map(|b| b.capacity()).sum();
    while pool.len() > MAX_POOLED_U8 || (pool.len() > 1 && total_bytes > MAX_POOLED_U8_BYTES) {
        let smallest = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| b.capacity())
            .map(|(i, _)| i)
            .expect("pool non-empty");
        let removed = pool.swap_remove(smallest);
        total_bytes -= removed.capacity();
    }
}

pub fn take_hash(n: usize) -> Vec<[u8; 32]> {
    let mut pool = HASH_POOL.lock().unwrap();
    let mut best: Option<usize> = None;
    for (i, v) in pool.iter().enumerate() {
        if v.capacity() >= n && best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
            best = Some(i);
        }
    }
    if let Some(i) = best {
        let mut v = pool.swap_remove(i);
        drop(pool);
        v.clear();
        unsafe { v.set_len(n) };
        return v;
    }
    drop(pool);
    crate::alloc_uninit_vec(n)
}

pub fn give_hash(v: Vec<[u8; 32]>) {
    const MAX_POOLED_HASH: usize = 4;
    const MAX_POOLED_HASH_ELEMS: usize = 4 * (1 << 20);
    if v.capacity() == 0 {
        return;
    }
    let mut pool = HASH_POOL.lock().unwrap();
    pool.push(v);
    let mut total_elems: usize = pool.iter().map(|b| b.capacity()).sum();
    while pool.len() > MAX_POOLED_HASH || (pool.len() > 1 && total_elems > MAX_POOLED_HASH_ELEMS) {
        let smallest = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| b.capacity())
            .map(|(i, _)| i)
            .expect("pool non-empty");
        let removed = pool.swap_remove(smallest);
        total_elems -= removed.capacity();
    }
}

pub fn prewarm_prover(m: usize) {
    use rayon::prelude::*;
    if !(8..=32).contains(&m) {
        return;
    }
    let half = 1usize << (m - 8);
    let small = 1usize << (m - 7);
    let large = 1usize << (m - 6);
    let mut bufs: Vec<Vec<F128>> = Vec::with_capacity(6);
    bufs.push(take_f128(large));
    for _ in 0..3 {
        bufs.push(take_f128(small));
    }
    for _ in 0..2 {
        bufs.push(take_f128(half));
    }
    let mut stripe = take_u8(1usize << (m - 3));
    bufs.par_iter_mut().for_each(|b| {
        b.par_chunks_mut(1 << 16).for_each(|chunk| unsafe {
            std::ptr::write_bytes(chunk.as_mut_ptr(), 0u8, chunk.len())
        });
    });
    stripe.par_chunks_mut(1 << 20).for_each(|chunk| {
        chunk.fill(0);
    });
    for b in bufs {
        give_f128(b);
    }
    give_u8(stripe);
    if m >= 13 {
        let mut h = take_hash((1usize << (m - 11)) - 1);
        h.par_chunks_mut(1 << 16).for_each(|chunk| {
            chunk.fill([0u8; 32]);
        });
        give_hash(h);
    }
}

pub fn clear() {
    POOL.lock().unwrap().clear();
    U8_POOL.lock().unwrap().clear();
    HASH_POOL.lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_reuses_given_buffer() {
        clear();
        let mut v = take_f128(1024);
        for slot in v.iter_mut() {
            *slot = F128 { lo: 7, hi: 9 };
        }
        let ptr = v.as_ptr();
        give_f128(v);
        let v2 = take_f128(512);
        assert_eq!(v2.as_ptr(), ptr);
        assert_eq!(v2.len(), 512);
        clear();
    }

    #[test]
    fn pool_is_bounded() {
        clear();
        for _ in 0..(MAX_POOLED + 4) {
            give_f128(take_f128(16));
        }
        assert!(POOL.lock().unwrap().len() <= MAX_POOLED);
        clear();
    }
}
