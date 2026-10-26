use super::*;

#[derive(Clone, Copy)]
enum Action {
    Skip,
    Full(usize),
    Start,
    End(usize, u64),
    Tail(usize, u64),
}

#[derive(Clone, Copy, Default)]
struct FullOp {
    source: u32,
    out: u32,
}
#[derive(Clone, Copy, Default)]
struct PairOp {
    source: u32,
    out: u32,
    mask: u64,
}
#[derive(Clone, Copy)]
struct Batch {
    full_start: u16,
    full_end: u16,
    pair_start: u16,
    pair_end: u16,
    leading_out: u16,
    trailing_source: u16,
    tail_source: u16,
    tail_out: u16,
    leading_mask: u64,
    tail_mask: u64,
}
impl Default for Batch {
    fn default() -> Self {
        Self {
            full_start: 0,
            full_end: 0,
            pair_start: 0,
            pair_end: 0,
            leading_out: u16::MAX,
            trailing_source: u16::MAX,
            tail_source: u16::MAX,
            tail_out: 0,
            leading_mask: 0,
            tail_mask: 0,
        }
    }
}

pub(crate) struct RootEmitPlan {
    actions: [Action; 256],
    full: [FullOp; 119],
    pairs: [PairOp; 56],
    batches: [Batch; 17],
}

impl RootEmitPlan {
    pub(crate) fn new(roots: &[usize]) -> Option<Self> {
        if roots.len() != 11313
            || roots.last() != Some(&15088)
            || !roots.windows(2).all(|pair| pair[0] < pair[1])
            || !roots[..897]
                .iter()
                .enumerate()
                .all(|(i, &wire)| wire == i + if i >= 256 { 256 } else { 0 })
        {
            return None;
        }

        let roots: Vec<usize> = roots
            .iter()
            .copied()
            .filter(|wire| !(1089..1153).contains(wire))
            .collect();
        let mut actions = [Action::Skip; 256];
        let mut pairs = 0;
        for (out, group) in roots.chunks(64).enumerate() {
            let first = group[0];
            let source = first / 64;
            if first % 64 != 0 || !matches!(actions[source], Action::Skip) {
                return None;
            }
            if group.len() < 64 {
                if group.len() != 49
                    || !group.iter().enumerate().all(|(i, &wire)| wire == first + i)
                {
                    return None;
                }
                actions[source] = Action::Tail(out, (1u64 << 49) - 1);
            } else if group[63] / 64 == source {
                if !group.iter().enumerate().all(|(i, &wire)| wire == first + i) {
                    return None;
                }
                actions[source] = Action::Full(out);
            } else {
                if group[63] / 64 != source + 1 || !matches!(actions[source + 1], Action::Skip) {
                    return None;
                }
                let split = group.iter().position(|&wire| wire >= first + 64)?;
                if split == 0
                    || !group
                        .iter()
                        .enumerate()
                        .all(|(i, &wire)| wire == first + i + if i >= split { 64 } else { 0 })
                {
                    return None;
                }
                actions[source] = Action::Start;
                actions[source + 1] = Action::End(out, (1u64 << split) - 1);
                pairs += 1;
            }
        }
        if pairs != 56 {
            return None;
        }
        let mut full = [FullOp::default(); 119];
        let mut joined = [PairOp::default(); 56];
        let mut batches = [Batch::default(); 17];
        let (mut nf, mut np) = (0usize, 0usize);
        for (index, batch) in batches.iter_mut().enumerate() {
            let (first, count) = if index < 2 {
                (index * 8, 8)
            } else {
                ((index - 1) * 16, 16)
            };
            batch.full_start = nf as u16;
            batch.pair_start = np as u16;
            for source in first..first + count {
                let offset = ((source - first) * 64) as u32;
                match actions[source] {
                    Action::Skip => {}
                    Action::Full(out) => {
                        full[nf] = FullOp {
                            source: offset,
                            out: (out * 512) as u32,
                        };
                        nf += 1;
                    }
                    Action::Start if source + 1 < first + count => {
                        let Action::End(out, mask) = actions[source + 1] else {
                            return None;
                        };
                        joined[np] = PairOp {
                            source: offset,
                            out: (out * 512) as u32,
                            mask,
                        };
                        np += 1;
                    }
                    Action::Start => batch.trailing_source = offset as u16,
                    Action::End(out, mask) if source == first => {
                        batch.leading_out = out as u16;
                        batch.leading_mask = mask;
                    }
                    Action::End(_, _) => {}
                    Action::Tail(out, mask) => {
                        batch.tail_source = offset as u16;
                        batch.tail_out = out as u16;
                        batch.tail_mask = mask;
                    }
                }
            }
            batch.full_end = nf as u16;
            batch.pair_end = np as u16;
        }
        debug_assert_eq!(nf, 119);
        Some(Self {
            actions,
            full,
            pairs: joined,
            batches,
        })
    }
}

pub(crate) struct RootOctaWriter<'a> {
    plan: &'a RootEmitPlan,
    out: *mut u8,
    carry: __m512i,
    pending: bool,
}

impl<'a> RootOctaWriter<'a> {
    pub(crate) unsafe fn new(plan: &'a RootEmitPlan, out: *mut u8) -> Self {
        debug_assert_eq!(out as usize % 64, 0);
        Self {
            plan,
            out,
            carry: unsafe { _mm512_setzero_si512() },
            pending: false,
        }
    }

    #[inline(never)]
    unsafe fn emit_generic(
        &mut self,
        base_word: usize,
        ring_word: usize,
        words: usize,
        a: *const V8,
        b: *const V8,
    ) {
        debug_assert_eq!(base_word % 2, 0);
        debug_assert_eq!(words % 2, 0);
        unsafe {
            for offset in (0..words).step_by(2) {
                let source = (base_word + offset) / 2;
                let action = self.plan.actions[source];
                if matches!(action, Action::Skip) {
                    continue;
                }
                let rw = ring_word + offset;
                let mut value = _mm512_loadu_si512(a.add(rw).cast());
                if base_word + offset >= 32 {
                    value = _mm512_and_si512(value, _mm512_loadu_si512(b.add(rw).cast()));
                }
                let (out, mask) = match action {
                    Action::Skip => unreachable!(),
                    Action::Start => {
                        debug_assert!(!self.pending);
                        self.carry = value;
                        self.pending = true;
                        continue;
                    }
                    Action::End(out, mask) => {
                        debug_assert!(self.pending);
                        let lo = _mm256_set1_epi32(mask as u32 as i32);
                        let hi = _mm256_set1_epi32((mask >> 32) as u32 as i32);
                        let select = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
                        value = _mm512_ternarylogic_epi64::<0xca>(select, self.carry, value);
                        self.pending = false;
                        (out, u64::MAX)
                    }
                    Action::Full(out) => (out, u64::MAX),
                    Action::Tail(out, mask) => (out, mask),
                };
                let lo = _mm256_set1_epi32(mask as u32 as i32);
                let hi = _mm256_set1_epi32((mask >> 32) as u32 as i32);
                let words = _mm512_and_si512(
                    value,
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi),
                );
                _mm512_stream_si512(self.out.add(out * 512).cast(), words);
            }
        }
    }
    #[inline(always)]
    unsafe fn load_pair<const STATIC_B: bool>(
        a: *const u8,
        b: *const u8,
        offset: usize,
    ) -> __m512i {
        unsafe {
            let x = _mm512_loadu_si512(a.add(offset).cast());
            if STATIC_B {
                x
            } else {
                _mm512_and_si512(x, _mm512_loadu_si512(b.add(offset).cast()))
            }
        }
    }

    #[inline(always)]
    unsafe fn select_pair(mask: u64, first: __m512i, second: __m512i) -> __m512i {
        unsafe {
            let lo = _mm256_set1_epi32(mask as u32 as i32);
            let hi = _mm256_set1_epi32((mask >> 32) as u32 as i32);
            let select = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
            _mm512_ternarylogic_epi64::<0xca>(select, first, second)
        }
    }

    #[inline(always)]
    unsafe fn emit_batch<const STATIC_B: bool>(
        &mut self,
        index: usize,
        a: *const V8,
        b: *const V8,
    ) {
        unsafe {
            let batch = &self.plan.batches[index];
            let emit = |out: usize, value| {
                _mm512_stream_si512(self.out.add(out).cast(), value);
            };
            let (a, b) = (a.cast::<u8>(), b.cast::<u8>());
            if batch.leading_out != u16::MAX {
                debug_assert!(self.pending);
                let next = Self::load_pair::<STATIC_B>(a, b, 0);
                emit(
                    batch.leading_out as usize * 512,
                    Self::select_pair(batch.leading_mask, self.carry, next),
                );
                self.pending = false;
            }
            for op in &self.plan.full[batch.full_start as usize..batch.full_end as usize] {
                emit(
                    op.out as usize,
                    Self::load_pair::<STATIC_B>(a, b, op.source as usize),
                );
            }
            for op in &self.plan.pairs[batch.pair_start as usize..batch.pair_end as usize] {
                debug_assert!(!self.pending);
                let first = Self::load_pair::<STATIC_B>(a, b, op.source as usize);
                let second = Self::load_pair::<STATIC_B>(a, b, op.source as usize + 64);
                emit(op.out as usize, Self::select_pair(op.mask, first, second));
            }
            if batch.tail_source != u16::MAX {
                let value = Self::load_pair::<STATIC_B>(a, b, batch.tail_source as usize);
                let lo = _mm256_set1_epi32(batch.tail_mask as u32 as i32);
                let hi = _mm256_set1_epi32((batch.tail_mask >> 32) as u32 as i32);
                let value = _mm512_and_si512(
                    value,
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi),
                );
                emit(batch.tail_out as usize * 512, value);
            }
            if batch.trailing_source != u16::MAX {
                debug_assert!(!self.pending);
                self.carry = Self::load_pair::<STATIC_B>(a, b, batch.trailing_source as usize);
                self.pending = true;
            }
        }
    }

    #[inline(never)]
    pub(crate) unsafe fn emit_range(
        &mut self,
        base_word: usize,
        ring_word: usize,
        words: usize,
        a: *const V8,
        b: *const V8,
    ) {
        unsafe {
            if base_word >= 32
                && base_word <= 480
                && base_word % 32 == 0
                && ring_word == 0
                && words == 32
            {
                self.emit_batch::<false>(base_word / 32 + 1, a, b);
            } else if base_word == 0 && ring_word == 0 && words == 16 {
                self.emit_batch::<true>(0, a, b);
            } else if base_word == 16 && ring_word == 16 && words == 16 {
                self.emit_batch::<true>(1, a.add(16), b.add(16));
            } else {
                self.emit_generic(base_word, ring_word, words, a, b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_root_selectors_fall_back() {
        assert!(RootEmitPlan::new(&[0, 1, 2]).is_none());
        let mut roots = super::super::super::Blake3AdjointPlan::build()
            .linear_fold_plan()
            .roots;
        roots[1024] = roots[1023];
        assert!(RootEmitPlan::new(&roots).is_none());
    }

    #[test]
    fn live_root_stripe_matches_canonical_bits_and_preserves_neighbors() {
        let fold = super::super::super::Blake3AdjointPlan::build().linear_fold_plan();
        assert_eq!(fold.roots.len(), 11313);

        for (wire, root) in [
            (512, 256),
            (1088, 832),
            (1089, 833),
            (1095, 839),
            (1121, 865),
            (1122, 866),
            (1124, 868),
            (1152, 896),
            (1153, 897),
        ] {
            assert_eq!(fold.roots[root], wire);
        }
        let plan = RootEmitPlan::new(&fold.roots).unwrap();
        let mut seed = 0x4a2e_8193_c17d_b805u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        };
        let a: [[u32; 8]; 512] = core::array::from_fn(|_| core::array::from_fn(|_| rng()));
        let b: [[u32; 8]; 512] = core::array::from_fn(|word| {
            core::array::from_fn(|_| if word < 32 { u32::MAX } else { rng() })
        });
        #[repr(align(64))]
        struct Tile([u8; 176 * 512]);
        let mut out = Box::new(Tile([0xa5; 176 * 512]));
        unsafe {
            let mut writer = RootOctaWriter::new(&plan, out.0.as_mut_ptr().add(3 * 64));

            for base in (16..512).step_by(16) {
                writer.emit_range(base, base, 16, a.as_ptr().cast(), b.as_ptr().cast());
            }
            writer.emit_range(0, 0, 16, a.as_ptr().cast(), b.as_ptr().cast());
            _mm_sfence();
        }
        for group in 0..176 {
            for stripe in 0..8 {
                for column in 0..64 {
                    let expected = if stripe != 3 {
                        0xa5
                    } else {
                        let lane = column % 32 / 4;
                        (0..8).fold(0u8, |byte, bit| {
                            let value = fold
                                .roots
                                .get({
                                    let index =
                                        group * 64 + column / 32 * 32 + column % 4 * 8 + bit;
                                    index + if index >= 833 { 64 } else { 0 }
                                })
                                .map_or(0, |&wire| {
                                    (a[wire / 32][lane] & b[wire / 32][lane]) >> (wire % 32) & 1
                                });
                            byte | (value as u8) << bit
                        })
                    };
                    assert_eq!(
                        out.0[group * 512 + stripe * 64 + column],
                        expected,
                        "group={group}, stripe={stripe}, column={column}"
                    );
                }
            }
        }
    }
}
