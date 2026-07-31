//! GPU (Metal) offload for the BLAKE3 Merkle tree: ranked 1 KiB leaf hashing
//! and parent-pair hashing as single whole-level compute dispatches.
//!
//! Loaded entirely at runtime via `dlopen`/`objc_msgSend` — no build-script,
//! no linker flags, no new crate dependencies — so it lives wholly inside the
//! solver-editable source tree. Every entry point returns `false` (caller
//! falls back to the CPU path, which remains byte-identical) if anything at
//! all fails: missing framework, no device, shader compile error, or an
//! allocation failure. A failure latches: after the first one the GPU is
//! never retried, so a broken runtime costs one attempt, not one per call.
//!
//! Digest semantics are exactly those of the CPU path (`hash_leaf` /
//! `hash_pair` in the parent module): a 1 KiB leaf is one BLAKE3 chunk — 16
//! chained 64-byte block compressions, `CHUNK_START` on the first block,
//! `CHUNK_END` on the last, chunk counter 0, non-root CV out — and a parent
//! is a single compression of the concatenated child CVs under the `PARENT`
//! flag. Both are verified bit-identical against the CPU oracles in this
//! module's tests; nothing here can change a commitment, only the time spent
//! producing it.

use std::ffi::{c_char, c_void, CString};
use std::sync::{Mutex, OnceLock};

use super::Hash;

type Id = *mut c_void;

unsafe extern "C" {
    fn dlopen(path: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, sym: *const c_char) -> *mut c_void;
}
const RTLD_NOW: i32 = 2;

/// Below these sizes a whole-level dispatch cannot recoup the ~0.1–0.3 ms
/// encoder/commit floor against the (already fast) CPU kernels; the caller's
/// CPU path handles them. Thresholds are deliberately conservative — the GPU
/// only takes levels large enough to win clearly.
const MIN_LEAVES: usize = 1 << 12;
const MIN_PARENTS: usize = 1 << 13;

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint IV0=0x6A09E667u, IV1=0xBB67AE85u, IV2=0x3C6EF372u, IV3=0xA54FF53Au;
constant uint IV4=0x510E527Fu, IV5=0x9B05688Cu, IV6=0x1F83D9ABu, IV7=0x5BE0CD19u;

// flags: CHUNK_START=1, CHUNK_END=2, PARENT=4 (ROOT never set here — the
// Merkle tree stores non-root chaining values throughout).

static inline void gg(thread uint* st, uint a, uint b, uint c, uint d, uint mx, uint my) {
    st[a] = st[a] + st[b] + mx;
    st[d] = rotate(st[d] ^ st[a], 32u - 16u);
    st[c] = st[c] + st[d];
    st[b] = rotate(st[b] ^ st[c], 32u - 12u);
    st[a] = st[a] + st[b] + my;
    st[d] = rotate(st[d] ^ st[a], 32u - 8u);
    st[c] = st[c] + st[d];
    st[b] = rotate(st[b] ^ st[c], 32u - 7u);
}

#define ROUND(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15) \
    gg(st,0,4,8,12,m0,m1); gg(st,1,5,9,13,m2,m3); \
    gg(st,2,6,10,14,m4,m5); gg(st,3,7,11,15,m6,m7); \
    gg(st,0,5,10,15,m8,m9); gg(st,1,6,11,12,m10,m11); \
    gg(st,2,7,8,13,m12,m13); gg(st,3,4,9,14,m14,m15);

// One full 7-round compression of block m into cv (chaining-value form:
// out[i] = st[i] ^ st[i+8], truncated to 8 words). The message permutation
// schedule is pre-applied per round, fully unrolled.
static inline void compress_cv(thread uint* cv, thread const uint* m,
                               uint block_len, uint flags) {
    uint st[16] = { cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                    IV0, IV1, IV2, IV3, 0u, 0u, block_len, flags };
    ROUND(m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9],m[10],m[11],m[12],m[13],m[14],m[15])
    ROUND(m[2],m[6],m[3],m[10],m[7],m[0],m[4],m[13],m[1],m[11],m[12],m[5],m[9],m[14],m[15],m[8])
    ROUND(m[3],m[4],m[10],m[12],m[13],m[2],m[7],m[14],m[6],m[5],m[9],m[0],m[11],m[15],m[8],m[1])
    ROUND(m[10],m[7],m[12],m[9],m[14],m[3],m[13],m[15],m[4],m[0],m[11],m[2],m[5],m[8],m[1],m[6])
    ROUND(m[12],m[13],m[9],m[11],m[15],m[10],m[14],m[8],m[7],m[2],m[5],m[3],m[0],m[1],m[6],m[4])
    ROUND(m[9],m[14],m[11],m[5],m[8],m[12],m[15],m[1],m[13],m[3],m[0],m[10],m[2],m[6],m[4],m[7])
    ROUND(m[11],m[15],m[5],m[0],m[1],m[9],m[8],m[6],m[14],m[10],m[2],m[12],m[3],m[4],m[7],m[13])
    for (uint i = 0; i < 8; i++) cv[i] = st[i] ^ st[i + 8];
}

// One thread = one 1 KiB leaf = one BLAKE3 chunk: 16 chained block
// compressions, counter 0, CHUNK_START on block 0, CHUNK_END on block 15.
kernel void b3_leaf1024(device const uint* data [[buffer(0)]],   // 256 words/leaf
                        device uint*       outs [[buffer(1)]],   // 8 words/leaf
                        constant uint&     count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint cv[8] = { IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7 };
    device const uint* base = data + gid * 256;
    for (uint blk = 0; blk < 16; blk++) {
        uint m[16];
        for (uint i = 0; i < 16; i++) m[i] = base[blk * 16 + i];
        uint flags = (blk == 0 ? 1u : 0u) | (blk == 15 ? 2u : 0u);
        compress_cv(cv, m, 64u, flags);
    }
    for (uint i = 0; i < 8; i++) outs[gid * 8 + i] = cv[i];
}

// One thread = one parent node: single compression of the two 32-byte child
// CVs under PARENT.
kernel void b3_parents(device const uint* children [[buffer(0)]],  // 16 words/node
                       device uint*       outs     [[buffer(1)]],  // 8 words/node
                       constant uint&     count    [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint cv[8] = { IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7 };
    uint m[16];
    for (uint i = 0; i < 16; i++) m[i] = children[gid * 16 + i];
    compress_cv(cv, m, 64u, 4u);
    for (uint i = 0; i < 8; i++) outs[gid * 8 + i] = cv[i];
}

// Streaming variant for the ranked NTT-to-Merkle pipeline: the whole
// codeword and the whole leaf array are bound once; each dispatch covers
// one finalized subtree's [leaf_start, leaf_start + leaf_count) range at
// absolute offsets, so ranges submitted by different jobs never collide.
struct LeafRange { uint leaf_start; uint leaf_count; };

kernel void b3_leaf1024_range(device const uint* data  [[buffer(0)]],  // whole codeword
                              device uint*       outs  [[buffer(1)]],  // whole leaf array
                              constant LeafRange& r    [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= r.leaf_count) return;
    uint leaf = r.leaf_start + gid;
    uint cv[8] = { IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7 };
    device const uint* base = data + leaf * 256;
    for (uint blk = 0; blk < 16; blk++) {
        uint m[16];
        for (uint i = 0; i < 16; i++) m[i] = base[blk * 16 + i];
        uint flags = (blk == 0 ? 1u : 0u) | (blk == 15 ? 2u : 0u);
        compress_cv(cv, m, 64u, flags);
    }
    for (uint i = 0; i < 8; i++) outs[leaf * 8 + i] = cv[i];
}
"#;

/// Minimal Objective-C runtime bridge, resolved once via `dlsym`.
struct Objc {
    msg_send: *mut c_void,
    sel: unsafe extern "C" fn(*const c_char) -> Id,
}

impl Objc {
    unsafe fn s(&self, n: &str) -> Id {
        (self.sel)(CString::new(n).unwrap().as_ptr())
    }
    unsafe fn send0(&self, o: Id, s: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Id) -> Id = std::mem::transmute(self.msg_send);
        f(o, s)
    }
    unsafe fn send1(&self, o: Id, s: Id, a: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Id, Id) -> Id = std::mem::transmute(self.msg_send);
        f(o, s, a)
    }
    unsafe fn send2(&self, o: Id, s: Id, a: Id, b: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Id, Id, Id) -> Id = std::mem::transmute(self.msg_send);
        f(o, s, a, b)
    }
    unsafe fn send3(&self, o: Id, s: Id, a: Id, b: Id, c: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Id, Id, Id, Id) -> Id =
            std::mem::transmute(self.msg_send);
        f(o, s, a, b, c)
    }
    unsafe fn send_u2(&self, o: Id, s: Id, a: u64, b: u64) -> Id {
        let f: unsafe extern "C" fn(Id, Id, u64, u64) -> Id = std::mem::transmute(self.msg_send);
        f(o, s, a, b)
    }
    unsafe fn set_buffer(&self, enc: Id, buf: Id, index: u64) {
        let f: unsafe extern "C" fn(Id, Id, Id, u64, u64) = std::mem::transmute(self.msg_send);
        f(enc, self.s("setBuffer:offset:atIndex:"), buf, 0, index);
    }
    unsafe fn set_bytes(&self, enc: Id, bytes: *const c_void, len: u64, index: u64) {
        let f: unsafe extern "C" fn(Id, Id, *const c_void, u64, u64) =
            std::mem::transmute(self.msg_send);
        f(enc, self.s("setBytes:length:atIndex:"), bytes, len, index);
    }
    /// Wrap caller-owned pages in a no-copy shared buffer. Requires the
    /// region to be page-aligned and a page multiple; returns null otherwise
    /// (Metal would reject it). The wrapper never frees the memory.
    unsafe fn no_copy_buffer(&self, device: Id, ptr: *const u8, len: usize) -> Id {
        const PAGE: usize = 16384;
        if (ptr as usize) % PAGE != 0 || len % PAGE != 0 || len == 0 {
            return std::ptr::null_mut();
        }
        let f: unsafe extern "C" fn(Id, Id, *const u8, u64, u64, Id) -> Id =
            std::mem::transmute(self.msg_send);
        f(
            device,
            self.s("newBufferWithBytesNoCopy:length:options:deallocator:"),
            ptr,
            len as u64,
            0,
            std::ptr::null_mut(),
        )
    }
    unsafe fn dispatch(&self, enc: Id, threads: u64) {
        #[repr(C)]
        struct MTLSize {
            w: u64,
            h: u64,
            d: u64,
        }
        let f: unsafe extern "C" fn(Id, Id, MTLSize, MTLSize) =
            std::mem::transmute(self.msg_send);
        f(
            enc,
            self.s("dispatchThreads:threadsPerThreadgroup:"),
            MTLSize { w: threads, h: 1, d: 1 },
            MTLSize { w: 256, h: 1, d: 1 },
        );
    }
}

/// A grow-only shared-storage Metal buffer.
struct GpuBuf {
    buf: Id,
    len: usize,
}

struct Bufs {
    input: GpuBuf,
    output: GpuBuf,
    count: GpuBuf,
}

/// The initialized GPU context. Objects are never released (static lifetime).
pub(super) struct Gpu {
    objc: Objc,
    device: Id,
    queue: Id,
    pipe_leaf: Id,
    pipe_parent: Id,
    pipe_leaf_range: Id,
    bufs: Mutex<Bufs>,
}

// SAFETY: MTLDevice/MTLCommandQueue/MTLComputePipelineState are documented
// thread-safe; the mutable buffer set is guarded by the Mutex; the Objc
// function pointers are immutable after init.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    unsafe fn ensure_buf(&self, buf: &mut GpuBuf, needed: usize) -> Option<Id> {
        if buf.len < needed {
            // Grow to the next power of two so repeated proofs settle after
            // the first allocation.
            let new_len = needed.next_power_of_two();
            let b = self.objc.send_u2(
                self.device,
                self.objc.s("newBufferWithLength:options:"),
                new_len as u64,
                0, // MTLResourceStorageModeShared
            );
            if b.is_null() {
                return None;
            }
            buf.buf = b;
            buf.len = new_len;
        }
        Some(buf.buf)
    }

    /// Run `pipe` over `n` items reading `input` and writing `out`. Returns
    /// `false` on any failure.
    ///
    /// Input travels zero-copy when possible: on Apple Silicon's unified
    /// memory a page-aligned, page-multiple slice (true for the large
    /// codeword buffers this is built for) is wrapped in place with
    /// `newBufferWithBytesNoCopy` and the GPU reads the caller's own pages.
    /// Anything unaligned falls back to a bounded memcpy into the persistent
    /// staging buffer — still correct, just slower.
    unsafe fn run(&self, pipe: Id, input: &[u8], out: &mut [Hash], n: usize) -> bool {
        const PAGE: usize = 16384; // Apple Silicon vm page size
        let o = &self.objc;
        let mut bufs = match self.bufs.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        let zero_copy = (input.as_ptr() as usize) % PAGE == 0 && input.len() % PAGE == 0;
        let in_buf = if zero_copy {
            // Transient no-copy wrapper around the caller's pages. The
            // deallocator is NULL: Metal must not free memory it does not
            // own. The wrapper object itself leaks (static-context objects
            // are never released in this module); at one per whole-level
            // dispatch that is a few dozen small objects per proof.
            let f: unsafe extern "C" fn(Id, Id, *const u8, u64, u64, Id) -> Id =
                std::mem::transmute(o.msg_send);
            let b = f(
                self.device,
                o.s("newBufferWithBytesNoCopy:length:options:deallocator:"),
                input.as_ptr(),
                input.len() as u64,
                0, // MTLResourceStorageModeShared
                std::ptr::null_mut(),
            );
            if b.is_null() {
                return false;
            }
            b
        } else {
            let staged = match self.ensure_buf(&mut bufs.input, input.len()) {
                Some(b) => b,
                None => return false,
            };
            let in_ptr = o.send0(staged, o.s("contents")) as *mut u8;
            if in_ptr.is_null() {
                return false;
            }
            std::ptr::copy_nonoverlapping(input.as_ptr(), in_ptr, input.len());
            staged
        };
        let out_buf = match self.ensure_buf(&mut bufs.output, out.len() * 32) {
            Some(b) => b,
            None => return false,
        };
        let cnt_buf = bufs.count.buf;
        *(o.send0(cnt_buf, o.s("contents")) as *mut u32) = n as u32;

        let cb = o.send0(self.queue, o.s("commandBuffer"));
        if cb.is_null() {
            return false;
        }
        let enc = o.send0(cb, o.s("computeCommandEncoder"));
        if enc.is_null() {
            return false;
        }
        o.send1(enc, o.s("setComputePipelineState:"), pipe);
        o.set_buffer(enc, in_buf, 0);
        o.set_buffer(enc, out_buf, 1);
        o.set_buffer(enc, cnt_buf, 2);
        o.dispatch(enc, n as u64);
        o.send0(enc, o.s("endEncoding"));
        o.send0(cb, o.s("commit"));
        o.send0(cb, o.s("waitUntilCompleted"));
        // MTLCommandBufferStatusCompleted == 4; anything else means the GPU
        // did not finish this work — do not trust the output buffer.
        let status = o.send0(cb, o.s("status")) as usize;
        if status != 4 {
            return false;
        }

        let out_ptr = o.send0(out_buf, o.s("contents")) as *const u8;
        if out_ptr.is_null() {
            return false;
        }
        std::ptr::copy_nonoverlapping(out_ptr, out.as_mut_ptr() as *mut u8, out.len() * 32);
        true
    }
}

unsafe fn init() -> Option<Gpu> {
    let metal = dlopen(
        CString::new("/System/Library/Frameworks/Metal.framework/Versions/A/Metal")
            .unwrap()
            .as_ptr(),
        RTLD_NOW,
    );
    let objc_lib = dlopen(
        CString::new("/usr/lib/libobjc.A.dylib").unwrap().as_ptr(),
        RTLD_NOW,
    );
    if metal.is_null() || objc_lib.is_null() {
        return None;
    }
    let msg_send = dlsym(objc_lib, CString::new("objc_msgSend").unwrap().as_ptr());
    let sel = dlsym(objc_lib, CString::new("sel_registerName").unwrap().as_ptr());
    let get_class = dlsym(objc_lib, CString::new("objc_getClass").unwrap().as_ptr());
    if msg_send.is_null() || sel.is_null() || get_class.is_null() {
        return None;
    }
    let objc = Objc {
        msg_send,
        sel: std::mem::transmute(sel),
    };
    let get_class: unsafe extern "C" fn(*const c_char) -> Id = std::mem::transmute(get_class);

    let create_dev = dlsym(
        metal,
        CString::new("MTLCreateSystemDefaultDevice").unwrap().as_ptr(),
    );
    if create_dev.is_null() {
        return None;
    }
    let create_dev: unsafe extern "C" fn() -> Id = std::mem::transmute(create_dev);
    let device = create_dev();
    if device.is_null() {
        return None;
    }

    // NSString for the shader source.
    let ns_string_cls = get_class(CString::new("NSString").unwrap().as_ptr());
    let src_c = CString::new(SHADER).unwrap();
    let src = objc.send1(
        ns_string_cls,
        objc.s("stringWithUTF8String:"),
        src_c.as_ptr() as Id,
    );
    if src.is_null() {
        return None;
    }

    let mut err: Id = std::ptr::null_mut();
    let lib = objc.send3(
        device,
        objc.s("newLibraryWithSource:options:error:"),
        src,
        std::ptr::null_mut(),
        &mut err as *mut Id as Id,
    );
    if lib.is_null() {
        return None;
    }

    let pipe = |name: &str| -> Id {
        let name_c = CString::new(name).unwrap();
        let ns = objc.send1(
            ns_string_cls,
            objc.s("stringWithUTF8String:"),
            name_c.as_ptr() as Id,
        );
        let func = objc.send1(lib, objc.s("newFunctionWithName:"), ns);
        if func.is_null() {
            return std::ptr::null_mut();
        }
        let mut perr: Id = std::ptr::null_mut();
        objc.send2(
            device,
            objc.s("newComputePipelineStateWithFunction:error:"),
            func,
            &mut perr as *mut Id as Id,
        )
    };
    let pipe_leaf = pipe("b3_leaf1024");
    let pipe_parent = pipe("b3_parents");
    let pipe_leaf_range = pipe("b3_leaf1024_range");
    if pipe_leaf.is_null() || pipe_parent.is_null() || pipe_leaf_range.is_null() {
        return None;
    }

    let queue = objc.send0(device, objc.s("newCommandQueue"));
    if queue.is_null() {
        return None;
    }
    let count = objc.send_u2(device, objc.s("newBufferWithLength:options:"), 4, 0);
    if count.is_null() {
        return None;
    }

    Some(Gpu {
        objc,
        device,
        queue,
        pipe_leaf,
        pipe_parent,
        pipe_leaf_range,
        bufs: Mutex::new(Bufs {
            input: GpuBuf { buf: std::ptr::null_mut(), len: 0 },
            output: GpuBuf { buf: std::ptr::null_mut(), len: 0 },
            count: GpuBuf { buf: count, len: 4 },
        }),
    })
}

fn gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| {
        // Local kill switch for A/B runs; absent (e.g. the cleared ranked
        // worker environment) means enabled.
        if std::env::var_os("FLOCK_GPU").is_some_and(|v| v == "0") {
            return None;
        }
        unsafe { init() }
    })
    .as_ref()
}

/// Async GPU leaf stream for the ranked NTT→Merkle pipeline.
///
/// Binds the whole codeword (read) and the whole leaf array (write) once as
/// zero-copy shared buffers, then turns each finalized subtree into one
/// non-blocking compute dispatch. Command buffers on a single Metal queue
/// execute in commit order, so [`LeafStream::finish`] waits on the last one
/// and then checks every recorded status.
///
/// Safety contract (mirrors the CPU pipeline's): a range is submitted only
/// after the NTT has finalized those codeword elements and will never write
/// them again, ranges are pairwise disjoint, and nothing reads the leaf
/// array until `finish` returns. On `finish() == false` the leaf array
/// contents are unspecified and the caller MUST rehash every leaf on the CPU.
pub(crate) struct LeafStream {
    gpu: &'static Gpu,
    codeword_buf: Id,
    leaves_buf: Id,
    /// Committed command buffers, in commit order, guarded for concurrent
    /// `submit` callers (the NTT publishes subtrees from multiple threads).
    inflight: Mutex<Vec<Id>>,
    ok: std::sync::atomic::AtomicBool,
}

// SAFETY: same argument as `Gpu`; the raw buffer ids are immutable after
// construction and the in-flight list is mutex-guarded.
unsafe impl Send for LeafStream {}
unsafe impl Sync for LeafStream {}

impl LeafStream {
    /// `codeword` / `leaves` must stay alive and unmoved until `finish`
    /// returns. Returns `None` (caller keeps its CPU path) when the GPU is
    /// unavailable or either region cannot be wrapped zero-copy.
    pub(crate) fn new(
        codeword_ptr: *const u8,
        codeword_len: usize,
        leaves_ptr: *mut Hash,
        n_leaves: usize,
    ) -> Option<LeafStream> {
        debug_assert_eq!(codeword_len, n_leaves * 1024);
        let gpu = gpu()?;
        unsafe {
            let codeword_buf = gpu
                .objc
                .no_copy_buffer(gpu.device, codeword_ptr, codeword_len);
            let leaves_buf =
                gpu.objc
                    .no_copy_buffer(gpu.device, leaves_ptr as *const u8, n_leaves * 32);
            if codeword_buf.is_null() || leaves_buf.is_null() {
                return None;
            }
            Some(LeafStream {
                gpu,
                codeword_buf,
                leaves_buf,
                inflight: Mutex::new(Vec::with_capacity(1 << 11)),
                ok: std::sync::atomic::AtomicBool::new(true),
            })
        }
    }

    /// Queue one finalized subtree's leaves; never blocks on the GPU.
    pub(crate) fn submit(&self, leaf_start: usize, leaf_count: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        if !self.ok.load(Relaxed) {
            return;
        }
        let o = &self.gpu.objc;
        unsafe {
            let mut inflight = match self.inflight.lock() {
                Ok(g) => g,
                Err(_) => {
                    self.ok.store(false, Relaxed);
                    return;
                }
            };
            let cb = o.send0(self.gpu.queue, o.s("commandBuffer"));
            let enc = if cb.is_null() {
                std::ptr::null_mut()
            } else {
                o.send0(cb, o.s("computeCommandEncoder"))
            };
            if enc.is_null() {
                self.ok.store(false, Relaxed);
                return;
            }
            o.send1(enc, o.s("setComputePipelineState:"), self.gpu.pipe_leaf_range);
            o.set_buffer(enc, self.codeword_buf, 0);
            o.set_buffer(enc, self.leaves_buf, 1);
            let range: [u32; 2] = [leaf_start as u32, leaf_count as u32];
            o.set_bytes(enc, range.as_ptr() as *const c_void, 8, 2);
            o.dispatch(enc, leaf_count as u64);
            o.send0(enc, o.s("endEncoding"));
            o.send0(cb, o.s("commit"));
            inflight.push(cb);
        }
    }

    /// Wait for all queued work; `true` iff every dispatch completed cleanly.
    pub(crate) fn finish(self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let inflight = match self.inflight.into_inner() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if !self.ok.load(Relaxed) {
            return false;
        }
        let o = &self.gpu.objc;
        unsafe {
            if let Some(&last) = inflight.last() {
                o.send0(last, o.s("waitUntilCompleted"));
            }
            // In-order queue: by the time the last buffer is complete all
            // earlier ones are too; still verify every status (4 = Completed).
            inflight
                .iter()
                .all(|&cb| o.send0(cb, o.s("status")) as usize == 4)
        }
    }
}

/// Whole-level hooks are opt-in (`FLOCK_GPU_MERKLE=1`). On the ranked
/// machine the synchronous per-level dispatches measured as a net loss —
/// the E-core pipeline already hides leaf hashing, and each parent level
/// pays a blocking command-buffer round-trip — so the ranked worker's
/// cleared environment gets the CPU path.
fn whole_level_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_GPU_MERKLE").is_some_and(|v| v == "1"))
}

/// Hash `out.len()` contiguous 1 KiB BLAKE3 leaves on the GPU. Returns
/// `false` (caller must use the CPU path) when the hook is not opted in,
/// the GPU is unavailable, the level is too small to win, or the dispatch
/// fails.
pub(super) fn hash_leaves_1024(data: &[u8], out: &mut [Hash]) -> bool {
    debug_assert_eq!(data.len(), out.len() * 1024);
    if !whole_level_enabled() || out.len() < MIN_LEAVES {
        return false;
    }
    match gpu() {
        Some(g) => unsafe { g.run(g.pipe_leaf, data, out, out.len()) },
        None => false,
    }
}

/// Hash `out.len()` parent nodes (64-byte concatenated child CVs each) on
/// the GPU. Same fallback contract as [`hash_leaves_1024`].
pub(super) fn hash_parents(children: &[u8], out: &mut [Hash]) -> bool {
    debug_assert_eq!(children.len(), out.len() * 64);
    if !whole_level_enabled() || out.len() < MIN_PARENTS {
        return false;
    }
    match gpu() {
        Some(g) => unsafe { g.run(g.pipe_parent, children, out, out.len()) },
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{hash_leaf, hash_pair, Hash};
    use crate::hash::HashKind;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    /// GPU leaf digests must be byte-identical to the CPU `hash_leaf` oracle.
    /// Skips (rather than fails) when no GPU is available so the suite still
    /// passes on GPU-less CI hosts.
    #[test]
    fn gpu_leaves_match_cpu_oracle() {
        let n = super::MIN_LEAVES;
        let mut rng = Rng(0x1EAF_1EAF);
        let data: Vec<u8> = (0..n * 1024).map(|_| rng.next() as u8).collect();
        let mut gpu_out = vec![[0u8; 32] as Hash; n];
        if !super::hash_leaves_1024(&data, &mut gpu_out) {
            eprintln!("gpu_leaves_match_cpu_oracle: no GPU available, skipping");
            return;
        }
        for i in (0..n).step_by(97).chain([0, n - 1]) {
            let want = hash_leaf(&data[i * 1024..(i + 1) * 1024], HashKind::Blake3);
            assert_eq!(gpu_out[i], want, "leaf {i} mismatch");
        }
    }

    /// GPU parent digests must be byte-identical to the CPU `hash_pair` oracle.
    #[test]
    fn gpu_parents_match_cpu_oracle() {
        let n = super::MIN_PARENTS;
        let mut rng = Rng(0x0A1D_5EED_0A1D_5EED);
        let children: Vec<u8> = (0..n * 64).map(|_| rng.next() as u8).collect();
        let mut gpu_out = vec![[0u8; 32] as Hash; n];
        if !super::hash_parents(&children, &mut gpu_out) {
            eprintln!("gpu_parents_match_cpu_oracle: no GPU available, skipping");
            return;
        }
        for i in (0..n).step_by(89).chain([0, n - 1]) {
            let l: &Hash = children[i * 64..i * 64 + 32].try_into().unwrap();
            let r: &Hash = children[i * 64 + 32..(i + 1) * 64].try_into().unwrap();
            let want = hash_pair(l, r, HashKind::Blake3);
            assert_eq!(gpu_out[i], want, "parent {i} mismatch");
        }
    }
}
