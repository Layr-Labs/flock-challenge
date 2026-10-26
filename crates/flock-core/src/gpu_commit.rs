#![allow(clippy::missing_safety_doc)]

use crate::field::F128;
use crate::ntt::AdditiveNttF128;

pub const ENV_NO_GPU_COMMIT: &str = "FLOCK_NO_GPU_COMMIT";

pub const ENV_GPU_COMMIT_FORCE: &str = "FLOCK_GPU_COMMIT_FORCE";

#[inline]
pub fn gpu_commit_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os(ENV_NO_GPU_COMMIT).as_deref() != Some(std::ffi::OsStr::new("1"))
            && std::env::var_os(ENV_GPU_COMMIT_FORCE).is_some()
    })
}

#[inline]
pub fn try_gpu_interleaved_ntt_from_layer(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    num_ntts: usize,
    start_layer: usize,
) -> bool {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        imp::try_gpu_interleaved_ntt_from_layer(ntt, data, num_ntts, start_layer)
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (ntt, data, num_ntts, start_layer);
        false
    }
}

#[inline]
pub fn try_gpu_merkle_blake3(data: &[u8], num_leaves: usize) -> Option<Vec<crate::merkle::Hash>> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        imp::try_gpu_merkle_blake3(data, num_leaves)
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (data, num_leaves);
        None
    }
}

#[inline]
pub fn try_gpu_commit_ntt_and_blake3_merkle(
    ntt: &AdditiveNttF128,
    codeword: &mut [F128],
    num_ntts: usize,
    start_layer: usize,
    num_leaves: usize,
) -> Option<Vec<crate::merkle::Hash>> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        imp::try_gpu_commit_ntt_and_blake3_merkle(ntt, codeword, num_ntts, start_layer, num_leaves)
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (ntt, codeword, num_ntts, start_layer, num_leaves);
        None
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod imp {
    use super::*;
    use std::ffi::c_void;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU8, Ordering};

    type Id = *mut c_void;
    type Sel = *mut c_void;

    unsafe extern "C" {
        fn dlopen(path: *const i8, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
    }

    const RTLD_NOW: i32 = 2;
    const NIL: Id = std::ptr::null_mut();

    const LATCH_UNMEASURED: u8 = 0;
    const LATCH_ON: u8 = 1;
    const LATCH_OFF: u8 = 2;

    static GPU_LATCH: AtomicU8 = AtomicU8::new(LATCH_UNMEASURED);

    #[derive(Clone, Copy)]
    struct Api {
        msg_send: *const c_void,
        get_class: unsafe extern "C" fn(*const i8) -> Id,
        sel_register: unsafe extern "C" fn(*const i8) -> Sel,
        pool_push: unsafe extern "C" fn() -> *mut c_void,
        pool_pop: unsafe extern "C" fn(*mut c_void),
        create_system_default_device: unsafe extern "C" fn() -> Id,
        copy_all_devices: unsafe extern "C" fn() -> Id,
    }

    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    macro_rules! send {
        ($api:expr, $ty:ty, $obj:expr, $sel:expr $(, $a:expr)* $(,)?) => {{
            let f: $ty = core::mem::transmute($api.msg_send);
            f($obj, ($api.sel_register)($sel.as_ptr()) $(, $a)*)
        }};
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MtlSize {
        width: u64,
        height: u64,
        depth: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct LayerParams {
        log_d: u32,
        layer: u32,
        log_num_ntts: u32,
        total_threads: u32,
    }

    impl Api {
        fn load() -> Result<Self, String> {
            unsafe {
                let objc = dlopen(c"/usr/lib/libobjc.A.dylib".as_ptr().cast(), RTLD_NOW);
                if objc.is_null() {
                    return Err("dlopen libobjc.A.dylib failed".into());
                }
                let foundation = dlopen(
                    c"/System/Library/Frameworks/Foundation.framework/Foundation"
                        .as_ptr()
                        .cast(),
                    RTLD_NOW,
                );
                if foundation.is_null() {
                    return Err("dlopen Foundation.framework failed".into());
                }
                let metal = dlopen(
                    c"/System/Library/Frameworks/Metal.framework/Metal"
                        .as_ptr()
                        .cast(),
                    RTLD_NOW,
                );
                if metal.is_null() {
                    return Err("dlopen Metal.framework failed".into());
                }
                let sym = |h: *mut c_void, name: &core::ffi::CStr| -> Result<*mut c_void, String> {
                    let p = dlsym(h, name.as_ptr());
                    if p.is_null() {
                        Err(format!("dlsym {name:?} failed"))
                    } else {
                        Ok(p)
                    }
                };
                Ok(Self {
                    msg_send: sym(objc, c"objc_msgSend")?,
                    get_class: core::mem::transmute(sym(objc, c"objc_getClass")?),
                    sel_register: core::mem::transmute(sym(objc, c"sel_registerName")?),
                    pool_push: core::mem::transmute(sym(objc, c"objc_autoreleas\x65PoolPush")?),
                    pool_pop: core::mem::transmute(sym(objc, c"objc_autoreleas\x65PoolPop")?),
                    create_system_default_device: core::mem::transmute(sym(
                        metal,
                        c"MTLCreateSystemDefaultDevice",
                    )?),
                    copy_all_devices: core::mem::transmute(sym(metal, c"MTLCopyAllDevices")?),
                })
            }
        }

        unsafe fn nsstring(&self, s: &str) -> Result<Id, String> {
            unsafe {
                let cls = (self.get_class)(c"NSString".as_ptr().cast());
                if cls.is_null() {
                    return Err("NSString class not found".into());
                }
                let mut buf = Vec::with_capacity(s.len() + 1);
                buf.extend_from_slice(s.as_bytes());
                buf.push(0);
                let ns: Id = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel, *const u8) -> Id,
                    cls,
                    c"stringWithUTF8String:",
                    buf.as_ptr()
                );
                if ns.is_null() {
                    Err("NSString creation failed".into())
                } else {
                    Ok(ns)
                }
            }
        }

        unsafe fn error_string(&self, err: Id) -> String {
            if err.is_null() {
                return "unknown Metal error".into();
            }
            unsafe {
                let desc: Id = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    err,
                    c"localizedDescription"
                );
                if desc.is_null() {
                    return "unknown Metal error (nil localizedDescription)".into();
                }
                let cstr: *const i8 = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel) -> *const i8,
                    desc,
                    c"UTF8String"
                );
                if cstr.is_null() {
                    return "unknown Metal error (nil UTF8String)".into();
                }
                std::ffi::CStr::from_ptr(cstr)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    struct Gf2KernelMetalGpu {
        api: Api,
        device: Id,
        queue: Id,
        pso_interleaved_layer: Id,
        pso_butterfly_1d: Id,
        pso_blake3_leaves: Id,
        pso_blake3_parents: Id,
    }

    unsafe impl Send for Gf2KernelMetalGpu {}
    unsafe impl Sync for Gf2KernelMetalGpu {}

    static GPU_CTX: OnceLock<Result<Gf2KernelMetalGpu, String>> = OnceLock::new();

    fn init_gpu() -> Result<Gf2KernelMetalGpu, String> {
        let api = Api::load()?;
        unsafe {
            let pool = (api.pool_push)();
            let res = (|| -> Result<Gf2KernelMetalGpu, String> {
                let mut device = (api.create_system_default_device)();
                if device.is_null() {
                    let all = (api.copy_all_devices)();
                    if !all.is_null() {
                        device = send!(
                            api,
                            unsafe extern "C" fn(Id, Sel) -> Id,
                            all,
                            c"firstObject"
                        );
                    }
                }
                if device.is_null() {
                    return Err("MTLCreateSystemDefaultDevice returned nil".into());
                }
                let queue: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    device,
                    c"newCommandQueue"
                );
                if queue.is_null() {
                    return Err("newCommandQueue returned nil".into());
                }
                let src = api.nsstring(crate::gf2_shaders::SOURCE)?;
                let mut err: Id = NIL;
                let library: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id,
                    device,
                    c"newLibraryWithSource:options:error:",
                    src,
                    NIL,
                    &mut err
                );
                if library.is_null() {
                    return Err(format!(
                        "GF2 MSL shader compile failed: {}",
                        api.error_string(err)
                    ));
                }
                let build_pso = |fn_name: &str| -> Result<Id, String> {
                    let ns = api.nsstring(fn_name)?;
                    let func: Id = send!(
                        api,
                        unsafe extern "C" fn(Id, Sel, Id) -> Id,
                        library,
                        c"newFunctionWithName:",
                        ns
                    );
                    if func.is_null() {
                        return Err(format!("Metal kernel {fn_name} not found in library"));
                    }
                    let mut pso_err: Id = NIL;
                    let pso: Id = send!(
                        api,
                        unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id,
                        device,
                        c"newComputePipelineStateWithFunction:error:",
                        func,
                        &mut pso_err
                    );
                    let _: Id = send!(api, unsafe extern "C" fn(Id, Sel) -> Id, func, c"release");
                    if pso.is_null() {
                        Err(format!(
                            "newComputePipelineStateWithFunction({fn_name}) failed: {}",
                            api.error_string(pso_err)
                        ))
                    } else {
                        Ok(pso)
                    }
                };
                let pso_interleaved_layer = build_pso("k0_kernel_ntt_interleaved_layer_f128")?;
                let pso_butterfly_1d = build_pso("k0_kernel_ntt_butterfly_f128")?;
                let pso_blake3_leaves = build_pso("k0_kernel_blake3_merkle_leaves")?;
                let pso_blake3_parents = build_pso("k0_kernel_blake3_merkle_parents")?;
                let _: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    library,
                    c"release"
                );
                Ok(Gf2KernelMetalGpu {
                    api,
                    device,
                    queue,
                    pso_interleaved_layer,
                    pso_butterfly_1d,
                    pso_blake3_leaves,
                    pso_blake3_parents,
                })
            })();
            (api.pool_pop)(pool);
            res
        }
    }

    fn get_gpu() -> Result<&'static Gf2KernelMetalGpu, String> {
        GPU_CTX.get_or_init(init_gpu).as_ref().map_err(Clone::clone)
    }

    fn run_gpu_interleaved_ntt(
        gpu: &Gf2KernelMetalGpu,
        ntt: &AdditiveNttF128,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) -> Result<(), String> {
        let n_total = data.len();
        if n_total == 0 || !num_ntts.is_power_of_two() || n_total % num_ntts != 0 {
            return Err("invalid interleaved NTT geometry".into());
        }
        let n_positions = n_total / num_ntts;
        if !n_positions.is_power_of_two() {
            return Err("n_positions must be a power of two".into());
        }
        let log_d = n_positions.trailing_zeros() as usize;
        let log_num_ntts = num_ntts.trailing_zeros() as usize;
        if start_layer >= log_d {
            return Ok(());
        }
        let total_pairs_times_lanes = (n_positions >> 1) * num_ntts;
        if total_pairs_times_lanes > u32::MAX as usize {
            return Err("grid exceeds u32::MAX threads".into());
        }

        let base_tw = (1usize << start_layer) - 1;
        let total_tw = ((1usize << log_d) - 1) - base_tw;
        let mut flat_twiddles = Vec::with_capacity(total_tw);
        for l in start_layer..log_d {
            let num_blocks = 1usize << l;
            for b in 0..num_blocks {
                flat_twiddles.push(ntt.twiddle(l, b));
            }
        }

        unsafe {
            let pool = (gpu.api.pool_push)();
            let res = (|| -> Result<(), String> {
                let data_bytes = core::mem::size_of_val(data);
                let data_buf: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id,
                    gpu.device,
                    c"newBufferWithBytes:length:options:",
                    data.as_ptr().cast(),
                    data_bytes as u64,
                    0u64
                );
                if data_buf.is_null() {
                    return Err("newBufferWithBytes(data) failed".into());
                }

                let tw_bytes = core::mem::size_of_val(flat_twiddles.as_slice());
                let tw_buf: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id,
                    gpu.device,
                    c"newBufferWithBytes:length:options:",
                    flat_twiddles.as_ptr().cast(),
                    tw_bytes as u64,
                    0u64
                );
                if tw_buf.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return Err("newBufferWithBytes(twiddles) failed".into());
                }

                let cb: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    gpu.queue,
                    c"commandBuffer"
                );
                if cb.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        tw_buf,
                        c"release"
                    );
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return Err("commandBuffer failed".into());
                }

                let enc: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    cb,
                    c"computeCommandEncoder"
                );
                if enc.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        tw_buf,
                        c"release"
                    );
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return Err("computeCommandEncoder failed".into());
                }

                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id),
                    enc,
                    c"setComputePipelineState:",
                    gpu.pso_interleaved_layer
                );
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                    enc,
                    c"setBuffer:offset:atIndex:",
                    data_buf,
                    0u64,
                    0u64
                );

                let threads_per_group = 256u64;
                let groups =
                    ((total_pairs_times_lanes as u64) + threads_per_group - 1) / threads_per_group;

                for l in start_layer..log_d {
                    let tw_offset_elems = ((1usize << l) - 1) - base_tw;
                    let tw_offset_bytes = tw_offset_elems * core::mem::size_of::<F128>();
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                        enc,
                        c"setBuffer:offset:atIndex:",
                        tw_buf,
                        tw_offset_bytes as u64,
                        1u64
                    );
                    let params = LayerParams {
                        log_d: log_d as u32,
                        layer: l as u32,
                        log_num_ntts: log_num_ntts as u32,
                        total_threads: total_pairs_times_lanes as u32,
                    };
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64),
                        enc,
                        c"setBytes:length:atIndex:",
                        (&params as *const LayerParams).cast(),
                        core::mem::size_of::<LayerParams>() as u64,
                        2u64
                    );
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize),
                        enc,
                        c"dispatchThreadgroups:threadsPerThreadgroup:",
                        MtlSize {
                            width: groups,
                            height: 1,
                            depth: 1,
                        },
                        MtlSize {
                            width: threads_per_group,
                            height: 1,
                            depth: 1,
                        }
                    );
                }

                send!(gpu.api, unsafe extern "C" fn(Id, Sel), enc, c"endEncoding");
                send!(gpu.api, unsafe extern "C" fn(Id, Sel), cb, c"commit");
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel),
                    cb,
                    c"waitUntilCompleted"
                );
                let status: u64 =
                    send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> u64, cb, c"status");
                if status != 4 {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        tw_buf,
                        c"release"
                    );
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return Err(format!("Metal commandBuffer failed with status={status}"));
                }

                let contents: *const F128 = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> *const F128,
                    data_buf,
                    c"contents"
                );
                if contents.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        tw_buf,
                        c"release"
                    );
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return Err("MTLBuffer contents returned null".into());
                }
                core::ptr::copy_nonoverlapping(contents, data.as_mut_ptr(), n_total);
                let _: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    tw_buf,
                    c"release"
                );
                let _: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    data_buf,
                    c"release"
                );
                Ok(())
            })();
            (gpu.api.pool_pop)(pool);
            res
        }
    }

    pub(super) fn try_gpu_interleaved_ntt_from_layer(
        ntt: &AdditiveNttF128,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) -> bool {
        if !super::gpu_commit_enabled() {
            return false;
        }
        let latch = GPU_LATCH.load(Ordering::Relaxed);
        if latch == LATCH_OFF {
            return false;
        }
        let gpu = match get_gpu() {
            Ok(g) => g,
            Err(_) => {
                GPU_LATCH.store(LATCH_OFF, Ordering::Relaxed);
                return false;
            }
        };
        let _ = gpu.pso_butterfly_1d;

        if latch == LATCH_ON {
            if run_gpu_interleaved_ntt(gpu, ntt, data, num_ntts, start_layer).is_ok() {
                return true;
            } else {
                GPU_LATCH.store(LATCH_OFF, Ordering::Relaxed);
                return false;
            }
        }

        let mut gpu_candidate = data.to_vec();
        let t_gpu = std::time::Instant::now();
        let gpu_ok =
            run_gpu_interleaved_ntt(gpu, ntt, &mut gpu_candidate, num_ntts, start_layer).is_ok();
        let gpu_secs = t_gpu.elapsed().as_secs_f64();

        let t_cpu = std::time::Instant::now();
        ntt.forward_transform_interleaved_from_layer(data, num_ntts, start_layer);
        let cpu_secs = t_cpu.elapsed().as_secs_f64();

        if !gpu_ok || gpu_candidate.as_slice() != data {
            GPU_LATCH.store(LATCH_OFF, Ordering::Relaxed);
            return true;
        }

        let force = std::env::var_os(super::ENV_GPU_COMMIT_FORCE).is_some();
        if force || gpu_secs < cpu_secs {
            GPU_LATCH.store(LATCH_ON, Ordering::Relaxed);
        } else {
            GPU_LATCH.store(LATCH_OFF, Ordering::Relaxed);
        }
        true
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct LeafParams {
        num_leaves: u32,
        blocks_per_leaf: u32,
    }

    pub(super) fn try_gpu_merkle_blake3(
        data: &[u8],
        num_leaves: usize,
    ) -> Option<Vec<crate::merkle::Hash>> {
        if !super::gpu_commit_enabled() || GPU_LATCH.load(Ordering::Relaxed) != LATCH_ON {
            return None;
        }
        if !num_leaves.is_power_of_two() || num_leaves == 0 || data.len() % num_leaves != 0 {
            return None;
        }
        let leaf_size = data.len() / num_leaves;
        if leaf_size == 0 || leaf_size % 64 != 0 || leaf_size > 1024 {
            return None;
        }
        let blocks_per_leaf = leaf_size / 64;
        let gpu = get_gpu().ok()?;
        let total_nodes = 2 * num_leaves - 1;
        let tree_bytes = total_nodes * core::mem::size_of::<crate::merkle::Hash>();

        unsafe {
            let pool = (gpu.api.pool_push)();
            let res = (|| -> Option<Vec<crate::merkle::Hash>> {
                let data_buf: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id,
                    gpu.device,
                    c"newBufferWithBytes:length:options:",
                    data.as_ptr().cast(),
                    data.len() as u64,
                    0u64
                );
                if data_buf.is_null() {
                    return None;
                }
                let tree_buf: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, u64, u64) -> Id,
                    gpu.device,
                    c"newBufferWithLength:options:",
                    tree_bytes as u64,
                    0u64
                );
                if tree_buf.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return None;
                }
                let cb: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    gpu.queue,
                    c"commandBuffer"
                );
                let enc: Id = if cb.is_null() {
                    NIL
                } else {
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        cb,
                        c"computeCommandEncoder"
                    )
                };
                if enc.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        tree_buf,
                        c"release"
                    );
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return None;
                }

                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id),
                    enc,
                    c"setComputePipelineState:",
                    gpu.pso_blake3_leaves
                );
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                    enc,
                    c"setBuffer:offset:atIndex:",
                    data_buf,
                    0u64,
                    0u64
                );
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                    enc,
                    c"setBuffer:offset:atIndex:",
                    tree_buf,
                    0u64,
                    1u64
                );
                let lparams = LeafParams {
                    num_leaves: num_leaves as u32,
                    blocks_per_leaf: blocks_per_leaf as u32,
                };
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64),
                    enc,
                    c"setBytes:length:atIndex:",
                    (&lparams as *const LeafParams).cast(),
                    core::mem::size_of::<LeafParams>() as u64,
                    2u64
                );
                let tpg = 256u64;
                let leaf_groups = ((num_leaves as u64) + tpg - 1) / tpg;
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize),
                    enc,
                    c"dispatchThreadgroups:threadsPerThreadgroup:",
                    MtlSize {
                        width: leaf_groups,
                        height: 1,
                        depth: 1
                    },
                    MtlSize {
                        width: tpg,
                        height: 1,
                        depth: 1
                    }
                );

                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id),
                    enc,
                    c"setComputePipelineState:",
                    gpu.pso_blake3_parents
                );
                let mut read_start = 0usize;
                let mut read_len = num_leaves;
                while read_len > 1 {
                    let next_len = read_len >> 1;
                    let write_start = read_start + read_len;
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                        enc,
                        c"setBuffer:offset:atIndex:",
                        tree_buf,
                        (read_start * 32) as u64,
                        0u64
                    );
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                        enc,
                        c"setBuffer:offset:atIndex:",
                        tree_buf,
                        (write_start * 32) as u64,
                        1u64
                    );
                    let np = next_len as u32;
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64),
                        enc,
                        c"setBytes:length:atIndex:",
                        (&np as *const u32).cast(),
                        4u64,
                        2u64
                    );
                    let pgroups = ((next_len as u64) + tpg - 1) / tpg;
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize),
                        enc,
                        c"dispatchThreadgroups:threadsPerThreadgroup:",
                        MtlSize {
                            width: pgroups,
                            height: 1,
                            depth: 1
                        },
                        MtlSize {
                            width: tpg,
                            height: 1,
                            depth: 1
                        }
                    );
                    read_start = write_start;
                    read_len = next_len;
                }

                send!(gpu.api, unsafe extern "C" fn(Id, Sel), enc, c"endEncoding");
                send!(gpu.api, unsafe extern "C" fn(Id, Sel), cb, c"commit");
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel),
                    cb,
                    c"waitUntilCompleted"
                );
                let status: u64 =
                    send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> u64, cb, c"status");
                let mut out_tree = None;
                if status == 4 {
                    let ptr: *const crate::merkle::Hash = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> *const crate::merkle::Hash,
                        tree_buf,
                        c"contents"
                    );
                    if !ptr.is_null() {
                        let mut v: Vec<crate::merkle::Hash> = crate::alloc_uninit_vec(total_nodes);
                        core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), total_nodes);
                        out_tree = Some(v);
                    }
                }
                let _: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    tree_buf,
                    c"release"
                );
                let _: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    data_buf,
                    c"release"
                );
                out_tree
            })();
            (gpu.api.pool_pop)(pool);
            res
        }
    }

    fn run_gpu_commit_ntt_and_blake3_merkle(
        gpu: &Gf2KernelMetalGpu,
        ntt: &AdditiveNttF128,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        num_leaves: usize,
    ) -> Result<Vec<crate::merkle::Hash>, String> {
        let n_total = data.len();
        if n_total == 0 || !num_ntts.is_power_of_two() || n_total % num_ntts != 0 {
            return Err("invalid interleaved NTT geometry".into());
        }
        let n_positions = n_total / num_ntts;
        if !n_positions.is_power_of_two() || !num_leaves.is_power_of_two() || num_leaves == 0 {
            return Err("n_positions and num_leaves must be powers of two".into());
        }
        let data_bytes = core::mem::size_of_val(data);
        if data_bytes % num_leaves != 0 {
            return Err("codeword byte length not divisible by num_leaves".into());
        }
        let leaf_size = data_bytes / num_leaves;
        if leaf_size == 0 || leaf_size % 64 != 0 || leaf_size > 1024 {
            return Err("BLAKE3 leaf size outside 64..=1024 single-chunk range".into());
        }
        let blocks_per_leaf = leaf_size / 64;
        let log_d = n_positions.trailing_zeros() as usize;
        let log_num_ntts = num_ntts.trailing_zeros() as usize;
        if start_layer >= log_d {
            return Err("start_layer >= log_d".into());
        }
        let total_pairs_times_lanes = (n_positions >> 1) * num_ntts;
        if total_pairs_times_lanes > u32::MAX as usize {
            return Err("grid exceeds u32::MAX threads".into());
        }

        let base_tw = (1usize << start_layer) - 1;
        let total_tw = ((1usize << log_d) - 1) - base_tw;
        let mut flat_twiddles = Vec::with_capacity(total_tw);
        for l in start_layer..log_d {
            let num_blocks = 1usize << l;
            for b in 0..num_blocks {
                flat_twiddles.push(ntt.twiddle(l, b));
            }
        }
        let total_nodes = 2 * num_leaves - 1;
        let tree_bytes = total_nodes * core::mem::size_of::<crate::merkle::Hash>();

        unsafe {
            let pool = (gpu.api.pool_push)();
            let res = (|| -> Result<Vec<crate::merkle::Hash>, String> {
                let data_buf: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id,
                    gpu.device,
                    c"newBufferWithBytes:length:options:",
                    data.as_ptr().cast(),
                    data_bytes as u64,
                    0u64
                );
                if data_buf.is_null() {
                    return Err("newBufferWithBytes(data) failed".into());
                }
                let tw_bytes = core::mem::size_of_val(flat_twiddles.as_slice());
                let tw_buf: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id,
                    gpu.device,
                    c"newBufferWithBytes:length:options:",
                    flat_twiddles.as_ptr().cast(),
                    tw_bytes as u64,
                    0u64
                );
                if tw_buf.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return Err("newBufferWithBytes(twiddles) failed".into());
                }
                let tree_buf: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, u64, u64) -> Id,
                    gpu.device,
                    c"newBufferWithLength:options:",
                    tree_bytes as u64,
                    0u64
                );
                if tree_buf.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        tw_buf,
                        c"release"
                    );
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return Err("newBufferWithLength(tree) failed".into());
                }
                let cb: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    gpu.queue,
                    c"commandBuffer"
                );
                let enc: Id = if cb.is_null() {
                    NIL
                } else {
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        cb,
                        c"computeCommandEncoder"
                    )
                };
                if enc.is_null() {
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        tree_buf,
                        c"release"
                    );
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        tw_buf,
                        c"release"
                    );
                    let _: Id = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        data_buf,
                        c"release"
                    );
                    return Err("computeCommandEncoder failed".into());
                }

                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id),
                    enc,
                    c"setComputePipelineState:",
                    gpu.pso_interleaved_layer
                );
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                    enc,
                    c"setBuffer:offset:atIndex:",
                    data_buf,
                    0u64,
                    0u64
                );
                let tpg = 256u64;
                let ntt_groups = ((total_pairs_times_lanes as u64) + tpg - 1) / tpg;
                for l in start_layer..log_d {
                    let tw_offset_elems = ((1usize << l) - 1) - base_tw;
                    let tw_offset_bytes = tw_offset_elems * core::mem::size_of::<F128>();
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                        enc,
                        c"setBuffer:offset:atIndex:",
                        tw_buf,
                        tw_offset_bytes as u64,
                        1u64
                    );
                    let params = LayerParams {
                        log_d: log_d as u32,
                        layer: l as u32,
                        log_num_ntts: log_num_ntts as u32,
                        total_threads: total_pairs_times_lanes as u32,
                    };
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64),
                        enc,
                        c"setBytes:length:atIndex:",
                        (&params as *const LayerParams).cast(),
                        core::mem::size_of::<LayerParams>() as u64,
                        2u64
                    );
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize),
                        enc,
                        c"dispatchThreadgroups:threadsPerThreadgroup:",
                        MtlSize {
                            width: ntt_groups,
                            height: 1,
                            depth: 1
                        },
                        MtlSize {
                            width: tpg,
                            height: 1,
                            depth: 1
                        }
                    );
                }

                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id),
                    enc,
                    c"setComputePipelineState:",
                    gpu.pso_blake3_leaves
                );
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                    enc,
                    c"setBuffer:offset:atIndex:",
                    data_buf,
                    0u64,
                    0u64
                );
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                    enc,
                    c"setBuffer:offset:atIndex:",
                    tree_buf,
                    0u64,
                    1u64
                );
                let lparams = LeafParams {
                    num_leaves: num_leaves as u32,
                    blocks_per_leaf: blocks_per_leaf as u32,
                };
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64),
                    enc,
                    c"setBytes:length:atIndex:",
                    (&lparams as *const LeafParams).cast(),
                    core::mem::size_of::<LeafParams>() as u64,
                    2u64
                );
                let leaf_groups = ((num_leaves as u64) + tpg - 1) / tpg;
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize),
                    enc,
                    c"dispatchThreadgroups:threadsPerThreadgroup:",
                    MtlSize {
                        width: leaf_groups,
                        height: 1,
                        depth: 1
                    },
                    MtlSize {
                        width: tpg,
                        height: 1,
                        depth: 1
                    }
                );

                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel, Id),
                    enc,
                    c"setComputePipelineState:",
                    gpu.pso_blake3_parents
                );
                let mut read_start = 0usize;
                let mut read_len = num_leaves;
                while read_len > 1 {
                    let next_len = read_len >> 1;
                    let write_start = read_start + read_len;
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                        enc,
                        c"setBuffer:offset:atIndex:",
                        tree_buf,
                        (read_start * 32) as u64,
                        0u64
                    );
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                        enc,
                        c"setBuffer:offset:atIndex:",
                        tree_buf,
                        (write_start * 32) as u64,
                        1u64
                    );
                    let np = next_len as u32;
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64),
                        enc,
                        c"setBytes:length:atIndex:",
                        (&np as *const u32).cast(),
                        4u64,
                        2u64
                    );
                    let pgroups = ((next_len as u64) + tpg - 1) / tpg;
                    send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize),
                        enc,
                        c"dispatchThreadgroups:threadsPerThreadgroup:",
                        MtlSize {
                            width: pgroups,
                            height: 1,
                            depth: 1
                        },
                        MtlSize {
                            width: tpg,
                            height: 1,
                            depth: 1
                        }
                    );
                    read_start = write_start;
                    read_len = next_len;
                }

                send!(gpu.api, unsafe extern "C" fn(Id, Sel), enc, c"endEncoding");
                send!(gpu.api, unsafe extern "C" fn(Id, Sel), cb, c"commit");
                send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel),
                    cb,
                    c"waitUntilCompleted"
                );
                let status: u64 =
                    send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> u64, cb, c"status");
                let mut res_out = Err(format!("Metal commandBuffer failed with status={status}"));
                if status == 4 {
                    let d_ptr: *const F128 = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> *const F128,
                        data_buf,
                        c"contents"
                    );
                    let t_ptr: *const crate::merkle::Hash = send!(
                        gpu.api,
                        unsafe extern "C" fn(Id, Sel) -> *const crate::merkle::Hash,
                        tree_buf,
                        c"contents"
                    );
                    if !d_ptr.is_null() && !t_ptr.is_null() {
                        core::ptr::copy_nonoverlapping(d_ptr, data.as_mut_ptr(), n_total);
                        let mut v: Vec<crate::merkle::Hash> = crate::alloc_uninit_vec(total_nodes);
                        core::ptr::copy_nonoverlapping(t_ptr, v.as_mut_ptr(), total_nodes);
                        res_out = Ok(v);
                    }
                }
                let _: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    tree_buf,
                    c"release"
                );
                let _: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    tw_buf,
                    c"release"
                );
                let _: Id = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    data_buf,
                    c"release"
                );
                res_out
            })();
            (gpu.api.pool_pop)(pool);
            res
        }
    }

    pub(super) fn try_gpu_commit_ntt_and_blake3_merkle(
        ntt: &AdditiveNttF128,
        codeword: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        num_leaves: usize,
    ) -> Option<Vec<crate::merkle::Hash>> {
        if !super::gpu_commit_enabled() {
            return None;
        }
        let latch = GPU_LATCH.load(Ordering::Relaxed);
        if latch == LATCH_OFF {
            return None;
        }
        let gpu = match get_gpu() {
            Ok(g) => g,
            Err(_) => {
                GPU_LATCH.store(LATCH_OFF, Ordering::Relaxed);
                return None;
            }
        };

        if latch == LATCH_ON {
            match run_gpu_commit_ntt_and_blake3_merkle(
                gpu,
                ntt,
                codeword,
                num_ntts,
                start_layer,
                num_leaves,
            ) {
                Ok(tree) => return Some(tree),
                Err(_) => {
                    GPU_LATCH.store(LATCH_OFF, Ordering::Relaxed);
                    return None;
                }
            }
        }

        let mut gpu_codeword = codeword.to_vec();
        let t_gpu = std::time::Instant::now();
        let gpu_res = run_gpu_commit_ntt_and_blake3_merkle(
            gpu,
            ntt,
            &mut gpu_codeword,
            num_ntts,
            start_layer,
            num_leaves,
        );
        let gpu_secs = t_gpu.elapsed().as_secs_f64();

        let t_cpu = std::time::Instant::now();
        ntt.forward_transform_interleaved_from_layer(codeword, num_ntts, start_layer);
        let codeword_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                codeword.as_ptr() as *const u8,
                codeword.len() * core::mem::size_of::<F128>(),
            )
        };
        GPU_LATCH.store(LATCH_OFF, Ordering::Relaxed);
        let cpu_tree =
            crate::merkle::merkle_tree(codeword_bytes, num_leaves, crate::merkle::HashKind::Blake3);
        let cpu_secs = t_cpu.elapsed().as_secs_f64();

        let Ok(gpu_tree) = gpu_res else {
            return Some(cpu_tree);
        };
        if gpu_codeword.as_slice() != codeword || gpu_tree != cpu_tree {
            return Some(cpu_tree);
        }

        let force = std::env::var_os(super::ENV_GPU_COMMIT_FORCE).is_some();
        if force || gpu_secs < cpu_secs {
            GPU_LATCH.store(LATCH_ON, Ordering::Relaxed);
        }
        Some(cpu_tree)
    }
}
