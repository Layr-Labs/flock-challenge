//! Fixed exact-shape CPU/Metal split for the production L0 Merkle leaves.
//!
//! The CPU owns leaves `[0, 9/16)` and Metal owns leaves `[9/16, 1)`.
//! Metal sees only page-aligned suffix slices, wrapped through
//! `newBufferWithBytesNoCopy`; there is no copied-buffer fallback.

use super::Hash;
use core::ffi::{c_char, c_uchar, c_void};
use std::ffi::{CStr, c_int};
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, OnceLock};

type ObjcId = *mut c_void;
type ObjcSel = *const c_void;
type DispatchData = *mut c_void;
type MachPort = u32;
type MachVmAddress = u64;
type MachVmSize = u64;

pub(super) const TOTAL_LEAVES: usize = 1 << 20;
pub(super) const CPU_LEAVES: usize = 9 * (1 << 16);
pub(super) const GPU_LEAVES: usize = 7 * (1 << 16);
pub(super) const LEAF_BYTES: usize = 1 << 10;

const OUTPUT_BYTES_PER_LEAF: usize = core::mem::size_of::<Hash>();
const GPU_INPUT_BYTES: usize = GPU_LEAVES * LEAF_BYTES;
const GPU_OUTPUT_BYTES: usize = GPU_LEAVES * OUTPUT_BYTES_PER_LEAF;
const MAX_INPUT_SEGMENTS: usize = 16;
const MAX_OUTPUT_SEGMENTS: usize = 16;
// Intersecting N input and M output VM partitions can produce N+M-1 joint
// intervals. Accept the full mathematically possible partition instead of
// assuming the local machine's observed 4+1 segmentation.
const MAX_JOINT_INTERVALS: usize = MAX_INPUT_SEGMENTS + MAX_OUTPUT_SEGMENTS - 1;
const THREADS_PER_THREADGROUP: usize = 32;
const MTL_STORAGE_MODE_SHARED: usize = 0;
const COMMAND_STATUS_COMPLETED: usize = 4;
const VM_REGION_BASIC_INFO_64: c_int = 9;
const VM_PROT_READ: c_int = 0x01;
const VM_PROT_WRITE: c_int = 0x02;
const METALLIB: &[u8] = include_bytes!("sha256_leaf.metallib");
#[cfg(test)]
const METAL_SOURCE: &str = include_str!("sha256_leaf.metal");
const FUNCTION_NAME: &CStr = c"sha256_leaf_checksum";

#[repr(C)]
#[derive(Clone, Copy)]
struct MtlSize {
    width: usize,
    height: usize,
    depth: usize,
}

// Darwin gives this public structure four-byte aggregate alignment.
#[repr(C, packed(4))]
#[derive(Clone, Copy, Default)]
struct VmRegionBasicInfo64 {
    protection: c_int,
    max_protection: c_int,
    inheritance: c_int,
    shared: c_int,
    reserved: c_int,
    offset: u64,
    behavior: c_int,
    user_wired_count: u16,
}

#[derive(Clone, Debug)]
struct RawVmRegion {
    region_start: u64,
    region_end: u64,
    protection: c_int,
    max_protection: c_int,
    shared: bool,
    reserved: bool,
    object_name_was_nonnull: bool,
    object_name_deallocated: bool,
}

#[derive(Clone, Debug)]
struct VmSegment {
    segment_index: usize,
    allocation_offset_bytes: usize,
    pointer: usize,
    length_bytes: usize,
    leaf_start: usize,
    leaf_end: usize,
    raw_region_start: u64,
    raw_region_end: u64,
    protection: c_int,
    max_protection: c_int,
    shared: bool,
    reserved: bool,
    object_name_was_nonnull: bool,
    object_name_deallocated: bool,
}

#[derive(Clone, Debug)]
struct JointLeafInterval {
    interval_index: usize,
    global_leaf_start: usize,
    leaf_count: usize,
    input_segment_index: usize,
    output_segment_index: usize,
    input_byte_offset: usize,
    output_byte_offset: usize,
    input_uint4_offset: u32,
    output_uint4_offset: u32,
}

#[derive(Debug)]
pub struct HybridStats {
    device_name: &'static str,
    max_buffer_length: usize,
    thread_execution_width: usize,
    max_threads_per_threadgroup: usize,
    input_segments: Vec<VmSegment>,
    output_segments: Vec<VmSegment>,
    intervals: Vec<JointLeafInterval>,
    input_buffer_attempts: usize,
    output_buffer_attempts: usize,
    no_copy_identity_checks: usize,
    commit_host_seconds: f64,
    cpu_finish_host_seconds: f64,
    wait_nanoseconds: u128,
    gpu_start_seconds: f64,
    gpu_end_seconds: f64,
    command_status: usize,
}

#[cfg(any(test, feature = "hash-count"))]
#[allow(dead_code)]
impl HybridStats {
    pub fn input_segment_count(&self) -> usize {
        self.input_segments.len()
    }

    pub fn output_segment_count(&self) -> usize {
        self.output_segments.len()
    }

    pub fn input_buffer_attempts(&self) -> usize {
        self.input_buffer_attempts
    }

    pub fn output_buffer_attempts(&self) -> usize {
        self.output_buffer_attempts
    }

    pub fn no_copy_identity_checks(&self) -> usize {
        self.no_copy_identity_checks
    }

    pub fn dispatch_global_ranges(&self) -> Vec<(usize, usize)> {
        self.intervals
            .iter()
            .map(|interval| {
                (
                    interval.global_leaf_start,
                    interval.global_leaf_start + interval.leaf_count,
                )
            })
            .collect()
    }

    pub fn commit_host_seconds(&self) -> f64 {
        self.commit_host_seconds
    }

    pub fn cpu_barrier_host_seconds(&self) -> f64 {
        self.cpu_finish_host_seconds
    }

    pub fn terminal_wait_nanoseconds(&self) -> u128 {
        self.wait_nanoseconds
    }

    pub fn gpu_start_seconds(&self) -> f64 {
        self.gpu_start_seconds
    }

    pub fn gpu_end_seconds(&self) -> f64 {
        self.gpu_end_seconds
    }

    pub fn command_status(&self) -> usize {
        self.command_status
    }
}

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> ObjcId;
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> ObjcId;
    fn sel_registerName(name: *const c_char) -> ObjcSel;
    fn objc_msgSend();
}

#[link(name = "System")]
unsafe extern "C" {
    fn dispatch_data_create(
        buffer: *const c_void,
        size: usize,
        queue: *mut c_void,
        destructor: *mut c_void,
    ) -> DispatchData;
    fn dispatch_release(object: *mut c_void);
    fn getpagesize() -> c_int;
    static mach_task_self_: MachPort;
    fn mach_vm_region(
        target_task: MachPort,
        address: *mut MachVmAddress,
        size: *mut MachVmSize,
        flavor: c_int,
        info: *mut c_int,
        info_count: *mut u32,
        object_name: *mut MachPort,
    ) -> c_int;
    fn mach_port_deallocate(task: MachPort, name: MachPort) -> c_int;
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> c_int;
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

#[inline]
fn selector(name: &CStr) -> ObjcSel {
    // SAFETY: the input is NUL-terminated and the runtime interns selectors.
    unsafe { sel_registerName(name.as_ptr()) }
}

#[inline]
fn msg_address() -> *const () {
    objc_msgSend as *const ()
}

#[inline]
unsafe fn send_id0(receiver: ObjcId, sel: ObjcSel) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel) }
}

#[inline]
unsafe fn send_id_cstr(receiver: ObjcId, sel: ObjcSel, value: *const c_char) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, *const c_char) -> ObjcId =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, value) }
}

#[inline]
unsafe fn send_id_data_error(
    receiver: ObjcId,
    sel: ObjcSel,
    data: DispatchData,
    error: *mut ObjcId,
) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, DispatchData, *mut ObjcId) -> ObjcId =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, data, error) }
}

#[inline]
unsafe fn send_id_id(receiver: ObjcId, sel: ObjcSel, value: ObjcId) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) -> ObjcId =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, value) }
}

#[inline]
unsafe fn send_id_id_error(
    receiver: ObjcId,
    sel: ObjcSel,
    value: ObjcId,
    error: *mut ObjcId,
) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, value, error) }
}

#[inline]
unsafe fn send_id_ptr_usize2_ptr(
    receiver: ObjcId,
    sel: ObjcSel,
    pointer: *mut c_void,
    length: usize,
    options: usize,
    deallocator: *mut c_void,
) -> ObjcId {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, *mut c_void, usize, usize, *mut c_void) -> ObjcId =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, pointer, length, options, deallocator) }
}

#[inline]
unsafe fn send_ptr0(receiver: ObjcId, sel: ObjcSel) -> *mut c_void {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel) -> *mut c_void =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel) }
}

#[inline]
unsafe fn send_cstr0(receiver: ObjcId, sel: ObjcSel) -> *const c_char {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel) }
}

#[inline]
unsafe fn send_usize0(receiver: ObjcId, sel: ObjcSel) -> usize {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel) -> usize =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel) }
}

#[inline]
unsafe fn send_f64_0(receiver: ObjcId, sel: ObjcSel) -> f64 {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel) -> f64 =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel) }
}

#[inline]
unsafe fn send_bool_sel(receiver: ObjcId, sel: ObjcSel, value: ObjcSel) -> bool {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, ObjcSel) -> c_uchar =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, value) != 0 }
}

#[inline]
unsafe fn send_void0(receiver: ObjcId, sel: ObjcSel) {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel) = unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel) }
}

#[inline]
unsafe fn send_void_id(receiver: ObjcId, sel: ObjcSel, value: ObjcId) {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId) =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, value) }
}

#[inline]
unsafe fn send_void_buffer_offset_index(
    receiver: ObjcId,
    sel: ObjcSel,
    buffer: ObjcId,
    offset: usize,
    index: usize,
) {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, usize, usize) =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, buffer, offset, index) }
}

#[inline]
unsafe fn send_void_bytes_len_index(
    receiver: ObjcId,
    sel: ObjcSel,
    bytes: *const c_void,
    length: usize,
    index: usize,
) {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, *const c_void, usize, usize) =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, bytes, length, index) }
}

#[inline]
unsafe fn send_void_sizes(receiver: ObjcId, sel: ObjcSel, grid: MtlSize, group: MtlSize) {
    let f: unsafe extern "C" fn(ObjcId, ObjcSel, MtlSize, MtlSize) =
        unsafe { std::mem::transmute(msg_address()) };
    unsafe { f(receiver, sel, grid, group) }
}

struct AutoreleasePool(ObjcId);

impl AutoreleasePool {
    fn new() -> Result<Self, String> {
        // SAFETY: NSAutoreleasePool is supplied by linked Foundation.
        let class = unsafe { objc_getClass(c"NSAutoreleasePool".as_ptr()) };
        if class.is_null() {
            return Err("Objective-C runtime did not provide NSAutoreleasePool".into());
        }
        let pool = unsafe { send_id0(class, selector(c"new")) };
        if pool.is_null() {
            return Err("failed to create NSAutoreleasePool".into());
        }
        Ok(Self(pool))
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        // SAFETY: this retained pool is drained exactly once.
        unsafe { send_void0(self.0, selector(c"drain")) };
    }
}

struct OwnedObject(ObjcId);

impl OwnedObject {
    fn require(value: ObjcId, what: &str, error: ObjcId) -> Result<Self, String> {
        if value.is_null() {
            return Err(format!("{what} failed{}", describe_error(error)));
        }
        Ok(Self(value))
    }

    #[inline]
    fn get(&self) -> ObjcId {
        self.0
    }
}

impl Drop for OwnedObject {
    fn drop(&mut self) {
        // SAFETY: every wrapped Create/new-family object has one retain.
        unsafe { send_void0(self.0, selector(c"release")) };
    }
}

struct OwnedDispatchData(DispatchData);

impl OwnedDispatchData {
    fn new(bytes: &[u8]) -> Result<Self, String> {
        // This copies only the immutable metallib metadata, never payload.
        let data = unsafe {
            dispatch_data_create(
                bytes.as_ptr().cast(),
                bytes.len(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if data.is_null() {
            return Err("dispatch_data_create failed".into());
        }
        Ok(Self(data))
    }
}

impl Drop for OwnedDispatchData {
    fn drop(&mut self) {
        // SAFETY: dispatch_data_create returned one retained object.
        unsafe { dispatch_release(self.0) };
    }
}

fn describe_error(error: ObjcId) -> String {
    if error.is_null() {
        return String::new();
    }
    let description = unsafe { send_id0(error, selector(c"localizedDescription")) };
    let text = nsstring_to_string(description);
    if text.is_empty() {
        String::new()
    } else {
        format!(": {text}")
    }
}

fn nsstring_to_string(string: ObjcId) -> String {
    if string.is_null() {
        return String::new();
    }
    let bytes = unsafe { send_cstr0(string, selector(c"UTF8String")) };
    if bytes.is_null() {
        return String::new();
    }
    // SAFETY: NSString guarantees a live NUL-terminated UTF-8 pointer.
    unsafe { CStr::from_ptr(bytes) }
        .to_string_lossy()
        .into_owned()
}

fn make_nsstring(value: &CStr) -> Result<ObjcId, String> {
    let class = unsafe { objc_getClass(c"NSString".as_ptr()) };
    if class.is_null() {
        return Err("Objective-C runtime did not provide NSString".into());
    }
    let string = unsafe { send_id_cstr(class, selector(c"stringWithUTF8String:"), value.as_ptr()) };
    if string.is_null() {
        return Err("failed to create NSString".into());
    }
    Ok(string)
}

fn vm_page_bytes() -> Result<usize, String> {
    let value = unsafe { getpagesize() };
    if value <= 0 {
        return Err("getpagesize returned a nonpositive value".into());
    }
    let page = value as usize;
    if !page.is_power_of_two() {
        return Err(format!("non-power-of-two VM page size {page}"));
    }
    Ok(page)
}

fn query_raw_vm_region(query_address: u64) -> Result<RawVmRegion, String> {
    let mut region_start = query_address;
    let mut region_size = 0u64;
    let mut info = VmRegionBasicInfo64::default();
    let expected_info_count =
        core::mem::size_of::<VmRegionBasicInfo64>() / core::mem::size_of::<c_int>();
    let mut info_count =
        u32::try_from(expected_info_count).map_err(|_| "VM region info count overflow")?;
    let mut object_name: MachPort = 0;
    let task = unsafe { mach_task_self_ };
    let result = unsafe {
        mach_vm_region(
            task,
            &mut region_start,
            &mut region_size,
            VM_REGION_BASIC_INFO_64,
            (&mut info as *mut VmRegionBasicInfo64).cast(),
            &mut info_count,
            &mut object_name,
        )
    };
    let object_name_was_nonnull = object_name != 0;
    let object_release_result = if object_name_was_nonnull {
        Some(unsafe { mach_port_deallocate(task, object_name) })
    } else {
        None
    };
    if result != 0 {
        return Err(format!(
            "mach_vm_region({query_address:#x}) failed with kern_return_t={result}; object_name_release={object_release_result:?}"
        ));
    }
    if let Some(release_result) = object_release_result
        && release_result != 0
    {
        return Err(format!(
            "mach_port_deallocate(object_name) failed with kern_return_t={release_result}"
        ));
    }
    if info_count < expected_info_count as u32 {
        return Err(format!(
            "mach_vm_region returned {info_count} info words, expected {expected_info_count}"
        ));
    }
    let region_end = region_start
        .checked_add(region_size)
        .ok_or("returned VM region end overflow")?;
    if region_end <= query_address {
        return Err(format!(
            "mach_vm_region made no progress at {query_address:#x}: returned [{region_start:#x},{region_end:#x})"
        ));
    }
    Ok(RawVmRegion {
        region_start,
        region_end,
        protection: info.protection,
        max_protection: info.max_protection,
        shared: info.shared != 0,
        reserved: info.reserved != 0,
        object_name_was_nonnull,
        object_name_deallocated: object_release_result.is_none_or(|release| release == 0),
    })
}

fn enumerate_vm_segments(
    pointer: *mut c_void,
    length: usize,
    bytes_per_leaf: usize,
    max_segments: usize,
    label: &str,
) -> Result<Vec<VmSegment>, String> {
    if pointer.is_null() || length == 0 || bytes_per_leaf == 0 {
        return Err(format!(
            "{label}: segment enumeration requires non-null/non-empty range and nonzero leaf size"
        ));
    }
    let page_bytes = vm_page_bytes()?;
    let allocation_start = pointer as usize;
    if allocation_start % page_bytes != 0 || length % page_bytes != 0 {
        return Err(format!(
            "{label}: range violates Metal no-copy page contract: pointer={allocation_start:#x}, length={length}, page={page_bytes}; no fallback"
        ));
    }
    if length % bytes_per_leaf != 0 {
        return Err(format!(
            "{label}: allocation length {length} is not a multiple of leaf bytes {bytes_per_leaf}"
        ));
    }
    let allocation_start_u64 = u64::try_from(allocation_start)
        .map_err(|_| format!("{label}: pointer does not fit mach_vm_address_t"))?;
    let allocation_end = allocation_start_u64
        .checked_add(u64::try_from(length).map_err(|_| format!("{label}: length overflow"))?)
        .ok_or_else(|| format!("{label}: range end overflow"))?;
    let expected_leaves = length / bytes_per_leaf;
    let required_protection = VM_PROT_READ | VM_PROT_WRITE;
    let mut cursor = allocation_start_u64;
    let mut segments = Vec::new();

    while cursor < allocation_end {
        if segments.len() >= max_segments {
            return Err(format!(
                "{label}: VM partition count exceeds cap {max_segments}; no fallback"
            ));
        }
        let raw = query_raw_vm_region(cursor)?;
        if raw.region_start > cursor {
            return Err(format!(
                "{label}: VM gap at {cursor:#x}, next region starts {:#x}",
                raw.region_start
            ));
        }
        if raw.protection & required_protection != required_protection {
            return Err(format!(
                "{label}: VM region [{:#x},{:#x}) lacks current RW protection: {:#x}",
                raw.region_start, raw.region_end, raw.protection
            ));
        }
        if !raw.object_name_deallocated {
            return Err(format!(
                "{label}: mach_vm_region object-name right was not deallocated"
            ));
        }
        let intersection_end = raw.region_end.min(allocation_end);
        if intersection_end <= cursor {
            return Err(format!(
                "{label}: non-advancing VM intersection at {cursor:#x}"
            ));
        }
        let segment_pointer =
            usize::try_from(cursor).map_err(|_| format!("{label}: segment pointer overflow"))?;
        let segment_length = usize::try_from(intersection_end - cursor)
            .map_err(|_| format!("{label}: segment length overflow"))?;
        if segment_pointer % page_bytes != 0 || segment_length % page_bytes != 0 {
            return Err(format!(
                "{label}: VM intersection violates page contract: pointer={segment_pointer:#x}, length={segment_length}, page={page_bytes}"
            ));
        }
        let allocation_offset = usize::try_from(cursor - allocation_start_u64)
            .map_err(|_| format!("{label}: allocation offset overflow"))?;
        let allocation_end_offset = allocation_offset
            .checked_add(segment_length)
            .ok_or_else(|| format!("{label}: segment end offset overflow"))?;
        if allocation_offset % bytes_per_leaf != 0 || allocation_end_offset % bytes_per_leaf != 0 {
            return Err(format!(
                "{label}: VM boundaries do not map to whole {bytes_per_leaf}-byte leaves: [{allocation_offset},{allocation_end_offset})"
            ));
        }
        let segment_index = segments.len();
        segments.push(VmSegment {
            segment_index,
            allocation_offset_bytes: allocation_offset,
            pointer: segment_pointer,
            length_bytes: segment_length,
            leaf_start: allocation_offset / bytes_per_leaf,
            leaf_end: allocation_end_offset / bytes_per_leaf,
            raw_region_start: raw.region_start,
            raw_region_end: raw.region_end,
            protection: raw.protection,
            max_protection: raw.max_protection,
            shared: raw.shared,
            reserved: raw.reserved,
            object_name_was_nonnull: raw.object_name_was_nonnull,
            object_name_deallocated: raw.object_name_deallocated,
        });
        cursor = intersection_end;
    }

    if segments.is_empty()
        || segments[0].leaf_start != 0
        || segments
            .last()
            .is_none_or(|segment| segment.leaf_end != expected_leaves)
    {
        return Err(format!(
            "{label}: VM partitions do not cover exact leaf range [0,{expected_leaves})"
        ));
    }
    for (index, segment) in segments.iter().enumerate() {
        if segment.segment_index != index || segment.leaf_start >= segment.leaf_end {
            return Err(format!(
                "{label}: invalid segment metadata at index {index}"
            ));
        }
        if index > 0 {
            let previous = &segments[index - 1];
            if previous.leaf_end != segment.leaf_start
                || previous
                    .pointer
                    .checked_add(previous.length_bytes)
                    .is_none_or(|end| end != segment.pointer)
            {
                return Err(format!(
                    "{label}: partitions are not gap-free at segment {index}"
                ));
            }
        }
    }
    Ok(segments)
}

fn validate_leaf_segment_cover(
    segments: &[VmSegment],
    expected_leaves: usize,
    bytes_per_leaf: usize,
    cap: usize,
    label: &str,
) -> Result<(), String> {
    if segments.is_empty() || segments.len() > cap {
        return Err(format!(
            "{label}: segment count {} outside 1..={cap}",
            segments.len()
        ));
    }
    let mut next_leaf = 0usize;
    for (index, segment) in segments.iter().enumerate() {
        if segment.segment_index != index
            || segment.leaf_start != next_leaf
            || segment.leaf_start >= segment.leaf_end
        {
            return Err(format!(
                "{label}: non-exact coverage at segment {index}: expected {next_leaf}, got [{},{})",
                segment.leaf_start, segment.leaf_end
            ));
        }
        let expected_length = (segment.leaf_end - segment.leaf_start)
            .checked_mul(bytes_per_leaf)
            .ok_or_else(|| format!("{label}: segment byte length overflow"))?;
        if segment.length_bytes != expected_length {
            return Err(format!(
                "{label}: segment {index} length {} != {expected_length}",
                segment.length_bytes
            ));
        }
        next_leaf = segment.leaf_end;
    }
    if next_leaf != expected_leaves {
        return Err(format!(
            "{label}: segment cover ends at {next_leaf}, expected {expected_leaves}"
        ));
    }
    Ok(())
}

fn build_joint_leaf_intervals(
    input_segments: &[VmSegment],
    output_segments: &[VmSegment],
) -> Result<Vec<JointLeafInterval>, String> {
    validate_leaf_segment_cover(
        input_segments,
        GPU_LEAVES,
        LEAF_BYTES,
        MAX_INPUT_SEGMENTS,
        "Metal input suffix",
    )?;
    validate_leaf_segment_cover(
        output_segments,
        GPU_LEAVES,
        OUTPUT_BYTES_PER_LEAF,
        MAX_OUTPUT_SEGMENTS,
        "Metal output suffix",
    )?;

    let mut boundaries = Vec::with_capacity(input_segments.len() + output_segments.len() + 2);
    boundaries.push(0usize);
    boundaries.push(GPU_LEAVES);
    for segment in input_segments.iter().chain(output_segments) {
        boundaries.push(segment.leaf_start);
        boundaries.push(segment.leaf_end);
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    if boundaries.first() != Some(&0) || boundaries.last() != Some(&GPU_LEAVES) {
        return Err("joint boundaries do not span the exact GPU suffix".into());
    }
    let interval_count = boundaries.len().saturating_sub(1);
    if interval_count == 0 || interval_count > MAX_JOINT_INTERVALS {
        return Err(format!(
            "joint interval count {interval_count} outside 1..={MAX_JOINT_INTERVALS}"
        ));
    }

    let mut intervals = Vec::with_capacity(interval_count);
    for (interval_index, pair) in boundaries.windows(2).enumerate() {
        let start = pair[0];
        let end = pair[1];
        if start >= end {
            return Err(format!("empty joint interval at {start}"));
        }
        let input_segment = input_segments
            .iter()
            .find(|segment| segment.leaf_start <= start && end <= segment.leaf_end)
            .ok_or_else(|| format!("interval [{start},{end}) lacks input segment"))?;
        let output_segment = output_segments
            .iter()
            .find(|segment| segment.leaf_start <= start && end <= segment.leaf_end)
            .ok_or_else(|| format!("interval [{start},{end}) lacks output segment"))?;
        let leaf_count = end - start;
        let input_byte_offset = (start - input_segment.leaf_start)
            .checked_mul(LEAF_BYTES)
            .ok_or("input interval offset overflow")?;
        let output_byte_offset = (start - output_segment.leaf_start)
            .checked_mul(OUTPUT_BYTES_PER_LEAF)
            .ok_or("output interval offset overflow")?;
        let input_bytes = leaf_count
            .checked_mul(LEAF_BYTES)
            .ok_or("input interval length overflow")?;
        let output_bytes = leaf_count
            .checked_mul(OUTPUT_BYTES_PER_LEAF)
            .ok_or("output interval length overflow")?;
        if input_byte_offset
            .checked_add(input_bytes)
            .is_none_or(|limit| limit > input_segment.length_bytes)
            || output_byte_offset
                .checked_add(output_bytes)
                .is_none_or(|limit| limit > output_segment.length_bytes)
        {
            return Err(format!("interval [{start},{end}) escapes selected buffers"));
        }
        if input_byte_offset % 16 != 0 || output_byte_offset % 16 != 0 {
            return Err(format!("interval [{start},{end}) is not uint4-aligned"));
        }
        let input_uint4_offset = input_byte_offset / 16;
        let output_uint4_offset = output_byte_offset / 16;
        let input_last_uint4 = input_uint4_offset
            .checked_add(input_bytes / 16)
            .and_then(|limit| limit.checked_sub(1))
            .ok_or("input shader index overflow")?;
        let output_last_uint4 = output_uint4_offset
            .checked_add(output_bytes / 16)
            .and_then(|limit| limit.checked_sub(1))
            .ok_or("output shader index overflow")?;
        if input_last_uint4 > i32::MAX as usize || output_last_uint4 > i32::MAX as usize {
            return Err(format!(
                "interval [{start},{end}) exceeds signed shader indexing"
            ));
        }
        let global_leaf_start = CPU_LEAVES
            .checked_add(start)
            .ok_or("global leaf offset overflow")?;
        if global_leaf_start
            .checked_add(leaf_count)
            .is_none_or(|limit| limit > TOTAL_LEAVES)
        {
            return Err(format!("interval [{start},{end}) escapes global L0"));
        }
        intervals.push(JointLeafInterval {
            interval_index,
            global_leaf_start,
            leaf_count,
            input_segment_index: input_segment.segment_index,
            output_segment_index: output_segment.segment_index,
            input_byte_offset,
            output_byte_offset,
            input_uint4_offset: u32::try_from(input_uint4_offset)
                .map_err(|_| "input uint4 offset exceeds u32")?,
            output_uint4_offset: u32::try_from(output_uint4_offset)
                .map_err(|_| "output uint4 offset exceeds u32")?,
        });
    }
    let covered: usize = intervals.iter().map(|interval| interval.leaf_count).sum();
    if covered != GPU_LEAVES {
        return Err(format!(
            "joint intervals cover {covered} leaves, expected {GPU_LEAVES}"
        ));
    }
    Ok(intervals)
}

fn require_no_copy_identity(
    buffer: ObjcId,
    expected_pointer: *mut c_void,
    expected_bytes: usize,
    label: &str,
) -> Result<(), String> {
    let length = unsafe { send_usize0(buffer, selector(c"length")) };
    let contents = unsafe { send_ptr0(buffer, selector(c"contents")) };
    let storage_mode = unsafe { send_usize0(buffer, selector(c"storageMode")) };
    if length != expected_bytes
        || contents != expected_pointer
        || storage_mode != MTL_STORAGE_MODE_SHARED
    {
        return Err(format!(
            "{label}: no-copy identity failed: length={length}/{expected_bytes}, original={expected_pointer:p}, contents={contents:p}, storageMode={storage_mode}; no fallback"
        ));
    }
    Ok(())
}

fn create_no_copy_segment_buffers(
    device: ObjcId,
    segments: &[VmSegment],
    max_buffer_length: usize,
    label: &str,
) -> Result<Vec<OwnedObject>, String> {
    let mut buffers = Vec::with_capacity(segments.len());
    for segment in segments {
        if segment.length_bytes > max_buffer_length {
            return Err(format!(
                "{label} segment {} length {} exceeds maxBufferLength {max_buffer_length}; no fallback",
                segment.segment_index, segment.length_bytes
            ));
        }
        let pointer = segment.pointer as *mut c_void;
        // Exactly one no-copy constructor attempt per VM intersection.
        let buffer = OwnedObject::require(
            unsafe {
                send_id_ptr_usize2_ptr(
                    device,
                    selector(c"newBufferWithBytesNoCopy:length:options:deallocator:"),
                    pointer,
                    segment.length_bytes,
                    MTL_STORAGE_MODE_SHARED,
                    std::ptr::null_mut(),
                )
            },
            &format!("{label} segment {} no-copy buffer", segment.segment_index),
            std::ptr::null_mut(),
        )?;
        require_no_copy_identity(
            buffer.get(),
            pointer,
            segment.length_bytes,
            &format!("{label} segment {} creation", segment.segment_index),
        )?;
        buffers.push(buffer);
    }
    Ok(buffers)
}

fn require_segment_buffer_identities(
    buffers: &[OwnedObject],
    segments: &[VmSegment],
    label: &str,
) -> Result<(), String> {
    if buffers.len() != segments.len() {
        return Err(format!(
            "{label}: buffer/segment count mismatch {}/{}",
            buffers.len(),
            segments.len()
        ));
    }
    for (buffer, segment) in buffers.iter().zip(segments) {
        require_no_copy_identity(
            buffer.get(),
            segment.pointer as *mut c_void,
            segment.length_bytes,
            &format!("{label} segment {} identity", segment.segment_index),
        )?;
    }
    Ok(())
}

struct MetalContext {
    queue: OwnedObject,
    pipeline: OwnedObject,
    _function: OwnedObject,
    _library: OwnedObject,
    device: OwnedObject,
    device_name: String,
    max_buffer_length: usize,
    thread_execution_width: usize,
    max_threads_per_threadgroup: usize,
    seconds_per_host_tick: f64,
    submission: Mutex<()>,
}

// The retained Metal objects are process-lifetime and Metal documents these
// objects as thread-safe. Every command submission is additionally serialized
// by `submission`, including its terminal wait and caller-owned buffer lifetime.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    fn new() -> Result<Self, String> {
        let _pool = AutoreleasePool::new()?;
        let device = OwnedObject::require(
            unsafe { MTLCreateSystemDefaultDevice() },
            "MTLCreateSystemDefaultDevice",
            std::ptr::null_mut(),
        )?;
        let device_name = nsstring_to_string(unsafe { send_id0(device.get(), selector(c"name")) });
        let max_buffer_length = unsafe { send_usize0(device.get(), selector(c"maxBufferLength")) };
        let data = OwnedDispatchData::new(METALLIB)?;
        let mut error = std::ptr::null_mut();
        let library = OwnedObject::require(
            unsafe {
                send_id_data_error(
                    device.get(),
                    selector(c"newLibraryWithData:error:"),
                    data.0,
                    &mut error,
                )
            },
            "newLibraryWithData:error:",
            error,
        )?;
        let function_name = make_nsstring(FUNCTION_NAME)?;
        let function = OwnedObject::require(
            unsafe {
                send_id_id(
                    library.get(),
                    selector(c"newFunctionWithName:"),
                    function_name,
                )
            },
            "newFunctionWithName:",
            std::ptr::null_mut(),
        )?;
        error = std::ptr::null_mut();
        let pipeline = OwnedObject::require(
            unsafe {
                send_id_id_error(
                    device.get(),
                    selector(c"newComputePipelineStateWithFunction:error:"),
                    function.get(),
                    &mut error,
                )
            },
            "newComputePipelineStateWithFunction:error:",
            error,
        )?;
        let thread_execution_width =
            unsafe { send_usize0(pipeline.get(), selector(c"threadExecutionWidth")) };
        let max_threads_per_threadgroup =
            unsafe { send_usize0(pipeline.get(), selector(c"maxTotalThreadsPerThreadgroup")) };
        if thread_execution_width == 0
            || !THREADS_PER_THREADGROUP.is_multiple_of(thread_execution_width)
            || max_threads_per_threadgroup < THREADS_PER_THREADGROUP
        {
            return Err(format!(
                "pipeline cannot run frozen group=32: execution_width={thread_execution_width}, max={max_threads_per_threadgroup}"
            ));
        }
        let queue = OwnedObject::require(
            unsafe { send_id0(device.get(), selector(c"newCommandQueue")) },
            "newCommandQueue",
            std::ptr::null_mut(),
        )?;
        let mut timebase = MachTimebaseInfo::default();
        let timebase_result = unsafe { mach_timebase_info(&mut timebase) };
        if timebase_result != 0 || timebase.numer == 0 || timebase.denom == 0 {
            return Err(format!(
                "mach_timebase_info failed: result={timebase_result}, numer={}, denom={}",
                timebase.numer, timebase.denom
            ));
        }
        let seconds_per_host_tick = f64::from(timebase.numer) / f64::from(timebase.denom) / 1.0e9;
        Ok(Self {
            queue,
            pipeline,
            _function: function,
            _library: library,
            device,
            device_name,
            max_buffer_length,
            thread_execution_width,
            max_threads_per_threadgroup,
            seconds_per_host_tick,
            submission: Mutex::new(()),
        })
    }

    #[inline]
    fn host_time_seconds(&self) -> f64 {
        (unsafe { mach_absolute_time() }) as f64 * self.seconds_per_host_tick
    }
}

static CONTEXT: OnceLock<Result<MetalContext, String>> = OnceLock::new();

fn context() -> Result<&'static MetalContext, String> {
    CONTEXT
        .get_or_init(MetalContext::new)
        .as_ref()
        .map_err(Clone::clone)
}

pub(super) fn init() -> Result<(), String> {
    context().map(|_| ())
}

pub(super) fn matches_exact_l0(input: &[u8], leaf_size: usize, output: &[Hash]) -> bool {
    CPU_LEAVES + GPU_LEAVES == TOTAL_LEAVES
        && input.len() == TOTAL_LEAVES * LEAF_BYTES
        && leaf_size == LEAF_BYTES
        && output.len() == TOTAL_LEAVES
}

fn encode_intervals(
    encoder: ObjcId,
    pipeline: ObjcId,
    input_buffers: &[OwnedObject],
    output_buffers: &[OwnedObject],
    intervals: &[JointLeafInterval],
) -> Result<(), String> {
    if intervals.is_empty() || intervals.len() > MAX_JOINT_INTERVALS {
        return Err(format!(
            "cannot encode {} intervals (cap={MAX_JOINT_INTERVALS})",
            intervals.len()
        ));
    }
    unsafe { send_void_id(encoder, selector(c"setComputePipelineState:"), pipeline) };
    for interval in intervals {
        let input_buffer = input_buffers
            .get(interval.input_segment_index)
            .ok_or_else(|| format!("interval {} input buffer missing", interval.interval_index))?;
        let output_buffer = output_buffers
            .get(interval.output_segment_index)
            .ok_or_else(|| format!("interval {} output buffer missing", interval.interval_index))?;
        let global_leaf_base = u32::try_from(interval.global_leaf_start)
            .map_err(|_| "global leaf base exceeds u32")?;
        let local_leaf_count =
            u32::try_from(interval.leaf_count).map_err(|_| "local leaf count exceeds u32")?;
        unsafe {
            send_void_buffer_offset_index(
                encoder,
                selector(c"setBuffer:offset:atIndex:"),
                input_buffer.get(),
                0,
                0,
            );
            send_void_buffer_offset_index(
                encoder,
                selector(c"setBuffer:offset:atIndex:"),
                output_buffer.get(),
                0,
                1,
            );
            send_void_bytes_len_index(
                encoder,
                selector(c"setBytes:length:atIndex:"),
                (&global_leaf_base as *const u32).cast(),
                core::mem::size_of::<u32>(),
                2,
            );
            send_void_bytes_len_index(
                encoder,
                selector(c"setBytes:length:atIndex:"),
                (&local_leaf_count as *const u32).cast(),
                core::mem::size_of::<u32>(),
                3,
            );
            send_void_bytes_len_index(
                encoder,
                selector(c"setBytes:length:atIndex:"),
                (&interval.input_uint4_offset as *const u32).cast(),
                core::mem::size_of::<u32>(),
                4,
            );
            send_void_bytes_len_index(
                encoder,
                selector(c"setBytes:length:atIndex:"),
                (&interval.output_uint4_offset as *const u32).cast(),
                core::mem::size_of::<u32>(),
                5,
            );
            send_void_sizes(
                encoder,
                selector(c"dispatchThreads:threadsPerThreadgroup:"),
                MtlSize {
                    width: interval.leaf_count,
                    height: 1,
                    depth: 1,
                },
                MtlSize {
                    width: THREADS_PER_THREADGROUP,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }
    Ok(())
}

pub(super) struct InFlight<'a> {
    command_buffer: ObjcId,
    context: &'static MetalContext,
    input_segments: Vec<VmSegment>,
    output_segments: Vec<VmSegment>,
    intervals: Vec<JointLeafInterval>,
    input_buffers: Vec<OwnedObject>,
    output_buffers: Vec<OwnedObject>,
    _submission: MutexGuard<'static, ()>,
    _pool: AutoreleasePool,
    commit_host_seconds: f64,
    completed: bool,
    _payload_borrow: PhantomData<(&'a [u8], &'a mut [Hash])>,
}

impl InFlight<'_> {
    /// Capture the CPU completion point and perform the mandatory terminal
    /// wait before either suffix slice can become accessible again.
    pub(super) fn finish(mut self) -> Result<HybridStats, String> {
        let cpu_finish_host_seconds = self.context.host_time_seconds();
        let wait_start = std::time::Instant::now();
        unsafe { send_void0(self.command_buffer, selector(c"waitUntilCompleted")) };
        let wait_nanoseconds = wait_start.elapsed().as_nanos();
        self.completed = true;

        let command_status = unsafe { send_usize0(self.command_buffer, selector(c"status")) };
        if command_status != COMMAND_STATUS_COMPLETED {
            let error = unsafe { send_id0(self.command_buffer, selector(c"error")) };
            return Err(format!(
                "Metal L0 suffix command status {command_status}{}",
                describe_error(error)
            ));
        }
        let start_selector = selector(c"GPUStartTime");
        let end_selector = selector(c"GPUEndTime");
        let has_start = unsafe {
            send_bool_sel(
                self.command_buffer,
                selector(c"respondsToSelector:"),
                start_selector,
            )
        };
        let has_end = unsafe {
            send_bool_sel(
                self.command_buffer,
                selector(c"respondsToSelector:"),
                end_selector,
            )
        };
        if !has_start || !has_end {
            return Err(format!(
                "Metal GPU timestamps unavailable (start={has_start}, end={has_end})"
            ));
        }
        let gpu_start_seconds = unsafe { send_f64_0(self.command_buffer, start_selector) };
        let gpu_end_seconds = unsafe { send_f64_0(self.command_buffer, end_selector) };
        if !gpu_start_seconds.is_finite()
            || !gpu_end_seconds.is_finite()
            || gpu_start_seconds < 0.0
            || gpu_end_seconds <= gpu_start_seconds
        {
            return Err(format!(
                "invalid GPU interval start={gpu_start_seconds:?}, end={gpu_end_seconds:?}"
            ));
        }

        require_segment_buffer_identities(
            &self.input_buffers,
            &self.input_segments,
            "Metal input suffix post-wait",
        )?;
        require_segment_buffer_identities(
            &self.output_buffers,
            &self.output_segments,
            "Metal output suffix post-wait",
        )?;
        let identities_per_gate = self.input_segments.len() + self.output_segments.len();
        let input_buffer_attempts = self.input_segments.len();
        let output_buffer_attempts = self.output_segments.len();
        Ok(HybridStats {
            device_name: self.context.device_name.as_str(),
            max_buffer_length: self.context.max_buffer_length,
            thread_execution_width: self.context.thread_execution_width,
            max_threads_per_threadgroup: self.context.max_threads_per_threadgroup,
            input_segments: std::mem::take(&mut self.input_segments),
            output_segments: std::mem::take(&mut self.output_segments),
            intervals: std::mem::take(&mut self.intervals),
            input_buffer_attempts,
            output_buffer_attempts,
            no_copy_identity_checks: 3 * identities_per_gate,
            commit_host_seconds: self.commit_host_seconds,
            cpu_finish_host_seconds,
            wait_nanoseconds,
            gpu_start_seconds,
            gpu_end_seconds,
            command_status,
        })
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if !self.completed && !self.command_buffer.is_null() {
            // A committed command must outlive every no-copy payload buffer,
            // including unwinding from CPU work or telemetry validation.
            unsafe { send_void0(self.command_buffer, selector(c"waitUntilCompleted")) };
            self.completed = true;
        }
    }
}

pub(super) fn submit<'a>(
    input_suffix: &'a [u8],
    output_suffix: &'a mut [Hash],
) -> Result<InFlight<'a>, String> {
    if input_suffix.len() != GPU_INPUT_BYTES || output_suffix.len() != GPU_LEAVES {
        return Err(format!(
            "fixed Metal suffix geometry mismatch: input={}/{GPU_INPUT_BYTES}, output={}/{GPU_LEAVES}",
            input_suffix.len(),
            output_suffix.len()
        ));
    }
    let output_bytes = output_suffix
        .len()
        .checked_mul(OUTPUT_BYTES_PER_LEAF)
        .ok_or("Metal output suffix byte length overflow")?;
    if output_bytes != GPU_OUTPUT_BYTES {
        return Err(format!(
            "fixed Metal output bytes mismatch: {output_bytes}/{GPU_OUTPUT_BYTES}"
        ));
    }

    let input_segments = enumerate_vm_segments(
        input_suffix.as_ptr() as *mut c_void,
        input_suffix.len(),
        LEAF_BYTES,
        MAX_INPUT_SEGMENTS,
        "Metal input suffix",
    )?;
    let output_segments = enumerate_vm_segments(
        output_suffix.as_mut_ptr().cast(),
        output_bytes,
        OUTPUT_BYTES_PER_LEAF,
        MAX_OUTPUT_SEGMENTS,
        "Metal output suffix",
    )?;
    let intervals = build_joint_leaf_intervals(&input_segments, &output_segments)?;
    let context = context()?;
    let submission = context
        .submission
        .lock()
        .map_err(|_| "Metal submission mutex poisoned".to_owned())?;
    let pool = AutoreleasePool::new()?;
    let input_buffers = create_no_copy_segment_buffers(
        context.device.get(),
        &input_segments,
        context.max_buffer_length,
        "Metal input suffix",
    )?;
    let output_buffers = create_no_copy_segment_buffers(
        context.device.get(),
        &output_segments,
        context.max_buffer_length,
        "Metal output suffix",
    )?;
    require_segment_buffer_identities(
        &input_buffers,
        &input_segments,
        "Metal input suffix pre-submit",
    )?;
    require_segment_buffer_identities(
        &output_buffers,
        &output_segments,
        "Metal output suffix pre-submit",
    )?;

    let command_buffer = unsafe { send_id0(context.queue.get(), selector(c"commandBuffer")) };
    if command_buffer.is_null() {
        return Err("Metal commandBuffer returned nil".into());
    }
    let encoder = unsafe { send_id0(command_buffer, selector(c"computeCommandEncoder")) };
    if encoder.is_null() {
        return Err("Metal computeCommandEncoder returned nil".into());
    }
    encode_intervals(
        encoder,
        context.pipeline.get(),
        &input_buffers,
        &output_buffers,
        &intervals,
    )?;
    unsafe { send_void0(encoder, selector(c"endEncoding")) };
    let commit_host_seconds = context.host_time_seconds();
    // No fallible operation is permitted between commit and returning the
    // guard whose Drop performs the terminal wait.
    unsafe { send_void0(command_buffer, selector(c"commit")) };

    Ok(InFlight {
        command_buffer,
        context,
        input_segments,
        output_segments,
        intervals,
        input_buffers,
        output_buffers,
        _submission: submission,
        _pool: pool,
        commit_host_seconds,
        completed: false,
        _payload_borrow: PhantomData,
    })
}

fn emit_trace(stats: &HybridStats) {
    eprintln!(
        "[merkle-metal-hybrid] device={:?} cpu_leaves={} gpu_leaves={} leaf_bytes={} input_bytes={} output_bytes={} max_buffer_length={} execution_width={} max_threads={} group={} input_segments={} output_segments={} input_buffer_attempts={} output_buffer_attempts={} identity_checks={} dispatches={} commit_host_s={:.9} cpu_finish_host_s={:.9} wait_ns={} gpu_start_s={:.9} gpu_end_s={:.9} gpu_elapsed_ns={:.0} status={}",
        stats.device_name,
        CPU_LEAVES,
        GPU_LEAVES,
        LEAF_BYTES,
        GPU_INPUT_BYTES,
        GPU_OUTPUT_BYTES,
        stats.max_buffer_length,
        stats.thread_execution_width,
        stats.max_threads_per_threadgroup,
        THREADS_PER_THREADGROUP,
        stats.input_segments.len(),
        stats.output_segments.len(),
        stats.input_buffer_attempts,
        stats.output_buffer_attempts,
        stats.no_copy_identity_checks,
        stats.intervals.len(),
        stats.commit_host_seconds,
        stats.cpu_finish_host_seconds,
        stats.wait_nanoseconds,
        stats.gpu_start_seconds,
        stats.gpu_end_seconds,
        (stats.gpu_end_seconds - stats.gpu_start_seconds) * 1.0e9,
        stats.command_status,
    );
    for (kind, segments) in [
        ("input", stats.input_segments.as_slice()),
        ("output", stats.output_segments.as_slice()),
    ] {
        for segment in segments {
            eprintln!(
                "[merkle-metal-vm] kind={kind} index={} allocation_offset={} pointer={:#x} bytes={} leaves=[{},{}) raw_region=[{:#x},{:#x}) protection={:#x} max_protection={:#x} shared={} reserved={} object_name_nonnull={} object_name_deallocated={}",
                segment.segment_index,
                segment.allocation_offset_bytes,
                segment.pointer,
                segment.length_bytes,
                segment.leaf_start,
                segment.leaf_end,
                segment.raw_region_start,
                segment.raw_region_end,
                segment.protection,
                segment.max_protection,
                segment.shared,
                segment.reserved,
                segment.object_name_was_nonnull,
                segment.object_name_deallocated,
            );
        }
    }
    for interval in &stats.intervals {
        eprintln!(
            "[merkle-metal-dispatch] index={} global_leaves=[{},{}) threads={} group={} input_segment={} output_segment={} input_byte_offset={} output_byte_offset={} input_uint4_offset={} output_uint4_offset={}",
            interval.interval_index,
            interval.global_leaf_start,
            interval.global_leaf_start + interval.leaf_count,
            interval.leaf_count,
            THREADS_PER_THREADGROUP,
            interval.input_segment_index,
            interval.output_segment_index,
            interval.input_byte_offset,
            interval.output_byte_offset,
            interval.input_uint4_offset,
            interval.output_uint4_offset,
        );
    }
}

#[cfg(any(test, feature = "hash-count"))]
static TEST_STATS: Mutex<Option<HybridStats>> = Mutex::new(None);

pub(super) fn trace(stats: HybridStats) {
    if super::strict_env_enabled("FLOCK_MERKLE_METAL_HYBRID_TRACE") {
        emit_trace(&stats);
    }
    // Move, rather than clone, the record into a single test-only slot. The
    // harness takes it after stopping the full-tree timer and emits it then.
    #[cfg(any(test, feature = "hash-count"))]
    {
        let mut slot = TEST_STATS.lock().unwrap();
        *slot = Some(stats);
    }
}

#[cfg(any(test, feature = "hash-count"))]
pub(super) fn take_test_stats() -> Option<HybridStats> {
    TEST_STATS.lock().unwrap().take()
}

#[cfg(test)]
pub(super) fn emit_test_trace(stats: &HybridStats) {
    emit_trace(stats);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_split_is_exact_and_page_integral() {
        assert_eq!(CPU_LEAVES + GPU_LEAVES, TOTAL_LEAVES);
        assert_eq!(CPU_LEAVES, 589_824);
        assert_eq!(GPU_LEAVES, 458_752);
        assert_eq!(GPU_INPUT_BYTES, 448 << 20);
        assert_eq!(GPU_OUTPUT_BYTES, 14 << 20);
        assert_eq!(CPU_LEAVES * LEAF_BYTES, 576 << 20);
        assert_eq!(CPU_LEAVES * OUTPUT_BYTES_PER_LEAF, 18 << 20);
        assert!(METAL_SOURCE.contains("kernel void sha256_leaf_checksum("));
        assert!(!METALLIB.is_empty());
    }
}
