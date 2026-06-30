//! OpenVM IO runtime: ctx (`OpenVmIoState`) borrowed from `VmState<F>`.

use std::{collections::VecDeque, ffi::c_void};

#[cfg(not(feature = "unprotected"))]
use openvm_platform::memory::MEM_SIZE;
use openvm_stark_backend::p3_field::PrimeField32;
use rand::rngs::StdRng;

use crate::arch::deferral::DeferralState;

/// IO execution state borrowed from the host `VmState<F>` for the duration of
/// one rvr call. Streams, rng, and the public-values byte slice are mutable
/// borrows; `memory_ptr` is a raw alias of VmState's main memory buffer
/// (raw because the C engine accesses it directly via pointer).
///
/// `deferral_memory` aliases AS=4 as `F` cells for deferral accumulator updates.
pub struct OpenVmIoState<'a, F: PrimeField32> {
    pub input_stream: &'a mut VecDeque<Vec<F>>,
    pub hint_stream: &'a mut VecDeque<F>,
    pub rng: &'a mut StdRng,
    pub memory_ptr: *mut u8,
    pub public_values: &'a mut [u8],
    pub deferral_memory: *mut F,
    pub deferral_memory_len: usize,
    pub deferrals: &'a mut Vec<DeferralState>,
}

/// Verify that `[start, start + num_bytes)` fits within AS_MEMORY
/// (`MEM_SIZE` bytes). Panics with a "Memory access out of bounds"
/// message on overflow; panicking across `extern "C"` aborts the process,
/// matching the C-side `abort_oob` termination used by `rd_mem_*`/`wr_mem_*`.
/// Compiles to a no-op under the `unprotected` feature.
#[cfg(not(feature = "unprotected"))]
pub fn check_mem_bounds_range(start: u64, num_bytes: usize) {
    let start = start as usize;
    if start > MEM_SIZE || num_bytes > MEM_SIZE - start {
        panic!(
            "Memory access out of bounds: start={start} size={num_bytes} memory_size={MEM_SIZE}"
        );
    }
}

#[cfg(feature = "unprotected")]
#[inline(always)]
pub fn check_mem_bounds_range(_start: u64, _num_bytes: usize) {}

/// Replace the hint stream contents. Called via `ext_hint_stream_set` from extension FFI.
///
/// # Safety
///
/// `ctx` must be a valid `OpenVmIoState` pointer. `data` must point to `len` bytes (or be null).
pub unsafe extern "C" fn host_hint_stream_set<F: PrimeField32>(
    ctx: *mut c_void,
    data: *const u8,
    len: u32,
) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    io.hint_stream.clear();
    if len > 0 && !data.is_null() {
        let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
        for &b in slice {
            io.hint_stream.push_back(F::from_u8(b));
        }
    }
}
