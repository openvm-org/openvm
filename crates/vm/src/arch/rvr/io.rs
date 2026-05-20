//! OpenVM IO runtime: Rust-side IO state with FFI callbacks.
//!
//! All execution IO state lives on the openvm `VmState<F>`; callbacks are
//! generic over the openvm field `F` and read/write VmState directly.
//! `Streams<F>` is consumed lazily — the F→u8 cast happens one byte at a
//! time inside the storew/buffer callbacks rather than upfront. Metering
//! adjustments for IO instructions are handled entirely in the generated C
//! code; these callbacks are pure IO logic.

use std::{collections::VecDeque, ffi::c_void, io::Write};

use openvm_stark_backend::p3_field::PrimeField32;
use rand::{rngs::StdRng, Rng};

use crate::arch::deferral::DeferralState;

/// IO execution state borrowed from the host `VmState<F>` for the duration of
/// one rvr call. Streams, rng, and the public-values byte slice are mutable
/// borrows; `memory_ptr` is a raw alias of VmState's main memory buffer
/// (raw because the C engine accesses it directly via pointer).
pub struct OpenVmIoState<'a, F: PrimeField32> {
    pub input_stream: &'a mut VecDeque<Vec<F>>,
    pub hint_stream: &'a mut VecDeque<F>,
    pub rng: &'a mut StdRng,
    pub memory_ptr: *mut u8,
    pub public_values: &'a mut [u8],
    pub deferrals: &'a mut Vec<DeferralState>,
}

/// Must match the C `OpenVmHostCallbacks` layout exactly.
#[repr(C)]
pub struct OpenVmHostCallbacks {
    pub ctx: *mut c_void,
    pub hint_input: extern "C" fn(*mut c_void),
    pub print_str: extern "C" fn(*mut c_void, u32, u32),
    pub hint_random: extern "C" fn(*mut c_void, u32),
    pub hint_storew: extern "C" fn(*mut c_void, u32),
    pub hint_buffer: extern "C" fn(*mut c_void, u32, u32),
    pub reveal: extern "C" fn(*mut c_void, u32, u32, u32),
    pub hint_stream_set: unsafe extern "C" fn(*mut c_void, *const u8, u32),
}

// ── Callback implementations ────────────────────────────────────────────────

/// HintInput: pop next input record from VmState's input_stream and overwrite
/// the active hint stream with `[len: u32 LE][data][padding to 4-byte align]`,
/// each byte stored as one field element.
pub extern "C" fn host_hint_input<F: PrimeField32>(ctx: *mut c_void) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    io.hint_stream.clear();
    if let Some(vec) = io.input_stream.pop_front() {
        let data_len = vec.len();
        let padded_len = (data_len + 3) & !3;
        let len_bytes = (data_len as u32).to_le_bytes();
        for &b in &len_bytes {
            io.hint_stream.push_back(F::from_u8(b));
        }
        io.hint_stream.extend(vec);
        for _ in data_len..padded_len {
            io.hint_stream.push_back(F::ZERO);
        }
    }
}

/// PrintStr: read UTF-8 from guest memory and print to stdout.
pub extern "C" fn host_print_str<F: PrimeField32>(ctx: *mut c_void, ptr: u32, len: u32) {
    let io = unsafe { &*(ctx as *const OpenVmIoState<'_, F>) };
    if len > 0 && !io.memory_ptr.is_null() {
        let slice =
            unsafe { std::slice::from_raw_parts(io.memory_ptr.add(ptr as usize), len as usize) };
        let _ = std::io::stdout().write_all(slice);
        let _ = std::io::stdout().flush();
    }
}

/// HintRandom: refill the hint stream with `num_words * 4` random bytes drawn
/// from VmState's persistent RNG (matches openvm's `Rv32HintRandomSubEx`).
pub extern "C" fn host_hint_random<F: PrimeField32>(ctx: *mut c_void, num_words: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    let nbytes = num_words as usize * 4;
    io.hint_stream.clear();
    for _ in 0..nbytes {
        io.hint_stream.push_back(F::from_u8(io.rng.random::<u8>()));
    }
}

/// HINT_STOREW: pop 4 field elements from the hint stream, cast each to a
/// byte, and write them to guest memory at `dest_addr`.
pub extern "C" fn host_hint_storew<F: PrimeField32>(ctx: *mut c_void, dest_addr: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    if io.hint_stream.len() < 4 || io.memory_ptr.is_null() {
        return;
    }
    let mut bytes = [0u8; 4];
    for byte in &mut bytes {
        *byte = io.hint_stream.pop_front().unwrap().as_canonical_u32() as u8;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), io.memory_ptr.add(dest_addr as usize), 4);
    }
}

/// HINT_BUFFER: pop `num_words * 4` field elements from the hint stream and
/// copy them as bytes into guest memory.
pub extern "C" fn host_hint_buffer<F: PrimeField32>(
    ctx: *mut c_void,
    dest_addr: u32,
    num_words: u32,
) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    let nbytes = num_words as usize * 4;
    if io.hint_stream.len() < nbytes || io.memory_ptr.is_null() {
        return;
    }
    let dst = unsafe { io.memory_ptr.add(dest_addr as usize) };
    for i in 0..nbytes {
        let byte = io.hint_stream.pop_front().unwrap().as_canonical_u32() as u8;
        unsafe { *dst.add(i) = byte };
    }
}

/// REVEAL: write public output bytes directly into the guest's `PUBLIC_VALUES_AS`
/// byte slice. Cost corrections handled in C.
pub extern "C" fn host_reveal<F: PrimeField32>(
    ctx: *mut c_void,
    src_val: u32,
    ptr: u32,
    offset: u32,
) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    let start = ptr as usize + offset as usize;
    let end = start + 4;
    assert!(
        end <= io.public_values.len(),
        "reveal out of bounds: writing bytes [{start}..{end}) but public_values size is {} (configured via SystemConfig::with_public_values)",
        io.public_values.len(),
    );
    io.public_values[start..end].copy_from_slice(&src_val.to_le_bytes());
}

/// HINT_STREAM_SET: replace the hint stream contents (used by extension phantoms).
///
/// # Safety
///
/// `ctx` must be a valid `OpenVmIoState` pointer. `data` must point to `len` bytes (or be null).
pub unsafe extern "C" fn host_hint_stream_set<F: PrimeField32>(
    ctx: *mut c_void,
    data: *const u8,
    len: u32,
) {
    let io = &mut *(ctx as *mut OpenVmIoState<'_, F>);
    io.hint_stream.clear();
    if len > 0 && !data.is_null() {
        let slice = std::slice::from_raw_parts(data, len as usize);
        for &b in slice {
            io.hint_stream.push_back(F::from_u8(b));
        }
    }
}
