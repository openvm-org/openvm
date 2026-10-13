#![no_std]

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
use core::{cmp::min, mem::MaybeUninit, ptr::copy_nonoverlapping};

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
use openvm_platform::alloc::AlignedBuf;

pub const OPCODE: u8 = 0x0b;
pub const KECCAKF_FUNCT3: u8 = 0b100;
pub const KECCAKF_FUNCT7: u8 = 0;
pub const XORIN_FUNCT3: u8 = 0b100;
pub const XORIN_FUNCT7: u8 = 1;

pub const KECCAK_WIDTH_BYTES: usize = 200;
pub const KECCAK_RATE: usize = 136;
pub const KECCAK_OUTPUT_SIZE: usize = 32;
pub const MIN_ALIGN: usize = 8;

/// XOR `len` bytes from `input` into `buffer` using the native XORIN instruction.
///
/// `len` must not exceed [`KECCAK_RATE`] (136). Overlapping operands use the values of both
/// ranges before any write. Neither range needs padding or alignment.
///
/// The rate bound is checked in debug builds and on the unaligned path. An oversized aligned
/// call in a release build is unsupported and may fail during execution or proving.
///
/// # Safety
///
/// - `buffer` must be valid for reading and writing `len` initialized bytes.
/// - `input` must be valid for reading `len` initialized bytes.
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[no_mangle]
pub unsafe extern "C" fn native_xorin(buffer: *mut u8, input: *const u8, len: usize) {
    debug_assert!(
        len <= KECCAK_RATE,
        "native_xorin: len exceeds the XORIN circuit's maximum rate of {KECCAK_RATE} bytes"
    );
    if len == 0 {
        return;
    }
    unsafe {
        if (buffer as usize).is_multiple_of(MIN_ALIGN)
            && (input as usize).is_multiple_of(MIN_ALIGN)
            && len.is_multiple_of(MIN_ALIGN)
        {
            __native_xorin(buffer, input, len);
        } else {
            xorin_unaligned(buffer, input, len);
        }
    }
}

/// XOR partial words in software and absorb the aligned middle with XORIN.
/// A single stack buffer holds either an overlapping input snapshot or a misaligned middle.
///
/// # Safety
///
/// Same as [`native_xorin`]: `buffer` and `input` must be valid for `len` bytes.
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[cold]
#[inline(never)]
unsafe fn xorin_unaligned(buffer: *mut u8, mut input: *const u8, len: usize) {
    // Extra bytes let a full-rate snapshot start at any offset within an aligned word.
    #[repr(align(8))]
    struct AlignedRate(MaybeUninit<[u8; KECCAK_RATE + MIN_ALIGN - 1]>);

    assert!(
        len <= KECCAK_RATE,
        "native_xorin: len exceeds the XORIN circuit's maximum rate of {KECCAK_RATE} bytes"
    );

    unsafe {
        let buffer_addr = buffer as usize;
        let misalignment = buffer_addr % MIN_ALIGN;
        let mut staged = AlignedRate(MaybeUninit::uninit());
        let staged_ptr = staged.0.as_mut_ptr().cast::<u8>();
        if buffer_addr.abs_diff(input as usize) < len {
            // Snapshot before the software prefix can overwrite input. Matching the buffer's
            // misalignment also aligns the snapshot's middle, so it needs no further copy.
            let snapshot = staged_ptr.add(misalignment);
            copy_nonoverlapping(input, snapshot, len);
            input = snapshot;
        }

        // Bring `buffer` up to alignment one byte at a time.
        let lead = if misalignment == 0 {
            0
        } else {
            min(MIN_ALIGN - misalignment, len)
        };
        xorin_bytes(buffer, input, lead);

        // Absorb the whole aligned words that remain.
        let bulk = (len - lead) & !(MIN_ALIGN - 1);
        if bulk != 0 {
            let bulk_buffer = buffer.add(lead);
            let mut bulk_input = input.add(lead);
            if !(bulk_input as usize).is_multiple_of(MIN_ALIGN) {
                // A snapshot's middle is already aligned; this source is disjoint from staged.
                copy_nonoverlapping(bulk_input, staged_ptr, bulk);
                bulk_input = staged_ptr;
            }
            __native_xorin(bulk_buffer, bulk_input, bulk);
        }

        // XOR the trailing partial word in software.
        let absorbed = lead + bulk;
        xorin_bytes(buffer.add(absorbed), input.add(absorbed), len - absorbed);
    }
}

/// XOR `len` bytes from `input` into `buffer` without using the XORIN instruction.
///
/// # Safety
///
/// `buffer` and `input` must be valid for `len` bytes.
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[inline(always)]
unsafe fn xorin_bytes(buffer: *mut u8, input: *const u8, len: usize) {
    unsafe {
        for i in 0..len {
            *buffer.add(i) ^= *input.add(i);
        }
    }
}

/// Apply the Keccak-f\[1600\] permutation to the 200-byte state buffer.
///
/// # Safety
///
/// - `buffer` must point to a buffer of at least `KECCAK_WIDTH_BYTES` (200) bytes.
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[no_mangle]
pub unsafe extern "C" fn native_keccakf(buffer: *mut u8) {
    unsafe {
        if (buffer as usize).is_multiple_of(MIN_ALIGN) {
            __native_keccakf(buffer);
        } else {
            let aligned_buffer = AlignedBuf::new(buffer, KECCAK_WIDTH_BYTES, MIN_ALIGN);
            __native_keccakf(aligned_buffer.ptr);
            core::ptr::copy_nonoverlapping(
                aligned_buffer.ptr as *const u8,
                buffer,
                KECCAK_WIDTH_BYTES,
            );
        }
    }
}

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[inline(always)]
fn __native_xorin(mut buffer: *mut u8, input: *const u8, len: usize) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = XORIN_FUNCT3,
        funct7 = XORIN_FUNCT7,
        rd = InOut buffer,
        rs1 = In input,
        rs2 = In len
    );
}

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[inline(always)]
fn __native_keccakf(mut buffer: *mut u8) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = KECCAKF_FUNCT3,
        funct7 = KECCAKF_FUNCT7,
        rd = InOut buffer,
        rs1 = Const "x0",
        rs2 = Const "x0",
    );
}
