#![no_std]

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
use core::{cmp::min, mem::MaybeUninit};

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
/// `len` must not exceed [`KECCAK_RATE`] (136): the XORIN circuit absorbs at most one rate
/// block per instruction, so a larger length could execute but never prove. The unaligned
/// fallback sizes its staging buffers for one block and asserts the bound; the check lives on
/// that cold path rather than here so the aligned path stays free of it.
///
/// # Safety
///
/// - `buffer` must point to a buffer of at least `len` bytes.
/// - `input` must point to a buffer of at least `len` bytes.
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[no_mangle]
pub unsafe extern "C" fn native_xorin(buffer: *mut u8, input: *const u8, len: usize) {
    debug_assert!(
        len <= KECCAK_RATE,
        "native_xorin: len exceeds the XORIN circuit's maximum rate of {} bytes",
        KECCAK_RATE
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

/// XOR `len` bytes from `input` into `buffer` when the operands do not satisfy the XORIN
/// instruction's requirements (both pointers 8-byte aligned, `len` a multiple of 8).
///
/// The instruction absorbs whole aligned words only, so the bytes before `buffer` reaches
/// alignment and the bytes past its last whole word are XORed in software instead. A
/// misaligned `input` cannot be fixed up in place and is staged in an aligned buffer, but
/// only for the part the instruction handles.
///
/// Neither pointer is accessed outside `[0, len)`, so this holds to the same contract as the
/// aligned path.
///
/// # Panics
///
/// Panics if `len > KECCAK_RATE`: the staging buffers hold one rate block, so a larger length
/// fails loudly here instead of overrunning them.
///
/// # Safety
///
/// Same as [`native_xorin`]: `buffer` and `input` must be valid for `len` bytes.
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[cold]
#[inline(never)]
unsafe fn xorin_unaligned(buffer: *mut u8, input: *const u8, len: usize) {
    /// Staging buffer for a misaligned `input`, sized for the largest absorb.
    #[repr(align(8))]
    struct AlignedRate(MaybeUninit<[u8; KECCAK_RATE + MIN_ALIGN - 1]>);

    assert!(
        len <= KECCAK_RATE,
        "native_xorin: len exceeds the XORIN circuit's maximum rate of {} bytes",
        KECCAK_RATE
    );

    unsafe {
        let buffer_addr = buffer as usize;
        let input_addr = input as usize;
        if buffer_addr < input_addr + len && input_addr < buffer_addr + len {
            // XORIN reads both ranges before writing the result. Snapshot overlapping input
            // so the software prefix and suffix have the same semantics. Staging at
            // `buffer`'s misalignment preserves the operands' relative alignment, and the
            // fresh local cannot itself overlap `buffer`, so the recursive call takes the
            // non-overlapping path below and reaches the instruction without a second copy.
            let mut staged = AlignedRate(MaybeUninit::uninit());
            let staged_input = (staged.0.as_mut_ptr() as *mut u8).add(buffer_addr % MIN_ALIGN);
            core::ptr::copy_nonoverlapping(input, staged_input, len);
            xorin_unaligned(buffer, staged_input, len);
            return;
        }

        // Bring `buffer` up to alignment one byte at a time.
        let misalignment = buffer_addr % MIN_ALIGN;
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
            let bulk_input = input.add(lead);
            if (bulk_input as usize).is_multiple_of(MIN_ALIGN) {
                __native_xorin(bulk_buffer, bulk_input, bulk);
            } else {
                let mut staged = AlignedRate(MaybeUninit::uninit());
                let staged_input = staged.0.as_mut_ptr() as *mut u8;
                core::ptr::copy_nonoverlapping(bulk_input, staged_input, bulk);
                __native_xorin(bulk_buffer, staged_input, bulk);
            }
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
