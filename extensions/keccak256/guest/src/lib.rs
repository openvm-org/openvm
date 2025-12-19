#![no_std]

#[cfg(target_os = "zkvm")]
extern crate alloc;
#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const KECCAK256_FUNCT3: u8 = 0b100;
pub const KECCAK256_FUNCT7: u8 = 0;

/// Native hook for keccak256 for use with `alloy-primitives` "native-keccak" feature.
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 32-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
///
/// [`keccak256`]: https://en.wikipedia.org/wiki/SHA-3
/// [`sha3`]: https://docs.rs/sha3/latest/sha3/
/// [`tiny_keccak`]: https://docs.rs/tiny-keccak/latest/tiny_keccak/
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    // SAFETY: assuming safety assumptions of the inputs, we handle all cases where `bytes` or
    // `output` are not aligned to 4 bytes.
    const MIN_ALIGN: usize = 4;
    unsafe {
        if bytes as usize % MIN_ALIGN != 0 {
            let aligned_buff = AlignedBuf::new(bytes, len, MIN_ALIGN);
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(32, MIN_ALIGN);
                __native_keccak256(aligned_buff.ptr, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 32);
            } else {
                __native_keccak256(aligned_buff.ptr, len, output);
            }
        } else {
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(32, MIN_ALIGN);
                __native_keccak256(bytes, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 32);
            } else {
                __native_keccak256(bytes, len, output);
            }
        };
    }
}

/// keccak256 intrinsic binding
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 32-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
/// - `bytes` and `output` must be 4-byte aligned.
#[cfg(target_os = "zkvm")]
#[inline(always)]
fn __native_keccak256(input: *const u8, mut len: usize, output: *mut u8) {
    let mut buffer: [u8; 200] = [0u8; 200];
    let buffer_ptr = buffer.as_mut_ptr();

    let mut ip = 0;
    let mut rate = 136;
    while len >= rate {
        unsafe {
            openvm_new_keccak256_guest::native_xorin(buffer_ptr, input.add(ip), rate);
            openvm_new_keccak256_guest::native_keccakf(buffer_ptr);
        }
        ip += rate;
        len -= rate; 
    }

    unsafe {
        openvm_new_keccak256_guest::native_xorin(buffer_ptr, input.add(ip), len)
    };

    // self.buffer.pad(self.offset, self.delim, self.rate)
    buffer[len] ^= 0x01;
    buffer[rate - 1] ^= 0x80;

    // self.fill_block()
    // which is:
    // self.keccak();
    // self.offset = 0;
    openvm_new_keccak256_guest::native_keccakf(buffer_ptr);
    
    // self.buffer.setout(&mut output[0..], 0, 32)
    unsafe {
        core::ptr::copy_nonoverlapping(buffer_ptr, output, 32)
    };
}
