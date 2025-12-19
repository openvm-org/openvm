#![no_std]

#[cfg(target_os = "zkvm")]
extern crate alloc;
#[cfg(target_os = "zkvm")]
use openvm_platform::alloc::AlignedBuf;

pub const OPCODE: u8 = 0x0b;
pub const KECCAKF_FUNCT3: u8 = 0b100;
pub const KECCAKF_FUNCT7: u8 = 0;
pub const XORIN_FUNCT3: u8 = 0b100;
pub const XORIN_FUNCT7: u8 = 1;
pub const KECCAK_WIDTH_BYTES: usize = 200;

#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn native_xorin(mut buffer: *mut u8, input: *const u8, len: usize) {
    const MIN_ALIGN: usize = 4;
    unsafe {
        if buffer as usize % MIN_ALIGN != 0 {
            let aligned_buffer = AlignedBuf::new(buffer, len, MIN_ALIGN);
            if input as usize % MIN_ALIGN != 0 {
                let aligned_input = AlignedBuf::new(input, len, MIN_ALIGN);
                __native_xorin(aligned_buffer.ptr, aligned_input.ptr, len);
                core::ptr::copy_nonoverlapping(aligned_buffer.ptr as *const u8, buffer, len);
            } else {
                __native_xorin(aligned_buffer.ptr, input, len);
                core::ptr::copy_nonoverlapping(aligned_buffer.ptr as *const u8, buffer, len);
            }
        } else {
            if input as usize % MIN_ALIGN != 0 {
                let aligned_input = AlignedBuf::new(input, len, MIN_ALIGN);
                __native_xorin(buffer, aligned_input.ptr, len);
            } else {
                __native_xorin(buffer, input, len);
            }
        };
    }
}

#[cfg(target_os = "zkvm")]
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

#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
pub extern "C" fn native_keccakf(mut buffer: *mut u8) {
    const MIN_ALIGN: usize = 4;
    unsafe {
        if buffer as usize % MIN_ALIGN != 0 {
            let aligned_buffer = AlignedBuf::new(buffer, KECCAK_WIDTH_BYTES, MIN_ALIGN);
            __native_keccakf(aligned_buffer.ptr);
            core::ptr::copy_nonoverlapping(
                aligned_buffer.ptr as *const u8,
                buffer,
                KECCAK_WIDTH_BYTES,
            );
        } else {
            __native_keccakf(buffer);
        };
    }
}

#[cfg(target_os = "zkvm")]
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

    if len % 4 != 0 {
        let mut adjusted_len = len;
        adjusted_len += 4 - (len % 4);

        let mut new_input: [u8; 136] = [0; 136];
        for i in 0..len {
            new_input[i] = unsafe {
                *input.add(i)
            };
        }
        unsafe {
            openvm_new_keccak256_guest::native_xorin(buffer_ptr, new_input.as_ptr(), adjusted_len)
        };
    } else {
        unsafe {
            openvm_new_keccak256_guest::native_xorin(buffer_ptr, input.add(ip), len)
        };
    }


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
