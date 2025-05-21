#![no_std]

#[cfg(target_os = "zkvm")]
extern crate alloc;
#[cfg(target_os = "zkvm")]
use {core::mem::MaybeUninit, zkvm::*};

/// This is custom-0 defined in RISC-V spec document
pub const OPCODE: u8 = 0x0b;
pub const KECCAK256_FUNCT3: u8 = 0b100;
pub const KECCAK256_FUNCT7: u8 = 0;

/// The keccak256 cryptographic hash function.
#[inline(always)]
pub fn keccak256(input: &[u8]) -> [u8; 32] {
    #[cfg(not(target_os = "zkvm"))]
    {
        let mut output = [0u8; 32];
        set_keccak256(input, &mut output);
        output
    }
    #[cfg(target_os = "zkvm")]
    {
        let mut output = MaybeUninit::<[u8; 32]>::uninit();
        native_keccak256(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
        unsafe { output.assume_init() }
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
unsafe fn __native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    openvm_platform::custom_insn_r!(
        opcode = OPCODE,
        funct3 = KECCAK256_FUNCT3,
        funct7 = KECCAK256_FUNCT7,
        rd = In output,
        rs1 = In bytes,
        rs2 = In len
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
extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    // SAFETY: assuming safety assumptions of the inputs, we handle all cases where `bytes` or
    // `output` are not aligned to 4 bytes.
    unsafe {
        if bytes as usize % MIN_ALIGN != 0 {
            let aligned_buff = AlignedBuf::new(bytes, len);
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(32);
                __native_keccak256(aligned_buff.ptr, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 32);
            } else {
                __native_keccak256(aligned_buff.ptr, len, output);
            }
        } else {
            if output as usize % MIN_ALIGN != 0 {
                let aligned_out = AlignedBuf::uninit(32);
                __native_keccak256(bytes, len, aligned_out.ptr);
                core::ptr::copy_nonoverlapping(aligned_out.ptr as *const u8, output, 32);
            } else {
                __native_keccak256(bytes, len, output);
            }
        };
    }
}

/// Sets `output` to the keccak256 hash of `input`.
pub fn set_keccak256(input: &[u8], output: &mut [u8; 32]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use tiny_keccak::Hasher;
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(input);
        hasher.finalize(output);
    }
    #[cfg(target_os = "zkvm")]
    native_keccak256(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
}

#[cfg(target_os = "zkvm")]
mod zkvm {
    use alloc::alloc::{alloc, dealloc, handle_alloc_error, Layout};
    use core::ptr::NonNull;

    use super::*;

    pub const MIN_ALIGN: usize = 4;

    /// Bytes aligned to 4 bytes.
    pub struct AlignedBuf {
        pub ptr: *mut u8,
        pub layout: Layout,
    }

    impl AlignedBuf {
        /// Allocate a new buffer whose start address is aligned to 4 bytes.
        pub fn uninit(len: usize) -> Self {
            let layout = Layout::from_size_align(len, MIN_ALIGN).unwrap();
            if layout.size() == 0 {
                return Self {
                    ptr: NonNull::<u32>::dangling().as_ptr() as *mut u8,
                    layout,
                };
            }
            // SAFETY: `len` is nonzero
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            AlignedBuf { ptr, layout }
        }

        /// Allocate a new buffer whose start address is aligned to 4 bytes
        /// and copy the given data into it.
        ///
        /// # Safety
        /// - `bytes` must not be null
        /// - `len` should not be zero
        ///
        /// See [alloc]. In particular `data` should not be empty.
        pub unsafe fn new(bytes: *const u8, len: usize) -> Self {
            let buf = Self::uninit(len);
            // SAFETY:
            // - src and dst are not null
            // - src and dst are allocated for size
            // - no alignment requirements on u8
            // - non-overlapping since ptr is newly allocated
            unsafe {
                core::ptr::copy_nonoverlapping(bytes, buf.ptr, len);
            }

            buf
        }
    }

    impl Drop for AlignedBuf {
        fn drop(&mut self) {
            if self.layout.size() != 0 {
                unsafe {
                    dealloc(self.ptr, self.layout);
                }
            }
        }
    }
}
