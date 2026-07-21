#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unused_variables)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
pub use openvm_custom_insn::{custom_insn_i, custom_insn_r};
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
pub mod alloc;
#[cfg(all(feature = "rust-runtime", any(openvm_intrinsics, target_os = "openvm")))]
pub mod heap;
#[cfg(all(feature = "export-libm", any(openvm_intrinsics, target_os = "openvm")))]
mod libm_extern;

pub mod memory;
pub mod print;
#[cfg(feature = "rust-runtime")]
pub mod rust_rt;

/// Size of a zkVM machine word in bytes.
/// 8 bytes (i.e. 64 bits) as the zkVM is an implementation of the rv64im ISA.
pub const WORD_SIZE_U32: u32 = u64::BITS / 8;
/// `usize` form for array lengths and indexing.
pub const WORD_SIZE: usize = WORD_SIZE_U32 as usize;

// OpenVM targets support at least 32-bit pointers, so VM indices and counts
// represented as u32 can be converted to usize without truncation.
const _: () = assert!(usize::BITS >= u32::BITS);

/// Standard IO file descriptors for use with sys_read and sys_write.
pub mod fileno {
    pub const STDIN: i32 = 0;
    pub const STDOUT: i32 = 1;
    pub const STDERR: i32 = 2;
}

/// Align address upwards.
///
/// Returns the smallest `x` with alignment `align` so that `x >= addr`.
///
/// `align` must be a power of 2.
pub const fn align_up(addr: usize, align: usize) -> usize {
    let mask = align - 1;
    (addr + mask) & !mask
}
