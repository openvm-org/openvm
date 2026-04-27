#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unused_variables)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[cfg(openvm_intrinsics)]
pub use openvm_custom_insn::{custom_insn_i, custom_insn_r};
#[cfg(openvm_intrinsics)]
pub mod alloc;
#[cfg(all(feature = "rust-runtime", openvm_intrinsics))]
pub mod heap;
#[cfg(all(feature = "export-libm", openvm_intrinsics))]
mod libm_extern;

pub mod memory;
pub mod print;
#[cfg(feature = "rust-runtime")]
pub mod rust_rt;

/// Size of a zkVM machine word in bytes.
/// 8 bytes (i.e. 64 bits) as the zkVM is an implementation of the rv64im ISA.
pub const WORD_SIZE: usize = core::mem::size_of::<u64>();

/// Standard IO file descriptors for use with sys_read and sys_write.
pub mod fileno {
    pub const STDIN: u32 = 0;
    pub const STDOUT: u32 = 1;
    pub const STDERR: u32 = 2;
    pub const JOURNAL: u32 = 3;
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
