#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unused_variables)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[cfg(all(feature = "rust-runtime", target_os = "zkvm"))]
pub use openvm_custom_insn::{custom_insn_i, custom_insn_r};
#[cfg(all(feature = "rust-runtime", target_os = "zkvm"))]
pub mod heap;
#[cfg(all(feature = "export-libm", target_os = "zkvm"))]
mod libm_extern;
pub mod memory;
pub mod print;
#[cfg(feature = "rust-runtime")]
pub mod rust_rt;

/// Size of a zkVM machine word in bytes.
/// 4 bytes (i.e. 32 bits) as the zkVM is an implementation of the rv32im ISA.
pub const WORD_SIZE: usize = core::mem::size_of::<u32>();

/// Size of a zkVM memory page.
pub const PAGE_SIZE: usize = 1024;

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
