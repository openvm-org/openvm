//! User IO functions

use alloc::vec::Vec;
use core::alloc::Layout;

use axvm_platform::bincode;
use serde::de::DeserializeOwned;

#[cfg(target_os = "zkvm")]
use crate::{hint_store_u32, intrinsics::hint_input};

/// Read `size: u32` and then `size` bytes from the hint stream into a vector.
pub fn read_vec() -> Vec<u8> {
    #[cfg(not(target_os = "zkvm"))]
    {
        todo!()
    }
    #[cfg(target_os = "zkvm")]
    {
        hint_input();
        read_vec_by_len(read_u32() as usize)
    }
}

/// Read the next vec and deserialize it into a type `T`.
pub fn read<T: DeserializeOwned>() -> T {
    let serialized_data = read_vec();
    bincode::serde::decode_from_slice(&serialized_data[..], bincode::config::standard())
        .expect("Deserialize from bytes failed")
        .0
}

/// Read the next 4 bytes from the hint stream into a register.
/// Because [hint_store_u32] stores a word to memory, this function first reads to memory and then
/// loads from memory to register.
#[cfg(target_os = "zkvm")]
#[inline(always)]
#[allow(asm_sub_register)]
pub fn read_u32() -> u32 {
    let ptr = unsafe { alloc::alloc::alloc(Layout::from_size_align(4, 4).unwrap()) };
    let addr = ptr as u32;
    hint_store_u32!(addr, 0);
    let result: u32;
    unsafe {
        core::arch::asm!("lw {rd}, ({rs1})", rd = out(reg) result, rs1 = in(reg) addr);
    }
    result
}

#[cfg(target_os = "zkvm")]
/// Read the next `len` bytes from the hint stream into a vector.
fn read_vec_by_len(len: usize) -> Vec<u8> {
    let num_words = (len + 3) / 4;
    let capacity = num_words * 4;
    // Allocate a buffer of the required length that is 4 byte aligned
    // Note: this expect message doesn't matter until our panic handler actually cares about it
    let layout = Layout::from_size_align(capacity, 4).expect("vec is too large");
    // SAFETY: We populate a `Vec<u8>` by hintstore-ing `num_words` 4 byte words. We set the length to `len` and don't care about the extra `capacity - len` bytes stored.
    let ptr_start = unsafe { alloc::alloc::alloc(layout) };
    let mut ptr = ptr_start;

    // Note: if len % 4 != 0, this will discard some last bytes
    for _ in 0..num_words {
        hint_store_u32!(ptr, 0);
        ptr = unsafe { ptr.add(4) };
    }
    unsafe { Vec::from_raw_parts(ptr_start, len, capacity) }
}
