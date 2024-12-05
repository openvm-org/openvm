//! User IO functions

use alloc::vec::Vec;
#[cfg(target_os = "zkvm")]
use core::alloc::Layout;

use axvm_platform::bincode;
#[cfg(target_os = "zkvm")]
use axvm_rv32im_guest::{hint_input, hint_store_u32};
use serde::de::DeserializeOwned;

#[cfg(not(target_os = "zkvm"))]
use crate::host::{hint_input, read_n_bytes, read_u32};

/// Read `size: u32` and then `size` bytes from the hint stream into a vector.
pub fn read_vec() -> Vec<u8> {
    hint_input();
    read_vec_by_len(read_u32() as usize)
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

/// Read the next `len` bytes from the hint stream into a vector.
fn read_vec_by_len(len: usize) -> Vec<u8> {
    let num_words = (len + 3) / 4;
    let capacity = num_words * 4;

    #[cfg(target_os = "zkvm")]
    {
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
    #[cfg(not(target_os = "zkvm"))]
    {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.append(&mut read_n_bytes(len));
        buffer
    }
}

/// Publish `x` as the `index`-th u32 output.
#[allow(unused_variables)]
pub fn reveal(x: u32, index: usize) {
    let byte_index = (index * 4) as u32;
    #[cfg(target_os = "zkvm")]
    axvm_rv32im_guest::reveal!(byte_index, x, 0);
    #[cfg(all(not(target_os = "zkvm"), feature = "std"))]
    println!("reveal {} at byte location {}", x, index * 4);
}

/// Print a UTF-8 string to stdout on host machine for debugging purposes.
#[allow(unused_variables)]
pub fn print<S: AsRef<str>>(s: S) {
    #[cfg(all(not(target_os = "zkvm"), feature = "std"))]
    print!("{}", s.as_ref());
    #[cfg(target_os = "zkvm")]
    axvm_rv32im_guest::print_str_from_bytes(s.as_ref().as_bytes());
}

pub fn println<S: AsRef<str>>(s: S) {
    print(s);
    print("\n");
}
