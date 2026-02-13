//! User IO functions

use alloc::vec::Vec;
#[cfg(openvm_guest)]
use core::alloc::Layout;
use core::fmt::Write;

#[cfg(openvm_guest)]
use openvm_rv64im_guest::{hint_buffer_chunked, hint_input, hint_store_u64};
use serde::de::DeserializeOwned;

#[cfg(not(openvm_guest))]
use crate::host::{hint_input, read_n_bytes, read_u64};
use crate::serde::Deserializer;

mod read;

pub use openvm_platform::print::{print, println};

/// Read `size: u64` and then `size` bytes from the hint stream into a vector.
pub fn read_vec() -> Vec<u8> {
    hint_input();
    read_vec_by_len(read_u64() as usize)
}

/// Deserialize the next item from the next input stream into a type `T`.
pub fn read<T: DeserializeOwned>() -> T {
    let reader = read::Reader::new();
    let mut deserializer = Deserializer::new(reader);
    T::deserialize(&mut deserializer).unwrap()
}

/// Read the next 8 bytes from the hint stream into a register.
/// Because [hint_store_u64] stores a dword to memory, this function first reads to memory and then
/// loads from memory to register.
#[cfg(openvm_guest)]
#[inline(always)]
pub fn read_u64() -> u64 {
    let ptr = unsafe { alloc::alloc::alloc(Layout::from_size_align(8, 8).unwrap()) };
    let addr = ptr as u64;
    hint_store_u64!(addr);
    let result: u64;
    unsafe {
        core::arch::asm!("ld {rd}, ({rs1})", rd = out(reg) result, rs1 = in(reg) addr);
    }
    result
}

fn hint_store_dword(ptr: *mut u64) {
    #[cfg(openvm_guest)]
    hint_store_u64!(ptr);
    #[cfg(not(openvm_guest))]
    unsafe {
        *ptr = crate::host::read_u64();
    }
}

/// Load hints by key and append into the input stream.
#[allow(unused_variables)]
#[inline(always)]
pub fn hint_load_by_key(key: &[u8]) {
    #[cfg(openvm_guest)]
    openvm_rv64im_guest::hint_load_by_key(key.as_ptr(), key.len() as u64);
    #[cfg(not(openvm_guest))]
    panic!("hint_load_by_key cannot run on non-zkVM platforms");
}

/// Read the next `len` bytes from the hint stream into a vector.
pub(crate) fn read_vec_by_len(len: usize) -> Vec<u8> {
    let num_dwords = len.div_ceil(8);
    let capacity = num_dwords * 8;

    #[cfg(openvm_guest)]
    {
        // Allocate a buffer of the required length
        // We prefer that the allocator should allocate this buffer to an 8-byte boundary,
        // but we do not specify it here because `Vec<u8>` safety requires the alignment to
        // exactly equal the alignment of `u8`, which is 1. See `Vec::from_raw_parts` for more
        // details.
        //
        // Note: the bump allocator we use by default has minimum alignment of 8 bytes on RV64.
        let mut bytes = Vec::with_capacity(capacity);
        hint_buffer_chunked(bytes.as_mut_ptr(), num_dwords as usize);
        // SAFETY: We populate a `Vec<u8>` by hintstore-ing `num_dwords` 8 byte dwords. We set the
        // length to `len` and don't care about the extra `capacity - len` bytes stored.
        unsafe {
            bytes.set_len(len);
        }
        bytes
    }
    #[cfg(not(openvm_guest))]
    {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.append(&mut read_n_bytes(len));
        buffer
    }
}

/// Publish `[u8; 32]` as the first 32 bytes of the user public output.
/// In general, it is *recommended* that you reveal a single `[u8; 32]` which is
/// the hash digest of all logical outputs.
///
/// Note: this will overwrite any previous data in the first 32 bytes of the user public
/// output if it had been previously set.
pub fn reveal_bytes32(bytes: [u8; 32]) {
    for (i_u32, chunk) in bytes.chunks_exact(4).enumerate() {
        let x = u32::from_le_bytes(chunk.try_into().unwrap());
        reveal_u32(x, i_u32);
    }
}

/// Publish `x` as the `index`-th u32 output.
///
/// This is a low-level API. It is **highly recommended** that developers use [reveal_bytes32]
/// instead to publish a hash digest of program's logical outputs.
#[allow(unused_variables)]
#[inline(always)]
pub fn reveal_u32(x: u32, index: usize) {
    let byte_index = (index * 4) as u64;
    #[cfg(openvm_guest)]
    openvm_rv64im_guest::reveal!(byte_index, x, 0);
    #[cfg(all(not(openvm_guest), feature = "std"))]
    println!("reveal {} at byte location {}", x, index * 4);
}

/// Store u64 `x` to the native address `native_addr`.
#[allow(unused_variables)]
#[inline(always)]
pub fn store_u64_to_native(native_addr: u64, x: u64) {
    #[cfg(openvm_guest)]
    openvm_rv64im_guest::store_to_native!(native_addr, x);
    #[cfg(not(openvm_guest))]
    panic!("store_to_native cannot run on non-guest platforms");
}

/// A no-alloc writer to print to stdout on host machine for debugging purposes.
pub struct Writer;

impl Write for Writer {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        print(s);
        Ok(())
    }
}
