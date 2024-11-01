//! Hints emulation for the non-zkVM environment.

use alloc::vec::Vec;
use core::cell::RefCell;

#[thread_local]
/// Hint streams in the non-zkVM environment.
pub static HINTS: RefCell<Vec<Vec<u8>>> = RefCell::new(Vec::new());
#[thread_local]
/// Current hint stream in the non-zkVM environment.
pub static HINT_STREAM: RefCell<Vec<u8>> = RefCell::new(Vec::new());

/// Set the hints and reset the current hint stream.
pub fn set_hints(hints: Vec<Vec<u8>>) {
    HINTS.replace(hints.into_iter().rev().collect());
    HINT_STREAM.replace(Vec::new());
}

/// Read the next hint stream from the hints.
pub fn hint_input() {
    let hint = HINTS.borrow_mut().pop().expect("No hint stream available");
    HINT_STREAM.replace(hint);
}

/// Read the next `n` bytes from the hint stream.
pub fn read_n_bytes(n: usize) -> Vec<u8> {
    HINT_STREAM.borrow_mut().drain(..n).collect()
}

/// Read the next 4 bytes from the hint stream as a `u32`.
pub fn read_u32() -> u32 {
    let bytes: Vec<u8> = read_n_bytes(4);
    u32::from_le_bytes(bytes.try_into().unwrap())
}
