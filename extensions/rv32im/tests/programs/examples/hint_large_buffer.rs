//! Test that reading a buffer larger than MAX_HINT_BUFFER_WORDS works correctly.
//! This proves that hint_buffer_chunked properly splits the read into multiple instructions.

#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::read_vec;
use openvm_rv32im_guest::MAX_HINT_BUFFER_WORDS;

openvm::entry!(main);

pub fn main() {
    let vec = read_vec();

    // We expect a buffer larger than MAX_HINT_BUFFER_WORDS * 4 bytes
    // This proves chunking is working because a single HINT_BUFFER instruction
    // would be rejected if num_words > MAX_HINT_BUFFER_WORDS
    let expected_words = MAX_HINT_BUFFER_WORDS + 100;
    let expected_len = expected_words * 4;

    if vec.len() != expected_len {
        openvm::process::panic();
    }

    // Verify some data integrity at boundaries
    // Check first word
    for i in 0..4 {
        if vec[i] != (i as u8) {
            openvm::process::panic();
        }
    }

    // Check at the MAX_HINT_BUFFER_WORDS boundary (where chunk split happens)
    let boundary_offset = MAX_HINT_BUFFER_WORDS * 4;
    for i in 0..4 {
        let expected = ((boundary_offset + i) % 256) as u8;
        if vec[boundary_offset + i] != expected {
            openvm::process::panic();
        }
    }

    // Check last word
    let last_offset = expected_len - 4;
    for i in 0..4 {
        let expected = ((last_offset + i) % 256) as u8;
        if vec[last_offset + i] != expected {
            openvm::process::panic();
        }
    }
}

