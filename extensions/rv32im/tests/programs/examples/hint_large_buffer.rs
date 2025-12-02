#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::read_vec;
use openvm_rv32im_guest::MAX_HINT_BUFFER_WORDS;

openvm::entry!(main);

pub fn main() {
    let vec = read_vec();

    // Create a hint buffer larger than MAX_HINT_BUFFER_WORDS, to test chunking
    let expected_words = MAX_HINT_BUFFER_WORDS + 100;
    let expected_len = expected_words * 4;

    if vec.len() != expected_len {
        openvm::process::panic();
    }

    for i in 0..vec.len() {
        if vec[i] != (i as u8) {
            openvm::process::panic();
        }
    }
}
