#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
use openvm::io::{hint_load_by_key, read_vec};

openvm::entry!(main);

pub fn main() {
    const KEY: &str = "key";
    hint_load_by_key(KEY.as_bytes());

    let vec = read_vec();
    // assert_eq!(vec.len(), 4);
    if vec.len() != 4 {
        openvm::process::panic();
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        if vec[i] != i as u8 {
            openvm::process::panic();
        }
    }
    let vec = read_vec();
    if vec.len() != 3 {
        openvm::process::panic();
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        if vec[i] != i as u8 {
            openvm::process::panic();
        }
    }
}
