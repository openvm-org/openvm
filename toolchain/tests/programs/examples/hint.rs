#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
use axvm::io::read_vec;

axvm::entry!(main);

pub fn main() {
    let vec = read_vec();
    // assert_eq!(vec.len(), 4);
    if vec.len() != 4 {
        axvm::process::panic();
    }
    for i in 0..4 {
        if vec[i] != i as u8 {
            axvm::process::panic();
        }
        // assert_eq!(vec[i], i as u8);
    }
}
