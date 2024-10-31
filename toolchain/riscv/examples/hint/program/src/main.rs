#![no_main]
#![no_std]
use axvm::intrinsics::io::read_vec;

axvm::entry!(main);

pub fn main() {
    let vec = read_vec();
    assert_eq!(vec.len(), 4);
    for i in 0..4 {
        assert_eq!(vec[i], i as u8);
    }
}
