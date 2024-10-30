#![no_main]
#![no_std]

axvm::entry!(main);

pub fn main() {
    let x = axvm::intrinsics::io::read_byte();
    if x == 0 {
        loop {}
    }
    let vec = axvm::intrinsics::io::read_size_and_vec();
    if vec.iter().sum::<u8>() == 0 {
        loop {}
    }
}
