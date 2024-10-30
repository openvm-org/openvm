#![no_main]
#![no_std]

axvm::entry!(main);

pub fn main() {
    axvm::intrinsics::io::hint_input();
    let x = axvm::intrinsics::io::read_u32();
    if x == 0 {
        loop {}
    }
    let vec = axvm::intrinsics::io::read_size_and_vec();
    if vec.iter().sum::<u8>() == 0 {
        loop {}
    }
}
