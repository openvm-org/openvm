#![no_main]
#![no_std]

axvm::entry!(main);

pub fn main() {
    axvm::intrinsics::io::hint_input();
    let vec = axvm::intrinsics::io::read_vec();
    if vec.iter().sum::<u8>() == 0 {
        axvm::process::panic();
    }
}
