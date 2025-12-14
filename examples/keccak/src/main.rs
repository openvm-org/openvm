// [!region imports]
use core::hint::black_box;

use hex::FromHex;
use openvm as _;
use openvm::io::reveal_u32;
// [!endregion imports]

// [!region main]
pub fn main() {
    let mut buffer = [1u8; 200];
    let input = [2u8; 136];
    let output = [3u8; 136];

    let len: usize = 136;
    println!("buffer pointer {:?}", buffer.as_mut_ptr());
    openvm_new_keccak256_guest::native_xorin(buffer.as_mut_ptr(), input.as_ptr(), len);
    assert_eq!(buffer[..136], output);
    println!("buffer pointer {:?}", buffer.as_mut_ptr());
    openvm_new_keccak256_guest::native_keccakf(buffer.as_mut_ptr());
    println!("{:?}", buffer);
}
// [!endregion main]
