// [!region imports]
use core::hint::black_box;

use hex::FromHex;
use openvm as _;
use openvm_keccak256::keccak256;
use openvm::io::reveal_u32;
// [!endregion imports]

// [!region main]
pub fn main() {
    let mut buffer = [1u8; 200];
    let input = [2u8; 136];
    let output = [3u8; 136];
    let len: usize = 136;
    openvm_new_keccak256_guest::native_xorin(buffer.as_mut_ptr(), input.as_ptr(), len);
    assert_eq!(buffer[..136], output);
    openvm_new_keccak256_guest::native_keccakf(buffer.as_mut_ptr());
}
// [!endregion main]
