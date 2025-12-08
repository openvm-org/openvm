// [!region imports]
use core::hint::black_box;

use hex::FromHex;
use openvm as _;
use tiny_keccak::Hasher;

// [!endregion imports]

// [!region main]
pub fn main() {
    let mut buffer = [1u8; 136];
    let input = [2u8; 136];
    let output = [3u8; 136];
    let len = 136;

    openvm_new_keccak256_guest::native_xorin(buffer.as_mut_ptr(), input.as_ptr(), len);

    assert_eq!(buffer, output, "native_xorin is incorrect");

}
// [!endregion main]
