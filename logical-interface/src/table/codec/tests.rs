use alloy_primitives::{FixedBytes, U128, U32};

use super::fixed_bytes::FixedBytesCodec;

#[test]
pub fn test_fixed_codec() {
    let index_fb = FixedBytesCodec::<u32, u128, 64, 256>::index_to_fixed_bytes(2);
    let data_fb = FixedBytesCodec::<u32, u128, 64, 256>::data_to_fixed_bytes(2);
    assert_eq!(index_fb, FixedBytes::<64>::left_padding_from(b));
}

#[test]
pub fn test_fixed_gen() {
    let a: u32 = 5;
    let a = a.to_be_bytes();
    let a = a.as_ref();
    let b = 64;
    FixedBytes::<b>::left_padding_from(a);
}
