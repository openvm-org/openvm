use super::fixed_bytes::FixedBytesCodec;

#[test]
pub fn test_convert_to_db() {
    let fbc = FixedBytesCodec::<u32, u128, 4, 16>::new(64, 256);

    let index_fb = fbc.index_to_fixed_bytes(2);
    let index_slice: [u8; 64] = index_fb.try_into().unwrap();
    let mut index_slice_cmp = [0; 64];
    index_slice_cmp[63] = 0x02;
    assert_eq!(index_slice, index_slice_cmp);

    let data_fb = fbc.data_to_fixed_bytes(3);
    let data_slice: [u8; 256] = data_fb.try_into().unwrap();
    let mut data_slice_cmp = [0; 256];
    data_slice_cmp[255] = 0x03;
    assert_eq!(data_slice, data_slice_cmp);
}

// #[test]
// pub fn test_convert_from_db() {
//     let fbc = FixedBytesCodec::<u32, u128>::new(64, 256);
// }

// #[test]
// pub fn test_fixed_gen() {
//     let a: u32 = 5;
//     const N: usize = 10;
//     let mut b: [u8; N] = [0; N];
//     b.copy_from_slice(&a.to_be_bytes());
//     assert_eq!(b, FixedBytes::<N>::left_padding_from(a));
// }
