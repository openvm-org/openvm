use crate::utils::uint_to_be_vec;

use super::fixed_bytes::FixedBytesCodec;

#[test]
pub fn test_convert_to_db() {
    let index_bytes = 4;
    let data_bytes = 8;
    let codec = FixedBytesCodec::new(index_bytes, data_bytes, 32, 256);

    let index_fb = codec.table_to_db_index_bytes(uint_to_be_vec(2, index_bytes));
    let index_slice: [u8; 32] = index_fb.try_into().unwrap();
    let mut index_slice_cmp = [0; 32];
    index_slice_cmp[31] = 0x02;
    assert_eq!(index_slice, index_slice_cmp);

    let data_fb = codec.table_to_db_data_bytes(uint_to_be_vec(3, data_bytes));
    let data_slice: [u8; 256] = data_fb.try_into().unwrap();
    let mut data_slice_cmp = [0; 256];
    data_slice_cmp[255] = 0x03;
    assert_eq!(data_slice, data_slice_cmp);
}
