use crate::mock_db::utils::string_to_fixed_bytes_be;
use std::error::Error;

use super::Table;

#[test]
pub fn test_create_new_table() {
    let table =
        Table::<U32, U64, 32, 1024>::new_from_vecs(vec![1, 2, 3, 4], vec![10, 20, 30, 40], 32, 256);
    assert_eq!(table.read(1), Some(&10));
    assert_eq!(table.read(2), Some(&20));
    assert_eq!(table.read(3), Some(&30));
    assert_eq!(table.read(4), Some(&40));
}

// #[test]
// pub fn test_creat_new_table_from_file() {
//     let input_file = "tests/data/test_input_file0.afi";
//     let table = Table::<FixedBytes<32>, FixedBytes<1024>>::new_from_file(input_file.to_string());
//     let index0 = FixedBytes::<32>(string_to_fixed_bytes_be::<32>("19000050".to_string()));
//     let data0 = FixedBytes::<1024>(string_to_fixed_bytes_be::<1024>(
//         "0x69963768F8407dE501029680dE46945F838Fc98B".to_string(),
//     ));
//     assert_eq!(table.read(index0), Some(&data0));
// }
