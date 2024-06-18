use alloy_primitives::{U256, U512};

use crate::{
    mock_db::MockDb,
    table::types::TableId,
    types::{Data, Index},
    utils::string_to_fixed_bytes_be_vec,
};

use super::AfsInterface;

pub enum AfsInterfaceEnum<'a> {
    Index2Data2(AfsInterface<'a, u16, u16>),
    Index4Data4(AfsInterface<'a, u32, u32>),
    Index32Data32(AfsInterface<'a, U256, U256>),
    // Add more variants as needed
}

pub fn create_interface<'a>(
    index_bytes: usize,
    data_bytes: usize,
    db_ref: &'a mut MockDb,
) -> AfsInterfaceEnum<'a> {
    match (index_bytes, data_bytes) {
        (2, 2) => AfsInterfaceEnum::Index2Data2(AfsInterface::<'a, u16, u16>::new(db_ref)),
        (4, 4) => AfsInterfaceEnum::Index4Data4(AfsInterface::<'a, u32, u32>::new(db_ref)),
        _ => panic!("Unknown variant"),
    }
}

pub fn string_to_table_id(s: String) -> TableId {
    let bytes = string_to_fixed_bytes_be_vec(s, 32);
    TableId::from_slice(bytes.as_slice())
}
