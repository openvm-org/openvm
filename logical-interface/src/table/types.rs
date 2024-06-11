use alloy_primitives::wrap_fixed_bytes;

use crate::utils::string_to_fixed_bytes_be_vec;

wrap_fixed_bytes!(pub struct TableId<32>;);

pub fn string_to_table_id(s: String) -> TableId {
    let bytes = string_to_fixed_bytes_be_vec(s, 32);
    TableId::from_slice(bytes.as_slice())
}

#[derive(Debug, Clone)]
pub struct TableMetadata {
    pub index_bytes: usize,
    pub data_bytes: usize,
}

impl TableMetadata {
    pub fn new(index_bytes: usize, data_bytes: usize) -> Self {
        Self {
            index_bytes,
            data_bytes,
        }
    }
}
