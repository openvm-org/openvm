use crate::{table::types::TableId, utils::string_to_u8_vec};

pub fn string_to_table_id(s: String) -> TableId {
    let bytes = string_to_u8_vec(s, 32);
    TableId::from_slice(bytes.as_slice())
}
