<<<<<<< HEAD
use crate::{table::types::TableId, utils::string_to_be_vec};

pub fn string_to_table_id(s: String) -> TableId {
    let bytes = string_to_be_vec(s, 32);
=======
use crate::{table::types::TableId, utils::string_to_u8_vec};

pub fn string_to_table_id(s: String) -> TableId {
    let bytes = string_to_u8_vec(s, 32);
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b
    TableId::from_slice(bytes.as_slice())
}
