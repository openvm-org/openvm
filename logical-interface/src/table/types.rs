use alloy_primitives::wrap_fixed_bytes;

wrap_fixed_bytes!(pub struct TableId<32>;);

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
