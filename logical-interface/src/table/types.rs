use alloy_primitives::wrap_fixed_bytes;
use color_eyre::eyre::Result;
use serde_derive::{Deserialize, Serialize};

wrap_fixed_bytes!(
    pub struct TableId<32>;
);

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl serde::Serialize for TableId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for TableId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let bytes: [u8; 32] = serde::de::Deserialize::deserialize(deserializer)?;
        Ok(TableId::from(bytes))
    }
}
