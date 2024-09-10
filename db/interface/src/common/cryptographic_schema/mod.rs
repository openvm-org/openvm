use afs_page::common::page::Page;
use datafusion::arrow::datatypes::Schema;

use super::cryptographic_object::CryptographicObjectTrait;
use crate::utils::generate_random_alpha_string;

#[derive(Debug, Clone)]
pub struct CryptographicSchema {
    pub id: String,
    pub schema: Schema,
    pub num_idx_fields: usize,
}

impl CryptographicSchema {
    pub fn new(schema: Schema, num_idx_fields: usize) -> Self {
        let id = generate_random_alpha_string(32);
        Self {
            id,
            schema,
            num_idx_fields,
        }
    }
}

impl CryptographicObjectTrait for CryptographicSchema {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn page(&self) -> Page {
        Page::from_page_cols(vec![])
    }
}
