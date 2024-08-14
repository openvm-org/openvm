use std::sync::Arc;

use afs_page::common::page::Page;
use datafusion::arrow::{
    array::{ArrayRef, RecordBatch, UInt32Array},
    datatypes::Schema,
};

use crate::{BITS_PER_FE, NUM_IDX_COLS};

pub fn convert_to_record_batch(page: Page, schema: Schema) -> RecordBatch {
    // Get the size of each data type for each field
    let field_sizes: Vec<usize> = schema
        .fields()
        .iter()
        .map(|field| {
            let data_type = (**field).data_type();
            ((data_type.size() as f64 * 8.0) / BITS_PER_FE as f64).ceil() as usize
        })
        .collect();
    let mut idx_cols = vec![vec![]; NUM_IDX_COLS];
    let mut data_cols = vec![vec![]; field_sizes.len() - NUM_IDX_COLS];

    for row in &page.rows {
        for (i, _field_size) in field_sizes.iter().enumerate() {
            // TODO: account for field_size
            if i < NUM_IDX_COLS {
                idx_cols[i].push(row.idx[i]);
            } else {
                data_cols[i - NUM_IDX_COLS].push(row.data[i - NUM_IDX_COLS]);
            }
        }
    }

    // TODO: support other data types
    let mut array_refs: Vec<ArrayRef> = idx_cols
        .into_iter()
        .map(|col| {
            let array = UInt32Array::from(col);
            Arc::new(array) as ArrayRef
        })
        .collect();

    array_refs.extend(data_cols.into_iter().map(|col| {
        let array = UInt32Array::from(col);
        Arc::new(array) as ArrayRef
    }));

    RecordBatch::try_new(Arc::new(schema), array_refs).unwrap()
}
