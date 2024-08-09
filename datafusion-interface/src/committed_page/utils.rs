use std::sync::Arc;

use afs_page::common::page::Page;
use datafusion::arrow::{
    array::{ArrayRef, RecordBatch, UInt32Array},
    datatypes::Schema,
};

use crate::BITS_PER_FE;

pub fn convert_to_record_batch(page: Page, schema: Schema) -> RecordBatch {
    // Get the size of each data type for each field
    let mut data_cols: Vec<Vec<_>> = vec![];
    let mut field_sizes: Vec<usize> = vec![];
    for field in schema.fields() {
        let data_type = (**field).data_type();
        let num_fe = ((data_type.size() as f64 * 8.0) / BITS_PER_FE as f64).ceil() as usize;
        field_sizes.push(num_fe);
        data_cols.push(vec![]);
    }

    for row in &page.rows {
        for (i, field_size) in field_sizes.iter().enumerate() {
            // TODO: account for field_size
            let field_data = row.data[i];
            data_cols[i].push(field_data);
        }
    }

    let array_refs: Vec<ArrayRef> = data_cols
        .into_iter()
        .map(|col| {
            // Assuming the data type is i32 for simplicity. Adjust as needed.
            let array = UInt32Array::from(col);
            Arc::new(array) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(Arc::new(schema), array_refs).unwrap()
}
