use std::sync::Arc;

use afs_page::common::page::Page;
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array, RecordBatch, UInt32Array},
    datatypes::{DataType, Schema},
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

/// Converts a vector of columns to rows of a Page (including the `is_alloc` column)
pub fn convert_columns_to_page_rows(
    columns: Vec<Arc<dyn Array>>,
    alloc_rows: usize,
) -> Vec<Vec<u32>> {
    let height = alloc_rows.next_power_of_two();

    // Initialize a vector to hold each row, with an extra column for `is_alloc`
    let mut rows: Vec<Vec<u32>> = vec![vec![0; columns.len() + 1]; alloc_rows];
    let zero_rows: Vec<Vec<u32>> = vec![vec![0; columns.len() + 1]; height - alloc_rows];

    // Iterate over columns and fill the rows
    for (col_idx, column) in columns.iter().enumerate() {
        // TODO: handle other data types
        let array = match column.data_type() {
            DataType::UInt32 => column.as_any().downcast_ref::<UInt32Array>().unwrap(),
            DataType::Int64 => {
                let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
                let array = array
                    .values()
                    .iter()
                    .map(|&v| v as u32)
                    .collect::<Vec<u32>>();
                &UInt32Array::from(array)
            }
            _ => panic!("Unsupported data type: {}", column.data_type()),
        };
        for (row_idx, row) in rows.iter_mut().enumerate() {
            row[0] = 1;
            row[col_idx + 1] = array.value(row_idx);
        }
    }
    rows.extend(zero_rows);
    rows
}

/// Returns the number of Schema Fields that are part of the Page's index, based on the size of the Page's
/// index columns.
pub fn get_num_idx_fields(_schema: &Schema, idx_len: usize, _bits_per_fe: usize) -> usize {
    // TODO: handle other data types
    // let num_idx_fields = schema
    //     .fields()
    //     .iter()
    //     .take(idx_len)
    //     .map(|field| {
    //         ((**field).data_type().primitive_width().unwrap() as f64 * 8.0) / bits_per_fe as f64
    //     })
    //     .sum::<f64>()
    //     .ceil() as usize;
    // num_idx_fields

    idx_len
}
