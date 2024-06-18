use crate::{common::page::Page, is_equal_vec::IsEqualVecAir};
use std::collections::HashMap;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Main struct defining constraints and dimensions for group-by operation
///
/// Operation:
/// 1. sends columns of interest to itself, constraining equal rows to be adjacent
/// 2. completes partial operations on aggregated column
/// 3. sends the aggregated columns to MyFinalPage
pub struct GroupByAir {
    pub internal_bus: usize,
    pub output_bus: usize,

    /// Has +1 to check equality on `is_alloc` column
    pub is_equal_vec_air: IsEqualVecAir,

    /// Does not include is_allocated column, so `idx_len + 1 == page_width`
    pub page_width: usize,
    pub group_by_cols: Vec<usize>,
    pub aggregated_col: usize,
    // for now, will default to addition
    // operation: SubAir::eval,
}

impl GroupByAir {
    pub fn new(
        page_width: usize,
        group_by_cols: Vec<usize>,
        aggregated_col: usize,
        internal_bus: usize,
        output_bus: usize,
    ) -> Self {
        Self {
            page_width,
            group_by_cols: group_by_cols.clone(),
            aggregated_col,
            // has +1 to check equality on is_alloc column
            is_equal_vec_air: IsEqualVecAir::new(group_by_cols.len() + 1),
            internal_bus,
            output_bus,
        }
    }

    /// Width of entire trace
    pub fn get_width(&self) -> usize {
        self.page_width + 3 * self.group_by_cols.len() + 7
    }

    /// Width of auxilliary trace, i.e. all non-input-page columns
    pub fn aux_width(&self) -> usize {
        3 * self.group_by_cols.len() + 7
    }

    /// This pure function computes the answer to the group-by operation
    pub fn request(&self, page: &Page) -> Page {
        let mut grouped_page: Vec<Vec<u32>> = page
            .rows
            .iter()
            .filter(|row| row.is_alloc == 1)
            .map(|row| {
                let mut selected_row: Vec<u32> = self
                    .group_by_cols
                    .iter()
                    .map(|&col_index| row.idx[col_index])
                    .collect();
                selected_row.push(row.idx[self.aggregated_col]);
                selected_row
            })
            .collect();

        grouped_page.sort();

        let mut sums_by_key: HashMap<Vec<u32>, u32> = HashMap::new();
        for row in grouped_page.iter() {
            let (value, index) = row.split_last().unwrap();
            *sums_by_key.entry(index.to_vec()).or_insert(0) += value;
        }

        // Convert the hashmap back to a sorted vector for further processing
        let mut grouped_sums: Vec<Vec<u32>> = sums_by_key
            .into_iter()
            .map(|(mut key, sum)| {
                key.insert(0, 1);
                key.push(sum);
                key
            })
            .collect();
        grouped_sums.sort();

        let idx_len = self.group_by_cols.len();
        let row_width = 1 + idx_len + 1;
        grouped_sums.resize(page.height(), vec![0; row_width]);

        Page::from_2d_vec(&grouped_sums, idx_len, 1)
    }
}
