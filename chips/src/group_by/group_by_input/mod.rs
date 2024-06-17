use crate::{common::page::Page, is_equal_vec::IsEqualVecAir};
use std::collections::HashMap;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct GroupByAir {
    pub internal_bus: usize,
    pub output_bus: usize,

    pub is_equal_vec_air: IsEqualVecAir,

    // does not include is_allocated column
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
            is_equal_vec_air: IsEqualVecAir::new(group_by_cols.len()),
            internal_bus,
            output_bus,
        }
    }
    pub fn get_width(&self) -> usize {
        self.page_width + 3 * self.group_by_cols.len() + 5
    }

    pub fn aux_width(&self) -> usize {
        3 * self.group_by_cols.len() + 5
    }

    pub fn request(&self, page: &Page) -> Page {
        let mut grouped_page: Vec<Vec<u32>> = page
            .rows
            .iter()
            .map(|row| {
                let mut selected_row: Vec<u32> = self
                    .group_by_cols
                    .iter()
                    .map(|&col_index| row.data[col_index])
                    .collect();
                selected_row.push(row.data[self.aggregated_col]);
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
