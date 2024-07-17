use crate::{common::page::Page, is_equal_vec::IsEqualVecAir};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

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

    /// Includes is_allocated column, so `data_len + 1 == page_width`
    pub page_width: usize,
    pub group_by_cols: Vec<usize>,
    pub aggregated_col: usize,

    /// Whether the input page is already sorted by the group-by columns
    pub sorted: bool,

    /// The operation to perform on the aggregated column
    pub op: GroupByOperation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GroupByOperation {
    Sum,
    Product,
}

impl GroupByAir {
    pub fn new(
        page_width: usize,
        group_by_cols: Vec<usize>,
        aggregated_col: usize,
        internal_bus: usize,
        output_bus: usize,
        sorted: bool,
        op: GroupByOperation,
    ) -> Self {
        Self {
            page_width,
            // has +1 to check equality on is_alloc column
            is_equal_vec_air: IsEqualVecAir::new(group_by_cols.len() + 1),
            group_by_cols,
            aggregated_col,
            sorted,
            op,
            internal_bus,
            output_bus,
        }
    }

    /// Width of entire trace
    pub fn get_width(&self) -> usize {
        if !self.sorted {
            self.page_width + 3 * self.group_by_cols.len() + 7
        } else {
            3 * self.group_by_cols.len() + 7
        }
    }

    /// Width of auxilliary trace, i.e. all non-input-page columns
    pub fn aux_width(&self) -> usize {
        if !self.sorted {
            3 * self.group_by_cols.len() + 7
        } else {
            2 * self.group_by_cols.len() + 5
        }
    }

    pub fn select_and_sort(&self, page: &Page) -> Vec<Vec<u32>> {
        if self.sorted {
            page.iter()
                .filter(|row| row.is_alloc == 1)
                .map(|row| row.data.clone())
                .collect()
        } else {
            let mut grouped_page: Vec<Vec<u32>> = page
                .iter()
                .filter(|row| row.is_alloc == 1)
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
            grouped_page
        }
    }

    /// This pure function computes the answer to the group-by operation
    pub fn request(&self, page: &Page) -> (Page, Page) {
        let grouped_page: Vec<Vec<u32>> = self.select_and_sort(page);

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

        let mut new_grouped_page: Vec<Vec<u32>> = grouped_page
            .iter()
            .map(|row| {
                let mut new_row = vec![1];
                new_row.append(&mut row.clone());
                new_row
            })
            .collect();
        new_grouped_page.resize(page.height(), vec![0; row_width]);
        (
            Page::from_2d_vec(&grouped_sums, idx_len, 1),
            Page::from_2d_vec(&new_grouped_page, idx_len, 1),
        )
    }
}

impl FromStr for GroupByOperation {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s_upper = s.to_uppercase();
        match s_upper.as_str() {
            "SUM" => Ok(GroupByOperation::Sum),
            "PRODUCT" => Ok(GroupByOperation::Product),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid operand",
            )),
        }
    }
}
