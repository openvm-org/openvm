use std::{
    iter,
    ops::{Index, IndexMut},
};

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;

use super::page_cols::PageCols;

/// A page is a collection of rows in the form
/// | is_alloc | idx | data |
///
/// It should be of a fixed height page.len(), which should be a power of 2.
#[derive(Clone)]
pub struct Page {
    pub rows: Vec<PageCols<u32>>,
}

impl Index<usize> for Page {
    type Output = PageCols<u32>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for Page {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl Page {
    pub fn from_2d_vec(page: &[Vec<u32>], idx_len: usize, data_len: usize) -> Self {
        Self {
            rows: page
                .iter()
                .map(|row| {
                    assert!(row.len() == 1 + idx_len + data_len);
                    PageCols::from_slice(row, idx_len, data_len)
                })
                .collect(),
        }
    }

    pub fn width(&self) -> usize {
        1 + self.rows[0].idx.len() + self.rows[0].data.len()
    }

    pub fn height(&self) -> usize {
        self.rows.len()
    }

    pub fn gen_trace<F: PrimeField>(&self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            self.rows
                .iter()
                .flat_map(|row| {
                    iter::once(row.is_alloc)
                        .chain(row.idx.clone())
                        .chain(row.data.clone())
                })
                .map(F::from_canonical_u32)
                .collect(),
            self.width(),
        )
    }
}
