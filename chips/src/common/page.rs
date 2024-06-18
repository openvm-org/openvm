use std::{
    collections::HashSet,
    fmt, iter,
    ops::{Index, IndexMut},
};

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use super::page_cols::PageCols;

/// A page is a collection of rows in the form
/// | is_alloc | idx | data |
///
/// It should be of a fixed height page.height(), which should be a power of 2.
/// In general pages should follow the following format:
/// - Allocated rows come first
/// - Allocated rows are sorted by idx and indices are distinct
/// - Unallocated rows are all zeros
#[derive(Clone, PartialEq)]
pub struct Page {
    pub rows: Vec<PageCols<u32>>,
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

    /// Returns a random page with the given parameters in the proper format
    /// Note that max_idx and max_data are not inclusive
    pub fn random(
        rng: &mut impl Rng,
        idx_len: usize,
        data_len: usize,
        max_idx: u32,
        max_data: u32,
        height: usize,
        rows_allocated: usize,
    ) -> Self {
        let mut gen_vec = |len: usize, max: u32| {
            (0..len)
                .map(|_| rng.gen_range(0..max))
                .collect::<Vec<u32>>()
        };

        assert!(rows_allocated <= height);
        let mut all_indices = HashSet::new();

        let mut rows = vec![];
        for _ in 0..rows_allocated {
            let mut idx;
            loop {
                idx = gen_vec(idx_len, max_idx);
                if !all_indices.contains(&idx) {
                    break;
                }
            }
            all_indices.insert(idx.clone());

            let data = gen_vec(data_len, max_data);
            rows.push(PageCols::new(1, idx, data));
        }
        rows.sort_by_key(|row| row.idx.clone());
        rows.resize(
            height,
            PageCols::new(0, vec![0; idx_len], vec![0; data_len]),
        );

        Page { rows }
    }

    pub fn idx_len(&self) -> usize {
        self.rows[0].idx.len()
    }

    pub fn data_len(&self) -> usize {
        self.rows[0].data.len()
    }

    pub fn width(&self) -> usize {
        1 + self.idx_len() + self.data_len()
    }

    pub fn height(&self) -> usize {
        self.rows.len()
    }

    /// Returns true only if the page contains an allocated row with index idx
    pub fn contains(&self, idx: &[u32]) -> bool {
        self.rows
            .iter()
            .any(|row| row.is_alloc == 1 && row.idx == idx)
    }

    /// This function inserts (idx, data) into the page
    /// It assumes that the page is not full and that the idx is not already in the page
    /// Does a linear scan
    pub fn insert(&mut self, idx: &[u32], data: &[u32]) {
        assert!(!self.contains(idx));
        assert!(
            self.rows.last().unwrap().is_alloc == 0,
            "Can't insert into a full Page"
        );
        assert!(idx.len() == self.idx_len());
        assert!(data.len() == self.data_len());

        let mut pos = 0;
        while self[pos].is_alloc == 1 && self[pos].idx < idx.to_vec() {
            pos += 1;
        }
        self.rows
            .insert(pos, PageCols::new(1, idx.to_vec(), data.to_vec()));
        self.rows.pop();
    }

    /// This function deletes the row with index idx
    /// It assumes that the page contains an allocated row with index idx
    /// Does a linear scan
    pub fn delete(&mut self, idx: &[u32]) {
        assert!(self.contains(idx));

        let mut pos = 0;
        while self[pos].idx != idx {
            pos += 1;
        }
        self.rows.remove(pos);
        self.rows.push(PageCols::new(
            0,
            vec![0; self.idx_len()],
            vec![0; self.data_len()],
        ));
    }

    pub fn get_rows_index(&self, idx: &[u32]) -> usize {
        assert!(self.contains(idx));
        self.rows.iter().position(|row| row.idx == idx).unwrap()
    }

    /// Returns a random index from an allocated row in the page
    pub fn get_random_idx(&self, rng: &mut impl Rng) -> Vec<u32> {
        let allocated_rows = self.rows.iter().filter(|row| row.is_alloc == 1).count();
        self.rows[rng.gen_range(0..allocated_rows)].idx.clone()
    }

    /// Generates the page trace
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

/// Provides indexing by a row index
impl Index<usize> for Page {
    type Output = PageCols<u32>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}

/// Provides mutable indexing by a row index
impl IndexMut<usize> for Page {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

/// This provides indexing by an idx (Vec<u32>)
/// It assumes that the page contains an allocated row with
/// index idx, does a linear search to find the first such
/// row, and returns a reference to its data
impl Index<&Vec<u32>> for Page {
    type Output = Vec<u32>;

    fn index(&self, idx: &Vec<u32>) -> &Self::Output {
        &self
            .rows
            .iter()
            .find(|row| row.is_alloc == 1 && row.idx == *idx)
            .unwrap()
            .data
    }
}

/// This provides mutable indexing by an idx (Vec<u32>)
/// It assumes that the page contains an allocated row with
/// index idx, does a linear search to find the first such
/// row, and returns a mutable reference to its data
impl IndexMut<&Vec<u32>> for Page {
    fn index_mut(&mut self, idx: &Vec<u32>) -> &mut Self::Output {
        &mut self
            .rows
            .iter_mut()
            .find(|row| row.is_alloc == 1 && row.idx == *idx)
            .unwrap()
            .data
    }
}

/// Prints the page, one row per line
impl fmt::Display for Page {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.rows {
            writeln!(f, "{:?}", row)?;
        }
        Ok(())
    }
}
