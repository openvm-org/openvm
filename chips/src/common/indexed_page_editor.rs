use std::{
    collections::BTreeMap,
    iter,
    ops::{Index, IndexMut},
};

use super::page::Page;

pub struct IndexedPageEditor {
    idx_len: usize,
    data_len: usize,
    height: usize,
    idx_data_map: BTreeMap<Vec<u32>, Vec<u32>>,
}

impl IndexedPageEditor {
    pub fn from_page(page: &Page) -> Self {
        let idx_len = page.idx_len();
        let data_len = page.data_len();
        let height = page.height();

        let mut idx_data_map = BTreeMap::new();
        for row in page.iter() {
            if row.is_alloc == 1 {
                assert!(!idx_data_map.contains_key(&row.idx));
                idx_data_map.insert(row.idx.clone(), row.data.clone());
            }
        }

        Self {
            idx_len,
            data_len,
            height,
            idx_data_map,
        }
    }

    pub fn into_page(self) -> Page {
        assert!(
            self.idx_data_map.len() <= self.height,
            "Page is over capacity"
        );

        let mut page_2d_vec = vec![];
        for (idx, data) in self.idx_data_map.into_iter() {
            page_2d_vec.push(
                iter::once(1)
                    .chain(idx.iter().copied())
                    .chain(data.iter().copied())
                    .collect(),
            );
        }

        while page_2d_vec.len() < self.height {
            page_2d_vec.push(vec![0; 1 + self.idx_len + self.data_len]);
        }

        Page::from_2d_vec(&page_2d_vec, self.idx_len, self.data_len)
    }

    /// Returns true only if the page contains an allocated row with index idx
    pub fn contains(&self, idx: &[u32]) -> bool {
        self.idx_data_map.contains_key(idx)
    }

    pub fn get(&self, idx: &[u32]) -> Option<&Vec<u32>> {
        self.idx_data_map.get(idx)
    }

    /// This function inserts (idx, data) into the page
    /// It assumes that the page is not full and that the idx is not already in the page
    pub fn insert(&mut self, idx: &[u32], data: &[u32]) {
        assert!(idx.len() == self.idx_len);
        assert!(data.len() == self.data_len);

        assert!(
            self.idx_data_map
                .insert(idx.to_vec(), data.to_vec())
                .is_none(),
            "Index already exists in Page"
        );
    }

    /// This function writes (idx, data) into the page. Does an insert if idx doesn't already exist.
    /// Assumes that the page is not full
    pub fn write(&mut self, idx: &[u32], data: &[u32]) {
        assert!(idx.len() == self.idx_len);
        assert!(data.len() == self.data_len);

        self.idx_data_map.insert(idx.to_vec(), data.to_vec());
    }

    /// This function deletes the row with index idx
    /// It assumes that the page contains an allocated row with index idx
    pub fn delete(&mut self, idx: &[u32]) {
        self.idx_data_map
            .remove(idx)
            .expect("Index doesn't exist in Page");
    }
}

/// This provides indexing by an idx (Vec<u32>)
/// It assumes that the page contains an allocated row with
/// index idx and returns a reference to its data
impl Index<&[u32]> for IndexedPageEditor {
    type Output = Vec<u32>;

    fn index(&self, idx: &[u32]) -> &Self::Output {
        self.idx_data_map
            .get(idx)
            .expect("Index doesn't exist in Page")
    }
}

/// This provides mutable indexing by an idx (Vec<u32>)
/// It assumes that the page contains an allocated row with
/// index idx and returns a mutable reference to its data
impl IndexMut<&[u32]> for IndexedPageEditor {
    fn index_mut(&mut self, idx: &[u32]) -> &mut Self::Output {
        self.idx_data_map
            .get_mut(idx)
            .expect("Index doesn't exist in Page")
    }
}
