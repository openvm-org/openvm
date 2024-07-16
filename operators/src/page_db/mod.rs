use std::collections::HashMap;

use afs_chips::common::page::Page;

use crate::common::Commitment;

pub struct PageDb<const COMMIT_LEN: usize> {
    map: HashMap<Commitment<COMMIT_LEN>, Page>,
}

impl<const COMMIT_LEN: usize> PageDb<COMMIT_LEN> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn get_page(&self, commit: &Commitment<COMMIT_LEN>) -> Option<Page> {
        self.map.get(commit).cloned()
    }
}
