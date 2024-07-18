use afs_chips::common::page::Page;
use serde::Serialize;

use crate::common::{hash_struct, Commitment};

#[derive(Clone, Serialize, derive_new::new)]
pub struct IndexRange {
    pub start: Vec<u32>,
    pub end: Vec<u32>,
}

#[derive(Clone, Serialize, derive_new::new)]
pub enum DataFrameType {
    Indexed(Vec<IndexRange>),
    Unindexed,
}

/// A DataFrame is a list of page commitments
/// A DataFrame with type Indexed is a DataFrame for indexed pages:
/// alongside the page commitments, it stores the start and end indices
/// for the pages
#[derive(Clone, Serialize)]
pub struct DataFrame<const COMMIT_LEN: usize> {
    pub page_commits: Vec<Commitment<COMMIT_LEN>>,
    pub df_type: DataFrameType,
}

impl<const COMMIT_LEN: usize> DataFrame<COMMIT_LEN> {
    pub fn new(page_commits: Vec<Commitment<COMMIT_LEN>>, df_type: DataFrameType) -> Self {
        Self {
            page_commits,
            df_type,
        }
    }

    pub fn empty_unindexed() -> Self {
        Self {
            page_commits: vec![],
            df_type: DataFrameType::Unindexed,
        }
    }

    pub fn empty_indexed() -> Self {
        Self {
            page_commits: vec![],
            df_type: DataFrameType::Indexed(vec![]),
        }
    }

    pub fn from_pages_with_ranges(pages: Vec<&Page>, ranges: &Vec<IndexRange>) -> Self {
        let page_commits = pages.iter().map(|p| hash_struct(p)).collect();
        Self::new(page_commits, DataFrameType::Indexed(ranges.clone()))
    }

    pub fn from_indexed_pages(pages: Vec<&Page>) -> Self {
        let page_commits = pages.iter().map(|p| hash_struct(p)).collect();
        let ranges = pages
            .iter()
            .map(|p| {
                let (start, end) = p.get_index_range();
                IndexRange { start, end }
            })
            .collect();

        Self::new(page_commits, DataFrameType::Indexed(ranges))
    }

    pub fn len(&self) -> usize {
        self.page_commits.len()
    }

    pub fn push_indexed_page(&mut self, commit: Commitment<COMMIT_LEN>, range: IndexRange) {
        if let DataFrameType::Indexed(ref mut ranges) = self.df_type {
            self.page_commits.push(commit);
            ranges.push(range);
        } else {
            panic!("DataFrame is unindexed");
        }
    }

    pub fn push_unindexed_page(&mut self, commit: Commitment<COMMIT_LEN>) {
        if let DataFrameType::Indexed(_) = self.df_type {
            panic!("DataFrame is indexed");
        } else {
            self.page_commits.push(commit);
        }
    }

    pub fn get_index_range(&self, index: usize) -> IndexRange {
        match self.df_type.clone() {
            DataFrameType::Indexed(index_range) => index_range[index].clone(),
            DataFrameType::Unindexed => panic!("DataFrame is unindexed"),
        }
    }

    pub fn edit_page_commit(&mut self, index: usize, new_commit: Commitment<COMMIT_LEN>) {
        self.page_commits[index] = new_commit;
    }
}
