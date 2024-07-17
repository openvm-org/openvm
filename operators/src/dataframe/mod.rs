use crate::common::Commitment;

#[derive(Clone)]
pub struct IndexRange {
    pub start: Vec<u32>,
    pub end: Vec<u32>,
}

#[derive(Clone, derive_new::new)]
pub enum DataFrameType {
    Indexed(Vec<IndexRange>),
    Unindexed,
}

/// A DataFrame is a list of page commitments
/// A DataFrame with type Indexed is a DataFrame for indexed pages:
/// alongside the page commitments, it stores the start and end indices
/// for the pages
#[derive(Clone)]
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

    pub fn empty(df_type: DataFrameType) -> Self {
        let empty_vec: Vec<Commitment<COMMIT_LEN>> = vec![];

        Self {
            page_commits: empty_vec,
            df_type,
        }
    }

    pub fn len(&self) -> usize {
        self.page_commits.len()
    }

    pub fn push(&mut self, commit: Commitment<COMMIT_LEN>) {
        self.page_commits.push(commit);
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
