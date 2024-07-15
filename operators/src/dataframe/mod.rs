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

/// TODO: update this comment to say it doesn't have to be of fixed length
/// A DataFrame is a list of fixed length of page commitments
/// An IndexedDataFrame is a DataFrame for indexed pages
/// TODO: add comment about the start and the end
/// TODO: maybe add struct indexing
/// TODO: this should probably be changed to be a struct for every page
#[derive(Clone)]
pub struct DataFrame<const COMMIT_LEN: usize> {
    pub commit: Commitment<COMMIT_LEN>,
    pub page_commits: Vec<Commitment<COMMIT_LEN>>,
    pub df_type: DataFrameType,
}

impl<const COMMIT_LEN: usize> DataFrame<COMMIT_LEN> {
    pub fn new(page_commits: Vec<Commitment<COMMIT_LEN>>, df_type: DataFrameType) -> Self {
        Self {
            commit: Commitment::<COMMIT_LEN>::default(), // TODO: change this to be the correct commitment
            page_commits,
            df_type,
        }
    }

    pub fn empty(df_type: DataFrameType) -> Self {
        Self {
            commit: Commitment::<COMMIT_LEN>::default(), // TODO: change this to be the correct commitment
            page_commits: vec![], // TODO: change this to be the correct commitment
            df_type,
        }
    }

    pub fn len(&self) -> usize {
        self.page_commits.len()
    }

    pub fn push(&mut self, commit: Commitment<COMMIT_LEN>) {
        self.page_commits.push(commit);
        // TODO: update self.commit
    }

    pub fn get_index_range(&self, index: usize) -> IndexRange {
        match self.df_type.clone() {
            DataFrameType::Indexed(index_range) => index_range[index].clone(),
            DataFrameType::Unindexed => panic!("DataFrame is unindexed"),
        }
    }

    pub fn edit_page_commit(&mut self, index: usize, new_commit: Commitment<COMMIT_LEN>) {
        self.page_commits[index] = new_commit;
        // TODO: update self.commit
    }
}
