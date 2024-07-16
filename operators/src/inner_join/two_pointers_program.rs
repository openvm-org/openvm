use std::marker::PhantomData;

use afs_test_utils::engine::StarkEngine;
use p3_uni_stark::StarkGenericConfig;
use parking_lot::Mutex;

use crate::{common::Commitment, dataframe::DataFrame};

#[derive(Default)]
pub struct TwoPointersProgram<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    pub pairs: Mutex<Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>>,
    pub pis: Mutex<TwoPointersProgramPis<COMMIT_LEN>>,

    _marker1: PhantomData<SC>, // This should be removed eventually
    _marker2: PhantomData<E>,  // This should be removed eventually
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>>
    TwoPointersProgram<COMMIT_LEN, SC, E>
{
    pub fn default() -> Self {
        Self {
            pairs: Mutex::new(vec![]),
            pis: Mutex::new(TwoPointersProgramPis::default()),
            _marker1: PhantomData::<SC>::default(),
            _marker2: PhantomData::<E>::default(),
        }
    }

    pub fn generate_trace(
        &self,
        parent_df: &DataFrame<COMMIT_LEN>,
        child_df: &DataFrame<COMMIT_LEN>,
    ) {
        let mut pis = self.pis.lock();
        let mut pairs = self.pairs.lock();

        pis.parent_table_commit = parent_df.commit.clone();
        pis.child_table_commit = child_df.commit.clone();
        *pairs = Self::run(parent_df, child_df);
        // TODO: update this to be the commitment of the pairs
        pis.pairs_commit = Commitment::<COMMIT_LEN>::default();
    }

    pub fn prove(&self, _engine: &E) {}

    pub fn verify(
        &self,
        _engine: &E,
        parent_df: &DataFrame<COMMIT_LEN>,
        child_df: &DataFrame<COMMIT_LEN>,
    ) {
        let pis = self.pis.lock();

        assert_eq!(parent_df.commit, pis.parent_table_commit);
        assert_eq!(child_df.commit, pis.child_table_commit);

        let _pairs = Self::run(parent_df, child_df);
        // TODO: assert that pairs.commit == self.pis.pairs_commit
    }

    fn run(
        parent_df: &DataFrame<COMMIT_LEN>,
        child_df: &DataFrame<COMMIT_LEN>,
    ) -> Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)> {
        // This is the vector of pairs of commitments of pages to join
        let mut ret = vec![];

        // Doing two-pointers to figure out which pages to join
        let mut i = 0;
        let mut j = 0;
        while i < parent_df.len() && j < child_df.len() {
            let parent_range = parent_df.get_index_range(i);
            let child_range = child_df.get_index_range(j);

            assert!(parent_range.start.len() == child_range.start.len());

            if std::cmp::max(&parent_range.start, &child_range.start)
                <= std::cmp::min(&parent_range.end, &child_range.end)
            {
                // This pair of pages need to be joined
                ret.push((
                    parent_df.page_commits[i].clone(),
                    child_df.page_commits[j].clone(),
                ));
            }

            if child_range.end <= parent_range.end {
                j += 1;
            } else {
                i += 1;
            }
        }

        ret
    }
}

#[derive(Clone, derive_new::new, Default)]
pub struct TwoPointersProgramPis<const COMMIT_LEN: usize> {
    pub parent_table_commit: Commitment<COMMIT_LEN>,
    pub child_table_commit: Commitment<COMMIT_LEN>,
    pub pairs_commit: Commitment<COMMIT_LEN>,
}
