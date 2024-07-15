use std::marker::PhantomData;

use afs_test_utils::engine::StarkEngine;
use p3_uni_stark::StarkGenericConfig;

use crate::{common::Commitment, dataframe::DataFrame};

#[derive(Clone, derive_new::new)]
pub struct TwoPointersProgram<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    pub pis: TwoPointersProgramPis<COMMIT_LEN>,

    _marker1: PhantomData<SC>, // This should be removed eventually
    _marker2: PhantomData<E>,  // This should be removed eventually
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>>
    TwoPointersProgram<COMMIT_LEN, SC, E>
{
    pub fn run(
        &mut self,
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

        // TODO: update this to be the commitment of the pairs
        self.pis.pairs_commit = Commitment::<COMMIT_LEN>::default();

        ret
    }

    pub fn generate_trace(&self) {}

    pub fn prove(&self, _engine: &E) {
        todo!("implement this")
    }

    pub fn verify(
        &self,
        _engine: &E,
        parent_df: DataFrame<COMMIT_LEN>,
        child_df: DataFrame<COMMIT_LEN>,
    ) {
        assert_eq!(parent_df.commit, self.pis.parent_table_commit);
        assert_eq!(child_df.commit, self.pis.child_table_commit);

        // TODO: assert that output commit is correct
    }
}

#[derive(Clone, derive_new::new)]
pub struct TwoPointersProgramPis<const COMMIT_LEN: usize> {
    pub parent_table_commit: Commitment<COMMIT_LEN>,
    pub child_table_commit: Commitment<COMMIT_LEN>,
    pub pairs_commit: Commitment<COMMIT_LEN>,
}
