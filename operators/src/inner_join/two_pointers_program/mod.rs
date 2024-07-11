use std::{any::Any, marker::PhantomData};

use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::engine::StarkEngine;
use p3_uni_stark::StarkGenericConfig;

use crate::{
    common::{Commitment, Verifiable},
    dataframe::DataFrame,
};

#[derive(Clone, derive_new::new)]
pub struct TwoPointersProgram<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    pub parent_table_df: DataFrame<COMMIT_LEN>,
    pub child_table_df: DataFrame<COMMIT_LEN>,

    pub pis: TwoPointersProgramPis<COMMIT_LEN>,

    _marker1: PhantomData<SC>, // This should be removed eventually
    _marker2: PhantomData<E>,  // This should be removed eventually
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>>
    TwoPointersProgram<COMMIT_LEN, SC, E>
{
    pub fn run(&mut self) -> Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)> {
        // This is the vector of pairs of commitments of pages to join
        let mut ret = vec![];

        // Doing two-pointers to figure out which pages to join
        let mut i = 0;
        let mut j = 0;
        while i < self.parent_table_df.len() && j < self.child_table_df.len() {
            let parent_range = self.parent_table_df.get_index_range(i);
            let child_range = self.child_table_df.get_index_range(j);

            assert!(parent_range.start.len() == child_range.start.len());

            if std::cmp::max(&parent_range.start, &child_range.start)
                <= std::cmp::min(&parent_range.end, &child_range.end)
            {
                // This pair of pages need to be joined
                ret.push((
                    self.parent_table_df.page_commits[i].clone(),
                    self.child_table_df.page_commits[j].clone(),
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
}

#[derive(Clone, derive_new::new)]
pub struct TwoPointersProgramPis<const COMMIT_LEN: usize> {
    pub parent_table_commit: Commitment<COMMIT_LEN>,
    pub child_table_commit: Commitment<COMMIT_LEN>,
    pub pairs_commit: Commitment<COMMIT_LEN>,
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    Verifiable<SC, E> for TwoPointersProgram<COMMIT_LEN, SC, E>
{
    fn verify(&mut self, _engine: &E) -> Result<(), VerificationError> {
        assert!(self.parent_table_df.commit == self.pis.parent_table_commit);
        assert!(self.child_table_df.commit == self.pis.child_table_commit);

        // TODO: assert that output commit is correct

        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
