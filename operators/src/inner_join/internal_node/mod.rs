use std::{any::Any, sync::Arc};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    verifier::VerificationError,
};
use afs_test_utils::engine::StarkEngine;
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use parking_lot::Mutex;

use super::{page_level_join::PageLevelJoin, two_pointers_program::TwoPointersProgram};
use crate::common::{Commitment, Verifiable};

#[derive(Clone)]
pub struct InternalNode<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    children: Vec<Arc<Mutex<dyn Verifiable<SC, E>>>>,
    pairs: Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,

    /// This field should be updated only in Verifiable::verify() and
    /// is used to keep track of the verification process while verifying
    /// children
    state: InternalNodeState<COMMIT_LEN>,

    pis: InternalNodePis<COMMIT_LEN>,
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>>
    InternalNode<COMMIT_LEN, SC, E>
{
    pub fn new(
        children: Vec<Arc<Mutex<dyn Verifiable<SC, E>>>>,
        pairs: Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,
        pis: InternalNodePis<COMMIT_LEN>,
    ) -> Self {
        Self {
            children,
            pairs,
            state: InternalNodeState::<COMMIT_LEN>::default(),
            pis,
        }
    }
}

#[derive(Clone, derive_new::new)]
pub struct InternalNodePis<const COMMIT_LEN: usize> {
    init_running_df_commit: Commitment<COMMIT_LEN>,
    // TODO: maybe move this parameter to be part of the dataframe struct
    df_cur_page: u32,
    pairs_commit: Commitment<COMMIT_LEN>,
    pairs_list_index: u32,
    final_rannung_df_commit: Commitment<COMMIT_LEN>,
    parent_table_commit: Commitment<COMMIT_LEN>,
    child_table_commit: Commitment<COMMIT_LEN>,
}

impl<const COMMIT_LEN: usize> InternalNodePis<COMMIT_LEN> {
    pub fn from_slice(pis: &[u32]) -> Self {
        Self {
            init_running_df_commit: Commitment::from_slice(&pis[0..COMMIT_LEN]),
            df_cur_page: pis[COMMIT_LEN],
            pairs_commit: Commitment::from_slice(&pis[COMMIT_LEN + 1..COMMIT_LEN * 2 + 1]),
            pairs_list_index: pis[COMMIT_LEN * 2 + 1],
            final_rannung_df_commit: Commitment::from_slice(
                &pis[COMMIT_LEN * 2 + 2..COMMIT_LEN * 3 + 2],
            ),
            parent_table_commit: Commitment::from_slice(
                &pis[COMMIT_LEN * 3 + 2..COMMIT_LEN * 4 + 2],
            ),
            child_table_commit: Commitment::from_slice(
                &pis[COMMIT_LEN * 4 + 2..COMMIT_LEN * 5 + 2],
            ),
        }
    }
}

/// This struct is used in the process of verifying children
/// of InternalNode. It holds some information used in the
/// verification process.
/// TODO: maybe add a note what the role of this is in the VM context?
#[derive(Default, Clone)]
struct InternalNodeState<const COMMIT_LEN: usize> {
    /// Keeps track of the number of PageLevelJoin circuits in this subtree
    /// Starts at 0 and is updated when verifying children
    page_level_cnt: u32,

    /// Starts as None, is updated when verifying children,
    /// and should match the final dataframe commitment after
    /// verifying all children
    running_df_commit: Option<Commitment<COMMIT_LEN>>,
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    InternalNode<COMMIT_LEN, SC, E>
where
    Val<SC>: PrimeField,
    Domain<SC>: Send + Sync,
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    pub fn verify_child(
        &mut self,
        circuit: Arc<Mutex<dyn Verifiable<SC, E>>>,
        engine: &E,
    ) -> Result<(), VerificationError> {
        let mut circuit = circuit.lock();

        if let Some(tp_program) = circuit
            .as_any_mut()
            .downcast_mut::<TwoPointersProgram<COMMIT_LEN, SC, E>>()
        {
            assert_eq!(
                tp_program.pis.parent_table_commit,
                self.pis.parent_table_commit
            );
            assert_eq!(
                tp_program.pis.child_table_commit,
                self.pis.child_table_commit
            );
            assert_eq!(tp_program.pis.pairs_commit, self.pis.pairs_commit);

            // Verifiable::<SC>::verify(tp_program)?;
            tp_program.verify(&engine)?;
        } else if let Some(page_level_circuit) =
            circuit
                .as_any_mut()
                .downcast_mut::<PageLevelJoin<COMMIT_LEN, SC, E>>()
        {
            assert_eq!(
                *self.state.running_df_commit.as_ref().unwrap(),
                page_level_circuit.pis.init_running_df_commit
            );

            assert_eq!(
                page_level_circuit.pis.df_cur_page,
                self.pis.df_cur_page + self.state.page_level_cnt
            );

            assert_eq!(
                page_level_circuit.pis.pairs_list_index,
                self.pis.pairs_list_index + self.state.page_level_cnt
            );

            // Ensuring that the page-level circuit has the right parent_page_commit
            assert_eq!(
                page_level_circuit.pis.parent_page_commit,
                self.pairs[(self.pis.pairs_list_index + self.state.page_level_cnt) as usize].0
            );

            // Ensuring that the page-level circuit has the right child_page_commit
            assert_eq!(
                page_level_circuit.pis.child_page_commit,
                self.pairs[(self.pis.pairs_list_index + self.state.page_level_cnt) as usize].1
            );

            // TODO: decommit self.pis.pairs_commit corresponds to self.pairs. think if this is necessary

            // TODO: I don't think this is necessary?
            assert_eq!(page_level_circuit.pis.pairs_commit, self.pis.pairs_commit);

            page_level_circuit.verify(&engine)?;

            // Updating state
            self.state.running_df_commit =
                Some(page_level_circuit.pis.final_running_df_commit.clone());
            self.state.page_level_cnt += 1;
        } else if let Some(internal_node_circuit) = circuit
            .as_any_mut()
            .downcast_mut::<InternalNode<COMMIT_LEN, SC, E>>()
        {
            assert_eq!(
                *self.state.running_df_commit.as_ref().unwrap(),
                internal_node_circuit.pis.init_running_df_commit
            );

            assert_eq!(
                internal_node_circuit.pis.df_cur_page,
                self.pis.df_cur_page + self.state.page_level_cnt
            );

            assert_eq!(
                internal_node_circuit.pis.pairs_list_index,
                self.pis.pairs_list_index + self.state.page_level_cnt
            );

            assert_eq!(
                self.pis.pairs_commit,
                internal_node_circuit.pis.pairs_commit
            );

            assert_eq!(
                self.pis.parent_table_commit,
                internal_node_circuit.pis.parent_table_commit
            );

            assert_eq!(
                self.pis.child_table_commit,
                internal_node_circuit.pis.child_table_commit
            );

            internal_node_circuit.verify(&engine)?;

            self.state.running_df_commit =
                Some(internal_node_circuit.pis.final_rannung_df_commit.clone());
            self.state.page_level_cnt += internal_node_circuit.state.page_level_cnt;
        } else {
            unreachable!("Circuit is not of a known type");
        }
        Ok(())
    }
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    Verifiable<SC, E> for InternalNode<COMMIT_LEN, SC, E>
where
    Val<SC>: PrimeField,
    Domain<SC>: Send + Sync,
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    fn verify(&mut self, engine: &E) -> Result<(), VerificationError> {
        self.state.running_df_commit = Some(self.pis.init_running_df_commit.clone());

        let children = self.children.clone();

        for child in children.iter() {
            self.verify_child(child.clone(), engine)?;
        }

        assert_eq!(
            *self.state.running_df_commit.as_ref().unwrap(),
            self.pis.final_rannung_df_commit
        );
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
