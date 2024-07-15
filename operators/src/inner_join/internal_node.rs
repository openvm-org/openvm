// TODO: remove this
// general points:
// - verify function should not mutate the circuit in any way. It should only verify it
// - generate_trace can mutate the circuit though, so a state wrapped in a mutex might be needed
// - generating public values for the circuits is a part of generating the trace
// - generally, parent.prove() should call parent.verify() and child.prove()
// - So there are three parts: generate_trace, prove, and verify
// - eventually, to parallelize trace generation, we should get rid of the mutex and generate
//   the public values in a smartmer way

use std::sync::Arc;

use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::engine::{self, StarkEngine};
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use parking_lot::Mutex;

use super::{page_level_join::PageLevelJoin, two_pointers_program::TwoPointersProgram};
use crate::{
    common::Commitment,
    dataframe::{DataFrame, DataFrameType},
};

pub enum JoinCircuit<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    PageLevelJoin(PageLevelJoin<COMMIT_LEN, SC, E>),
    TwoPointersProgram(TwoPointersProgram<COMMIT_LEN, SC, E>),
    InternalNode(InternalNode<COMMIT_LEN, SC, E>),
}

pub struct InternalNode<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    children: Vec<Arc<JoinCircuit<COMMIT_LEN, SC, E>>>,
    pairs: Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,

    /// TODO: update this comment
    /// This field should be updated only in Verifiable::verify() and
    /// is used to keep track of the verification process while verifying
    /// children
    state: Mutex<InternalNodeState<COMMIT_LEN>>,

    pis: InternalNodePis<COMMIT_LEN>,
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
    pub fn new(
        children: Vec<Arc<JoinCircuit<COMMIT_LEN, SC, E>>>,
        pairs: Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,
        pis: InternalNodePis<COMMIT_LEN>,
    ) -> Self {
        Self {
            children,
            pairs,
            state: Mutex::new(InternalNodeState::<COMMIT_LEN>::default()),
            pis,
        }
    }

    pub fn generate_trace_internal(
        &self,
        output_df: &mut DataFrame<COMMIT_LEN>,
        pairs_list_index: &mut u32,
    ) {
        for child in self.children.iter() {
            match child.as_ref() {
                JoinCircuit::PageLevelJoin(circuit) => {
                    // TODO: note that here I pass pairs_list_index twice
                    circuit.generate_trace(output_df, *pairs_list_index);
                    *pairs_list_index += 1;
                }
                JoinCircuit::TwoPointersProgram(circuit) => circuit.generate_trace(),
                JoinCircuit::InternalNode(circuit) => {
                    circuit.generate_trace_internal(output_df, pairs_list_index)
                }
            }
        }
    }

    pub fn generate_trace(&self) {
        let mut output_df = DataFrame::empty(DataFrameType::Unindexed);
        let mut pairs_list_index = 0;

        self.generate_trace_internal(&mut output_df, &mut pairs_list_index);
    }

    pub fn prove(&self, engine: &E) {
        // TODO: generate the proof for this circuit and call verify
        for child in self.children.iter() {
            match child.as_ref() {
                JoinCircuit::PageLevelJoin(circuit) => circuit.prove(engine),
                JoinCircuit::TwoPointersProgram(circuit) => circuit.prove(engine),
                JoinCircuit::InternalNode(circuit) => circuit.prove(engine),
            }
        }
    }

    pub fn verify(&self, engine: &E) {
        let mut state = self.state.lock();
        state.running_df_commit = Some(self.pis.init_running_df_commit.clone());

        let children = self.children.clone();

        for child in children.iter() {
            self.verify_child(child.clone(), engine);
        }

        assert_eq!(
            *state.running_df_commit.as_ref().unwrap(),
            self.pis.final_rannung_df_commit
        );
    }

    fn verify_child(&self, circuit: Arc<JoinCircuit<COMMIT_LEN, SC, E>>, _engine: &E) {
        // TODO: make an enum actually to avoid downcasting

        match circuit.as_ref() {
            JoinCircuit::TwoPointersProgram(tp_program) => {
                assert_eq!(
                    tp_program.pis.parent_table_commit,
                    self.pis.parent_table_commit
                );
                assert_eq!(
                    tp_program.pis.child_table_commit,
                    self.pis.child_table_commit
                );
                assert_eq!(tp_program.pis.pairs_commit, self.pis.pairs_commit);
            }
            JoinCircuit::PageLevelJoin(page_level_circuit) => {
                let mut state = self.state.lock();
                // TODO: ideally I shouldn't have a lock around the public inputs for child, but seems like
                // that might be necessary...
                let child_pis = page_level_circuit.pis.lock();

                assert_eq!(
                    *state.running_df_commit.as_ref().unwrap(),
                    child_pis.init_running_df_commit
                );

                assert_eq!(
                    child_pis.pairs_list_index,
                    self.pis.pairs_list_index + state.page_level_cnt
                );

                // Ensuring that the page-level circuit has the right parent_page_commit
                assert_eq!(
                    child_pis.parent_page_commit,
                    self.pairs[(self.pis.pairs_list_index + state.page_level_cnt) as usize].0
                );

                // Ensuring that the page-level circuit has the right child_page_commit
                assert_eq!(
                    child_pis.child_page_commit,
                    self.pairs[(self.pis.pairs_list_index + state.page_level_cnt) as usize].1
                );

                // TODO: decommit self.pis.pairs_commit corresponds to self.pairs. think if this is necessary

                // TODO: I don't think this is necessary?
                assert_eq!(child_pis.pairs_commit, self.pis.pairs_commit);

                // Updating state
                state.running_df_commit = Some(child_pis.final_running_df_commit.clone());
                state.page_level_cnt += 1;
            }
            JoinCircuit::InternalNode(internal_node_circuit) => {
                let mut state = self.state.lock();

                assert_eq!(
                    *state.running_df_commit.as_ref().unwrap(),
                    internal_node_circuit.pis.init_running_df_commit
                );

                assert_eq!(
                    internal_node_circuit.pis.df_cur_page,
                    self.pis.df_cur_page + state.page_level_cnt
                );

                assert_eq!(
                    internal_node_circuit.pis.pairs_list_index,
                    self.pis.pairs_list_index + state.page_level_cnt
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

                state.running_df_commit =
                    Some(internal_node_circuit.pis.final_rannung_df_commit.clone());

                // TODO: I think to parallelize, I should avoid doing this
                let child_state = internal_node_circuit.state.lock();
                state.page_level_cnt += child_state.page_level_cnt;
            }
        }
    }
}
