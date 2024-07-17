use std::{cell::RefCell, sync::Arc};

use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::engine::StarkEngine;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use super::{page_level_join::PageLevelJoin, two_pointers_program::TwoPointersProgram};
use crate::{common::Commitment, dataframe::DataFrame, page_db::PageDb};

pub enum JoinCircuit<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    PageLevelJoin(PageLevelJoin<COMMIT_LEN, SC, E>),
    TwoPointersProgram(TwoPointersProgram<COMMIT_LEN, SC, E>),
    InternalNode(InternalNode<COMMIT_LEN, SC, E>),
}

pub struct InternalNode<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    children: Vec<Arc<JoinCircuit<COMMIT_LEN, SC, E>>>,
    state: RefCell<InternalNodeState<COMMIT_LEN>>,
    pis: RefCell<InternalNodePis<COMMIT_LEN>>,
}

#[derive(Clone, derive_new::new, Default)]
pub struct InternalNodePis<const COMMIT_LEN: usize> {
    init_running_df_commit: Commitment<COMMIT_LEN>,
    pairs_commit: Commitment<COMMIT_LEN>,
    pairs_list_index: u32,
    final_rannung_df_commit: Commitment<COMMIT_LEN>,
    parent_table_commit: Commitment<COMMIT_LEN>,
    child_table_commit: Commitment<COMMIT_LEN>,
}

/// This struct is used during the verification process of this
/// node. After the verification process of this node is done,
/// the state is queried by the parent to get some information
/// about the node.
#[derive(Default, Clone)]
struct InternalNodeState<const COMMIT_LEN: usize> {
    /// Keeps track of the number of PageLevelJoin circuits in this subtree
    /// Starts at 0 and is updated when verifying children
    page_level_cnt: u32,

    /// Starts as the initial dataframe commitment, is updated when
    /// verifying children, and should match the final dataframe
    /// commitment after verifying all children
    running_df_commit: Option<Commitment<COMMIT_LEN>>,
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    InternalNode<COMMIT_LEN, SC, E>
{
    pub fn new(children: Vec<Arc<JoinCircuit<COMMIT_LEN, SC, E>>>) -> Self {
        Self {
            children,
            state: RefCell::new(InternalNodeState::<COMMIT_LEN>::default()),
            pis: RefCell::new(InternalNodePis::<COMMIT_LEN>::default()),
        }
    }

    pub fn generate_trace_for_tree(
        &self,
        page_db: Arc<PageDb<COMMIT_LEN>>,
        output_df: &mut DataFrame<COMMIT_LEN>,
        pairs_commit: &Commitment<COMMIT_LEN>,
        pairs_list_index: &mut u32,
        parent_df: &DataFrame<COMMIT_LEN>,
        child_df: &DataFrame<COMMIT_LEN>,
        engine: &E,
    ) where
        Val<SC>: PrimeField,
    {
        let mut pis = self.pis.borrow_mut();

        pis.init_running_df_commit = output_df.commit.clone();
        pis.pairs_commit = pairs_commit.clone();
        pis.pairs_list_index = *pairs_list_index;
        pis.parent_table_commit = parent_df.commit.clone();
        pis.child_table_commit = child_df.commit.clone();

        for child in self.children.iter() {
            match child.as_ref() {
                JoinCircuit::PageLevelJoin(circuit) => {
                    circuit.generate_trace(page_db.clone(), output_df, pairs_list_index, engine);
                }
                JoinCircuit::TwoPointersProgram(circuit) => {
                    circuit.generate_trace(parent_df, child_df)
                }
                JoinCircuit::InternalNode(circuit) => circuit.generate_trace_for_tree(
                    page_db.clone(),
                    output_df,
                    pairs_commit,
                    pairs_list_index,
                    parent_df,
                    child_df,
                    engine,
                ),
            }
        }

        pis.final_rannung_df_commit = output_df.commit.clone();
    }

    pub fn prove_tree(&self, engine: &E)
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
        for child in self.children.iter() {
            match child.as_ref() {
                JoinCircuit::PageLevelJoin(circuit) => circuit.prove(engine),
                JoinCircuit::TwoPointersProgram(circuit) => circuit.prove(engine),
                JoinCircuit::InternalNode(circuit) => circuit.prove_tree(engine),
            }
        }
    }

    pub fn verify_tree(
        &self,
        engine: &E,
        parent_df: &DataFrame<COMMIT_LEN>,
        child_df: &DataFrame<COMMIT_LEN>,
        output_df: &mut DataFrame<COMMIT_LEN>,
        pairs: &Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,
    ) where
        Val<SC>: PrimeField,
        Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
        Com<SC>: Into<[BabyBear; COMMIT_LEN]>,
    {
        self.verify(pairs);
        for child in self.children.iter() {
            match child.as_ref() {
                JoinCircuit::PageLevelJoin(circuit) => circuit.verify(engine, output_df),
                JoinCircuit::TwoPointersProgram(circuit) => {
                    circuit.verify(engine, parent_df, child_df)
                }
                JoinCircuit::InternalNode(circuit) => {
                    circuit.verify_tree(engine, parent_df, child_df, output_df, pairs)
                }
            }
        }
    }

    pub fn verify(&self, pairs: &Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>) {
        let pis = self.pis.borrow();

        let mut state = self.state.borrow_mut();
        state.running_df_commit = Some(pis.init_running_df_commit.clone());

        for child in self.children.iter() {
            self.verify_child(child.clone(), pairs);
        }

        assert_eq!(
            *state.running_df_commit.as_ref().unwrap(),
            pis.final_rannung_df_commit
        );
    }

    fn verify_child(
        &self,
        circuit: Arc<JoinCircuit<COMMIT_LEN, SC, E>>,
        pairs: &Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,
    ) {
        let pis = self.pis.borrow();

        match circuit.as_ref() {
            JoinCircuit::TwoPointersProgram(tp_program) => {
                let tp_pis = tp_program.pis.borrow();

                assert_eq!(tp_pis.parent_table_commit, pis.parent_table_commit);
                assert_eq!(tp_pis.child_table_commit, pis.child_table_commit);
                assert_eq!(tp_pis.pairs_commit, pis.pairs_commit);
            }
            JoinCircuit::PageLevelJoin(page_level_circuit) => {
                let mut state = self.state.borrow_mut();
                let child_pis = page_level_circuit.pis.borrow();

                assert_eq!(
                    *state.running_df_commit.as_ref().unwrap(),
                    child_pis.init_running_df_commit
                );

                assert_eq!(
                    child_pis.pairs_list_index,
                    pis.pairs_list_index + state.page_level_cnt
                );

                // Ensuring that the page-level circuit has the right parent_page_commit
                assert_eq!(
                    child_pis.parent_page_commit,
                    pairs[(pis.pairs_list_index + state.page_level_cnt) as usize].0
                );

                // Ensuring that the page-level circuit has the right child_page_commit
                assert_eq!(
                    child_pis.child_page_commit,
                    pairs[(pis.pairs_list_index + state.page_level_cnt) as usize].1
                );

                // TODO: decommit self.pis.pairs_commit corresponds to self.pairs. think if this is necessary

                // TODO: I don't think this is necessary?
                assert_eq!(child_pis.pairs_commit, pis.pairs_commit);

                // Updating state
                state.running_df_commit = Some(child_pis.final_running_df_commit.clone());
                state.page_level_cnt += 1;
            }
            JoinCircuit::InternalNode(internal_node_circuit) => {
                let child_pis = internal_node_circuit.pis.borrow();
                let mut state = self.state.borrow_mut();

                assert_eq!(
                    *state.running_df_commit.as_ref().unwrap(),
                    child_pis.init_running_df_commit
                );

                assert_eq!(
                    child_pis.pairs_list_index,
                    pis.pairs_list_index + state.page_level_cnt
                );

                assert_eq!(pis.pairs_commit, child_pis.pairs_commit);

                assert_eq!(pis.parent_table_commit, child_pis.parent_table_commit);

                assert_eq!(pis.child_table_commit, child_pis.child_table_commit);

                state.running_df_commit = Some(child_pis.final_rannung_df_commit.clone());

                let child_state = internal_node_circuit.state.borrow();
                state.page_level_cnt += child_state.page_level_cnt;
            }
        }
    }
}
