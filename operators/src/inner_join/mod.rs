// TODO: make sure all pubic values are computed correctly after calls to generate_trace for all circuits

pub mod internal_node;
pub mod page_level_join;
pub mod two_pointers_program;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use afs_chips::inner_join::controller::{T2Format, TableFormat};
use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::engine::StarkEngine;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::common::provider::BTreeMapPageLoader;
use crate::common::Commitment;
use crate::dataframe::DataFrameType;
use crate::inner_join::page_level_join::PageLevelJoin;
use crate::inner_join::two_pointers_program::TwoPointersProgram;

use self::internal_node::{InternalNode, JoinCircuit};
use super::dataframe::DataFrame;

pub struct TableJoinController<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>>
{
    parent_table_df: DataFrame<COMMIT_LEN>,
    child_table_df: DataFrame<COMMIT_LEN>,
    page_provider: Arc<BTreeMapPageLoader<SC, COMMIT_LEN>>,
    root: InternalNode<COMMIT_LEN, SC, E>,
    pairs: Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,
    pairs_commit: Commitment<COMMIT_LEN>,
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    TableJoinController<COMMIT_LEN, SC, E>
{
    pub fn new(
        parent_df: DataFrame<COMMIT_LEN>,
        child_df: DataFrame<COMMIT_LEN>,
        page_provider: Arc<BTreeMapPageLoader<SC, COMMIT_LEN>>,
        parent_table_format: &TableFormat,
        child_table_format: &T2Format,
        decomp: usize,
        engine: &E,
    ) -> Self
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
        let mut leaves: Vec<Option<Box<JoinCircuit<COMMIT_LEN, SC, E>>>> = vec![];

        // Adding the two-pointers program as a leaf
        let tp_program = TwoPointersProgram::<COMMIT_LEN, SC, E>::default();

        let pairs = tp_program.pairs.borrow().clone();
        let pairs_commit = tp_program.pis.borrow().pairs_commit.clone();

        leaves.push(Some(Box::new(JoinCircuit::TwoPointersProgram(tp_program))));

        for (parent_commit, child_commit) in pairs.iter() {
            let page_level_join = PageLevelJoin::<COMMIT_LEN, SC, E>::new(
                parent_commit.clone(),
                child_commit.clone(),
                parent_table_format.clone(),
                child_table_format.clone(),
                decomp,
            );

            leaves.push(Some(Box::new(JoinCircuit::PageLevelJoin(page_level_join))));
        }

        assert!(leaves.len() > 1);

        let mut df_cur_page = 0;
        let mut pairs_list_index = 0;

        let (root, _) = Self::build_tree_dfs(
            0,
            leaves.len() - 1,
            leaves,
            &mut df_cur_page,
            &pairs,
            &pairs_commit,
            &mut pairs_list_index,
            &engine,
        );

        let root = match *root {
            JoinCircuit::InternalNode(node) => node,
            _ => unreachable!(),
        };

        Self {
            parent_table_df: parent_df,
            child_table_df: child_df,
            page_provider,
            pairs,
            pairs_commit,
            root,
        }
    }

    pub fn generate_trace(&mut self, engine: &E)
    where
        Val<SC>: PrimeField,
    {
        let mut output_df = DataFrame::<COMMIT_LEN>::empty(DataFrameType::new_unindexed());
        let mut pairs_list_index = 0;

        self.root.generate_trace_for_tree(
            self.page_provider.clone(),
            &mut output_df,
            &self.pairs_commit,
            &mut pairs_list_index,
            &self.parent_table_df,
            &self.child_table_df,
            engine,
        );
    }

    pub fn prove(&self, engine: &E)
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
        self.root.prove_tree(engine);
    }

    pub fn verify(&self, engine: &E, output_df: &mut DataFrame<COMMIT_LEN>)
    where
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
        self.root.verify_tree(
            engine,
            &self.parent_table_df,
            &self.child_table_df,
            output_df,
            &self.pairs,
        );
    }

    fn build_tree_dfs(
        l: usize,
        r: usize,
        mut leaves: Vec<Option<Box<JoinCircuit<COMMIT_LEN, SC, E>>>>,
        df_cur_page: &mut u32,
        pairs: &Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,
        pairs_commit: &Commitment<COMMIT_LEN>,
        pairs_list_index: &mut u32,
        engine: &E,
    ) -> (
        Box<JoinCircuit<COMMIT_LEN, SC, E>>,
        Vec<Option<Box<JoinCircuit<COMMIT_LEN, SC, E>>>>,
    )
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
        if l == r {
            return (leaves[l].take().unwrap(), leaves);
        }

        let mid = (l + r) / 2;
        let (left, leaves) = Self::build_tree_dfs(
            l,
            mid,
            leaves,
            df_cur_page,
            pairs,
            pairs_commit,
            pairs_list_index,
            engine,
        );
        let (right, leaves) = Self::build_tree_dfs(
            mid + 1,
            r,
            leaves,
            df_cur_page,
            pairs,
            pairs_commit,
            pairs_list_index,
            engine,
        );

        let internal_node =
            InternalNode::<COMMIT_LEN, SC, E>::new(vec![Arc::from(*left), Arc::from(*right)]);

        (Box::new(JoinCircuit::InternalNode(internal_node)), leaves)
    }
}
