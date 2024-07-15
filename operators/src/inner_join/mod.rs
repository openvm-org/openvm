pub mod internal_node;
pub mod page_level_join;
pub mod two_pointers_program;

use std::marker::PhantomData;
use std::sync::Arc;

use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::engine::StarkEngine;
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::common::Commitment;
use crate::dataframe::DataFrameType;
use crate::inner_join::internal_node::InternalNodePis;
use crate::inner_join::page_level_join::PageLevelJoin;
use crate::inner_join::two_pointers_program::{TwoPointersProgram, TwoPointersProgramPis};

use self::internal_node::{InternalNode, JoinCircuit};
use super::dataframe::DataFrame;

#[derive(derive_new::new)]
pub struct TableJoinController<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>>
{
    parent_table_df: DataFrame<COMMIT_LEN>,
    child_table_df: DataFrame<COMMIT_LEN>,

    _phantom: PhantomData<SC>,       // TODO: try removing this later
    _phantom_engine: PhantomData<E>, // TODO: try removing this later
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    TableJoinController<COMMIT_LEN, SC, E>
{
    pub fn build_tree(&self) -> InternalNode<COMMIT_LEN, SC, E>
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
        let mut tp_program =
            TwoPointersProgram::<COMMIT_LEN, SC, E>::new(TwoPointersProgramPis::new(
                self.parent_table_df.commit.clone(),
                self.child_table_df.commit.clone(),
                Commitment::<COMMIT_LEN>::default(), // This is a placeholder. Should be updated later when program.run() is called
            ));
        let pairs = tp_program.run(&self.parent_table_df, &self.child_table_df);

        let pairs_commit = tp_program.pis.pairs_commit.clone();

        leaves.push(Some(Box::new(JoinCircuit::TwoPointersProgram(tp_program))));

        let mut output_df = DataFrame::<COMMIT_LEN>::empty(DataFrameType::Unindexed);

        for (parent_commit, child_commit) in pairs.iter() {
            let page_level_join = PageLevelJoin::<COMMIT_LEN, SC, E>::load_pages_from_commits(
                parent_commit.clone(),
                child_commit.clone(),
            );

            leaves.push(Some(Box::new(JoinCircuit::PageLevelJoin(page_level_join))));
        }

        let mut df_cur_page = 0;
        let mut pairs_list_index = 0;

        self.build_tree_dfs(
            0,
            leaves.len() - 1,
            leaves,
            &mut output_df,
            &mut df_cur_page,
            &pairs,
            &pairs_commit,
            &mut pairs_list_index,
        );
        todo!()
    }

    fn build_tree_dfs(
        &self,
        l: usize,
        r: usize,
        mut leaves: Vec<Option<Box<JoinCircuit<COMMIT_LEN, SC, E>>>>,
        output_df: &mut DataFrame<COMMIT_LEN>,
        df_cur_page: &mut u32,
        pairs: &Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,
        pairs_commit: &Commitment<COMMIT_LEN>,
        pairs_list_index: &mut u32,
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
            // TODO: is df_cur_page exactly pairs_list_index? maybe make them the same

            let mut leaf = leaves[l].take().unwrap();

            match leaf.as_mut() {
                JoinCircuit::PageLevelJoin(ref mut page_level_join) => {
                    page_level_join.generate_trace(output_df, *pairs_list_index);
                    *pairs_list_index += 1;
                }
                _ => {}
            }

            return (leaf, leaves);
        }

        let init_running_df_commit = output_df.commit.clone();
        // TODO: make sure those are updated
        let init_df_cur_page = *df_cur_page;
        let init_pairs_list_index = *pairs_list_index;

        let mid = (l + r) / 2;
        let (left, leaves) = self.build_tree_dfs(
            l,
            mid,
            leaves,
            output_df,
            df_cur_page,
            pairs,
            pairs_commit,
            pairs_list_index,
        );
        let (right, leaves) = self.build_tree_dfs(
            mid + 1,
            r,
            leaves,
            output_df,
            df_cur_page,
            pairs,
            pairs_commit,
            pairs_list_index,
        );
        let final_running_df_commit = output_df.commit.clone();

        let internal_node = InternalNode::<COMMIT_LEN, SC, E>::new(
            vec![Arc::from(*left), Arc::from(*right)],
            pairs.clone(), // TODO: I think pairs should be removed from here
            InternalNodePis::new(
                init_running_df_commit,
                init_df_cur_page,
                pairs_commit.clone(),
                init_pairs_list_index,
                final_running_df_commit.clone(),
                self.parent_table_df.commit.clone(),
                self.child_table_df.commit.clone(),
            ),
        );

        (Box::new(JoinCircuit::InternalNode(internal_node)), leaves)
    }

    // TODO: Consider adding generate_trace, prove, and verify functions to the controller
    // that will just call the corresponding functions on the root
}
