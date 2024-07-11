pub mod internal_node;
pub mod page_level_join;
pub mod two_pointers_program;

use std::marker::PhantomData;
use std::sync::Arc;

use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::engine::StarkEngine;
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use parking_lot::Mutex;

use crate::common::Commitment;
use crate::dataframe::DataFrameType;
use crate::inner_join::internal_node::InternalNodePis;
use crate::inner_join::page_level_join::PageLevelJoin;
use crate::inner_join::two_pointers_program::{TwoPointersProgram, TwoPointersProgramPis};
use crate::utils::next_power_of_two;

use self::internal_node::InternalNode;
use super::common::Verifiable;
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
        let mut leaves: Vec<Arc<Mutex<dyn Verifiable<SC, E>>>> = vec![];

        // Adding the two-pointers program as a leaf
        let mut tp_program = TwoPointersProgram::<COMMIT_LEN, SC, E>::new(
            self.parent_table_df.clone(), // TODO: remove this -- don't clone
            self.child_table_df.clone(),
            TwoPointersProgramPis::new(
                self.parent_table_df.commit.clone(),
                self.child_table_df.commit.clone(),
                Commitment::<COMMIT_LEN>::default(), // This is a placeholder. Should be updated later when program.run() is called
            ),
        );
        let pairs = tp_program.run();

        let pairs_commit = tp_program.pis.pairs_commit.clone();

        leaves.push(Arc::new(Mutex::new(tp_program)));

        let output_df_len = next_power_of_two(pairs.len() as u32) as usize;
        let mut output_df =
            DataFrame::<COMMIT_LEN>::empty(output_df_len, DataFrameType::new_unindexed());

        // TODO: make sure those are updated correctly
        let mut df_cur_page = 0;
        let mut pairs_list_index = 0;

        for (parent_commit, child_commit) in pairs.iter() {
            let mut page_level_join = PageLevelJoin::<COMMIT_LEN, SC, E>::load_pages_from_commits(
                parent_commit.clone(),
                child_commit.clone(),
                &output_df,
                df_cur_page,
                pairs_commit.clone(),
                pairs_list_index,
            );

            page_level_join.generate_trace(&mut output_df, df_cur_page);
            df_cur_page += 1;
            pairs_list_index += 1;

            leaves.push(Arc::new(Mutex::new(page_level_join)));
        }

        self.build_tree_dfs(
            0,
            leaves.len() - 1,
            &leaves,
            &mut output_df,
            &mut df_cur_page,
            &pairs,
            &pairs_commit,
            &mut pairs_list_index,
        );
        todo!()
    }

    pub fn verify(
        &self,
        root: &mut InternalNode<COMMIT_LEN, SC, E>,
        engine: &E,
    ) -> Result<(), VerificationError>
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
        root.verify(engine)
    }

    fn build_tree_dfs(
        &self,
        l: usize,
        r: usize,
        leaves: &Vec<Arc<Mutex<dyn Verifiable<SC, E>>>>,
        output_df: &mut DataFrame<COMMIT_LEN>,
        df_cur_page: &mut u32,
        pairs: &Vec<(Commitment<COMMIT_LEN>, Commitment<COMMIT_LEN>)>,
        pairs_commit: &Commitment<COMMIT_LEN>,
        pairs_list_index: &mut u32,
    ) -> Arc<Mutex<dyn Verifiable<SC, E>>>
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
            // TODO: move the trace generation to here?
            return leaves[l].clone();
        }

        let init_running_df_commit = output_df.commit.clone();
        // TODO: make sure those are updated
        let init_df_cur_page = *df_cur_page;
        let init_pairs_list_index = *pairs_list_index;

        let mid = (l + r) / 2;
        let left = self.build_tree_dfs(
            l,
            mid,
            leaves,
            output_df,
            df_cur_page,
            pairs,
            pairs_commit,
            pairs_list_index,
        );
        let right = self.build_tree_dfs(
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

        let children = vec![Arc::from(left), Arc::from(right)];

        let internal_node = InternalNode::<COMMIT_LEN, SC, E>::new(
            children,
            pairs.clone(),
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

        Arc::new(Mutex::new(internal_node))
    }
}
