// TODO: make sure all pubic values are computed correctly after calls to generate_trace for all circuits

pub mod internal_node;
pub mod page_level_join;
pub mod two_pointers_program;

use std::marker::PhantomData;
use std::sync::Arc;

use afs_chips::inner_join::controller::{T2Format, TableFormat};
use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::engine::StarkEngine;
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::inner_join::page_level_join::PageLevelJoin;
use crate::inner_join::two_pointers_program::TwoPointersProgram;
use crate::{common::Commitment, page_db::PageDb};

use self::internal_node::{InternalNode, JoinCircuit};
use super::dataframe::DataFrame;

pub struct TableJoinController<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>>
{
    parent_table_df: DataFrame<COMMIT_LEN>,
    child_table_df: DataFrame<COMMIT_LEN>,
    page_db: Arc<PageDb<COMMIT_LEN>>,
    root: InternalNode<COMMIT_LEN, SC, E>,

    _phantom: PhantomData<SC>,       // TODO: try removing this later
    _phantom_engine: PhantomData<E>, // TODO: try removing this later
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    TableJoinController<COMMIT_LEN, SC, E>
{
    pub fn new(
        parent_df: DataFrame<COMMIT_LEN>,
        child_df: DataFrame<COMMIT_LEN>,
        page_db: Arc<PageDb<COMMIT_LEN>>,
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

        let pairs = tp_program.pairs.lock().clone();
        let pairs_commit = tp_program.pis.lock().pairs_commit.clone();

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
            page_db,
            root,
            _phantom: PhantomData,
            _phantom_engine: PhantomData,
        }
    }

    // pub fn generate_trace(&self) {
    //     self.root.generate_trace_for_tree();
    // }

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

        let internal_node = InternalNode::<COMMIT_LEN, SC, E>::new(
            vec![Arc::from(*left), Arc::from(*right)],
            pairs.clone(), // TODO: I think pairs should be removed from here
        );

        (Box::new(JoinCircuit::InternalNode(internal_node)), leaves)
    }

    // TODO: Consider adding generate_trace, prove, and verify functions to the controller
    // that will just call the corresponding functions on the root
}
