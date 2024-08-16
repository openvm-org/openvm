use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{self, BabyBearPoseidon2Engine},
        EngineType, FriParameters,
    },
    engine::StarkEngine,
};
use datafusion::{
    arrow::array::RecordBatch,
    common::{internal_datafusion_err, internal_err},
    error::Result,
    execution::context::{SessionContext, SessionState},
    logical_expr::LogicalPlan,
};
use futures::{lock::Mutex, StreamExt, TryStreamExt};
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};

use crate::{afs_node::AfsNode, committed_page::CommittedPage, PCS_LOG_DEGREE};

/// To avoid needing to pass single child wrapped in a Vec for nodes
/// with only one child.
// pub enum ChildrenContainer<SC: StarkGenericConfig, E: StarkEngine<SC>> {
//     None,
//     One(Arc<AfsNode<SC, E>>),
//     Multiple(Arc<Vec<AfsNode<SC, E>>>),
// }

// #[derive(Debug)]
// enum NodeState<SC: StarkGenericConfig, E: StarkEngine<SC>> {
//     ZeroOrOneChild,
//     /// Nodes with multiple children will have multiple tasks accessing it,
//     /// and each task will append their contribution until the last task takes
//     /// all the children to build the parent node.
//     TwoOrMoreChildren(Mutex<Vec<ExecutionPlanChild<SC, E>>>),
// }

// #[derive(Debug)]
// struct LogicalNode<'a, SC: StarkGenericConfig, E: StarkEngine<SC>> {
//     node: &'a LogicalPlan,
//     // None if root
//     parent_index: Option<usize>,
//     state: NodeState<SC, E>,
// }

// #[derive(Debug)]
// struct ExecutionPlanChild<SC: StarkGenericConfig, E: StarkEngine<SC>> {
//     /// Index needed to order children of parent to ensure consistency with original
//     /// `LogicalPlan`
//     index: usize,
//     plan: Arc<Vec<AfsNode<SC, E>>>,
// }

pub struct AfsExec<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    /// The session context from DataFusion
    pub ctx: SessionContext,
    /// STARK engine used for cryptographic operations
    pub engine: E,
    /// AfsNode tree flattened into a vec to be executed sequentially
    pub afs_execution_plan: Vec<Arc<Mutex<AfsNode<SC, E>>>>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> AfsExec<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub async fn new(ctx: SessionContext, root: LogicalPlan, engine: E) -> Self {
        let root = ctx.state().optimize(&root).unwrap();
        let afs_execution_plan = Self::create_execution_plan(root).await.unwrap();
        Self {
            ctx,
            engine,
            afs_execution_plan,
        }
    }

    pub async fn last_node(&self) -> Result<Arc<Mutex<AfsNode<SC, E>>>> {
        let last_node = self.afs_execution_plan.last().unwrap().to_owned();
        Ok(last_node)
    }

    pub async fn execute(&mut self) -> Result<()> {
        for node in &mut self.afs_execution_plan {
            let mut node = node.lock().await;
            node.execute(&self.ctx, &self.engine).await?;
        }
        Ok(())
    }

    pub async fn keygen(&mut self) -> Result<()> {
        for node in &mut self.afs_execution_plan {
            let mut node = node.lock().await;
            node.keygen(&self.ctx, &self.engine).await?;
        }
        Ok(())
    }

    pub async fn prove(&mut self) -> Result<()> {
        for node in &mut self.afs_execution_plan {
            let mut node = node.lock().await;
            node.prove(&self.ctx, &self.engine).await?;
        }
        Ok(())
    }

    /// Creates the flattened execution plan from a LogicalPlan tree root node.
    async fn create_execution_plan(root: LogicalPlan) -> Result<Vec<Arc<Mutex<AfsNode<SC, E>>>>> {
        let mut flattened = vec![];
        Self::flatten_logical_plan_tree(&mut flattened, &root).await?;
        Ok(flattened)
    }

    /// Converts a LogicalPlan tree to a flat AfsNode vec. Starts from the root (output) node and works backwards until it reaches the input(s).
    async fn flatten_logical_plan_tree(
        flattened: &mut Vec<Arc<Mutex<AfsNode<SC, E>>>>,
        root: &LogicalPlan,
    ) -> Result<usize> {
        println!("flatten_logical_plan_tree {:?}", root);
        let current_index = flattened.len();

        let inputs = root.inputs();

        if inputs.is_empty() {
            let afs_node = Arc::new(Mutex::new(AfsNode::from(root, vec![])));
            flattened.push(afs_node);
        } else {
            let mut input_indexes = vec![];
            for &input in inputs.iter() {
                let input_index =
                    Box::pin(Self::flatten_logical_plan_tree(flattened, input)).await?;
                input_indexes.push(input_index);
            }

            let input_pointers = input_indexes
                .iter()
                .map(|i| Arc::clone(&flattened[*i]))
                .collect::<Vec<Arc<Mutex<AfsNode<SC, E>>>>>();
            let afs_node = Arc::new(Mutex::new(AfsNode::from(root, input_pointers)));
            flattened.push(afs_node);
        }

        Ok(current_index)
    }
    // async fn flatten_logical_plan_tree(
    //     flattened: &mut Vec<AfsNode<SC, E>>,
    //     root: &LogicalPlan,
    // ) -> Result<usize> {
    //     println!("flatten_logical_plan_tree {:?}", root);
    //     let mut dfs_visit_stack = vec![(root, None)];
    //     let current_index = flattened.len();

    //     while let Some((node, parent_index)) = dfs_visit_stack.pop() {
    //         let inputs = node.inputs();

    //         if inputs.is_empty() {
    //             let afs_node = AfsNode::from(node, vec![]);
    //             flattened.push(afs_node);
    //         } else {
    //             // let mut child_afs_nodes: Vec<AfsNode<SC, E>> = vec![];

    //             for &input in inputs.iter().rev() {
    //                 dfs_visit_stack.push((input, Some(current_index)));
    //             }

    //             let mut child_indexes = vec![];
    //             for &child in inputs.iter() {
    //                 let child_index =
    //                     Box::pin(Self::flatten_logical_plan_tree(flattened, child)).await?;
    //                 child_indexes.push(child_index);
    //             }

    //             // let children_container = if child_afs_nodes.len() == 1 {
    //             //     let n = child_afs_nodes.remove(0);
    //             //     ChildrenContainer::One(Arc::new(n))
    //             // } else {
    //             //     ChildrenContainer::Multiple(Arc::new(child_afs_nodes))
    //             // };
    //             let child_pointers = child_indexes
    //                 .iter()
    //                 .map(|i| Arc::new(&flattened.get))
    //                 .collect::<Vec<Arc<AfsNode<SC, E>>>>();
    //             let afs_node = AfsNode::from(node, child_pointers);
    //             flattened.push(afs_node);

    //             // if let Some(parent_idx) = parent_index {
    //             //     if let Some(parent_node) = flattened.get_mut(parent_idx) {
    //             //         parent_node.add_child(Arc::new(afs_node));
    //             //     }
    //             // }
    //         }
    //     }

    //     Ok(current_index)
    // }
}
