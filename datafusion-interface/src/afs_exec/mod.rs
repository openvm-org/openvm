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
use futures::{StreamExt, TryStreamExt};
use p3_field::PrimeField64;
use serde::{de::DeserializeOwned, Serialize};
use tokio::sync::Mutex;

use crate::{afs_node::AfsNode, committed_page::CommittedPage, PCS_LOG_DEGREE};

#[derive(Debug)]
enum NodeState<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    ZeroOrOneChild,
    /// Nodes with multiple children will have multiple tasks accessing it,
    /// and each task will append their contribution until the last task takes
    /// all the children to build the parent node.
    TwoOrMoreChildren(Mutex<Vec<ExecutionPlanChild<SC, E>>>),
}

/// To avoid needing to pass single child wrapped in a Vec for nodes
/// with only one child.
pub enum ChildrenContainer<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    None,
    One(Arc<Vec<AfsNode<SC, E>>>),
    Multiple(Arc<Vec<AfsNode<SC, E>>>),
}

#[derive(Debug)]
struct LogicalNode<'a, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    node: &'a LogicalPlan,
    // None if root
    parent_index: Option<usize>,
    state: NodeState<SC, E>,
}

#[derive(Debug)]
struct ExecutionPlanChild<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    /// Index needed to order children of parent to ensure consistency with original
    /// `LogicalPlan`
    index: usize,
    plan: Arc<Vec<AfsNode<SC, E>>>,
}

pub struct AfsExec<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    /// The session context from DataFusion
    pub ctx: SessionContext,
    /// STARK engine used for cryptographic operations
    pub engine: E,
    /// AfsNode tree flattened into a vec to be executed sequentially
    pub afs_execution_plan: Vec<AfsNode<SC, E>>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> AfsExec<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub async fn new(ctx: SessionContext, root: LogicalPlan, engine: E) -> Self {
        let root = ctx.state().optimize(&root).unwrap();
        let mut afs_execution_plan = vec![];
        Self::create_execution_plan(&mut afs_execution_plan, &root)
            .await
            .unwrap();
        Self {
            ctx,
            engine,
            afs_execution_plan,
        }
    }

    pub async fn execute(&mut self) -> Result<Arc<CommittedPage<SC>>> {
        for node in &mut self.afs_execution_plan {
            node.execute(&self.ctx).await?;
        }
        Ok(self.afs_execution_plan.last().unwrap().output().unwrap())
    }

    pub async fn keygen(&mut self) -> Result<()> {
        for node in &mut self.afs_execution_plan {
            node.keygen(&self.ctx, &self.engine).await?;
        }
        Ok(())
    }

    pub fn prove(&mut self) -> Result<RecordBatch> {
        unimplemented!();
        // for node in &self.afs_execution_plan {
        //     node.prove()?;
        // }
    }

    /// Converts a LogicalPlan tree to a flat AfsNode vec. Starts from the root (output) node and works backwards until it reaches the input(s).
    pub async fn create_execution_plan(
        flattened: &mut Vec<AfsNode<SC, E>>,
        root: &LogicalPlan,
    ) -> Result<()> {
        let mut dfs_visit_stack = vec![(root, None)];

        while let Some((node, _parent_index)) = dfs_visit_stack.pop() {
            let inputs = node.inputs();

            if inputs.is_empty() {
                let afs_node = AfsNode::from(node, ChildrenContainer::None);
                flattened.push(afs_node);
            } else {
                let mut child_afs_nodes: Vec<AfsNode<SC, E>> = vec![];

                for &input in inputs.iter().rev() {
                    dfs_visit_stack.push((input, Some(flattened.len())));
                }

                for &child in inputs.iter() {
                    Box::pin(Self::create_execution_plan(&mut child_afs_nodes, child)).await?;
                }

                let children_container = if child_afs_nodes.len() == 1 {
                    ChildrenContainer::One(Arc::new(child_afs_nodes))
                } else {
                    ChildrenContainer::Multiple(Arc::new(child_afs_nodes))
                };

                let afs_node = AfsNode::from(node, children_container);
                flattened.push(afs_node);
            }
        }

        Ok(())
    }
}
