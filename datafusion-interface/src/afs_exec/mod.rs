use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
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

use crate::{afs_node::AfsNode, committed_page::CommittedPage};

#[derive(Debug)]
enum NodeState<SC: StarkGenericConfig> {
    ZeroOrOneChild,
    /// Nodes with multiple children will have multiple tasks accessing it,
    /// and each task will append their contribution until the last task takes
    /// all the children to build the parent node.
    TwoOrMoreChildren(Mutex<Vec<ExecutionPlanChild<SC>>>),
}

/// To avoid needing to pass single child wrapped in a Vec for nodes
/// with only one child.
pub enum ChildrenContainer<SC: StarkGenericConfig> {
    None,
    One(Arc<AfsNode<SC>>),
    Multiple(Arc<Vec<AfsNode<SC>>>),
}

#[derive(Debug)]
struct LogicalNode<'a, SC: StarkGenericConfig> {
    node: &'a LogicalPlan,
    // None if root
    parent_index: Option<usize>,
    state: NodeState<SC>,
}

#[derive(Debug)]
struct ExecutionPlanChild<SC: StarkGenericConfig> {
    /// Index needed to order children of parent to ensure consistency with original
    /// `LogicalPlan`
    index: usize,
    plan: Arc<Vec<AfsNode<SC>>>,
}

pub struct AfsExec<SC: StarkGenericConfig> {
    /// The session context
    pub ctx: SessionContext,
    /// AfsNode tree flattened into a vec to be executed sequentially
    pub afs_execution_plan: Vec<AfsNode<SC>>,
}

impl<SC: StarkGenericConfig> AfsExec<SC>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub async fn new(ctx: SessionContext, root: LogicalPlan) -> Self {
        let root = ctx.state().optimize(&root).unwrap();
        // let afs_logical_plan = Self::convert_logical_tree(&root);
        // let afs_execution_plan = Self::flatten_tree(afs_logical_plan);
        let afs_execution_plan = Self::create_execution_plan(&root, &ctx.state())
            .await
            .unwrap();
        Self {
            ctx,
            afs_execution_plan,
        }
    }

    pub async fn execute(&mut self) -> Result<Arc<CommittedPage<SC>>> {
        let mut last_result = None;
        for node in &mut self.afs_execution_plan {
            node.execute(&self.ctx).await?;
            last_result = node.output();
        }
        Ok(last_result.unwrap())
    }

    pub async fn keygen(&mut self) -> Result<()> {
        for node in &mut self.afs_execution_plan {
            node.keygen(&self.ctx).await?;
        }
        Ok(())
    }

    pub fn prove(&mut self) -> Result<RecordBatch> {
        unimplemented!();
        // for node in &self.afs_execution_plan {
        //     node.prove()?;
        // }
    }

    pub async fn create_execution_plan(
        root: &LogicalPlan,
        state: &SessionState,
    ) -> Result<Vec<AfsNode<SC>>> {
        let mut flat_tree = vec![];
        let mut dfs_visit_stack = vec![(root, None)];
        let mut node_map: HashMap<usize, Arc<&AfsNode<SC>>> = std::collections::HashMap::new();

        while let Some((node, _parent_index)) = dfs_visit_stack.pop() {
            let current_index = flat_tree.len();
            let children = node.inputs();

            if children.is_empty() {
                let afs_node = AfsNode::from(node, ChildrenContainer::None);
                flat_tree.push(afs_node);
                // let last_node = Arc::new(flat_tree.last().unwrap());
                // node_map.insert(current_index, last_node);
            } else {
                let mut child_indices = vec![];
                for &child in children.iter().rev() {
                    dfs_visit_stack.push((child, Some(current_index)));
                    child_indices.push(flat_tree.len() + dfs_visit_stack.len());
                }

                // let children_container = if child_indices.len() == 1 {
                //     ChildrenContainer::One(Arc::new(*node_map[&child_indices[0]]))
                // } else {
                //     let children = child_indices
                //         .iter()
                //         .map(|&i| *node_map[&i].clone())
                //         .collect::<Vec<_>>();
                //     ChildrenContainer::Multiple(Arc::new(children))
                // };

                // let mut child_afs_nodes = vec![];
                // for &child in children.iter() {
                //     let child_afs_node =
                //         Box::pin(Self::create_execution_plan(child, state)).await?;
                //     child_afs_nodes.extend(child_afs_node);
                // }

                // let children_container = if child_afs_nodes.len() == 1 {
                //     let first = &child_afs_nodes[0];
                //     ChildrenContainer::One(Arc::new(first))
                // } else {
                //     ChildrenContainer::Multiple(Arc::new(child_afs_nodes))
                // };

                // let afs_node = AfsNode::from(node, children_container);
                // flat_tree.push(afs_node);
                // node_map.insert(current_index, Arc::new(flat_tree.last().unwrap()));
            }
        }

        Ok(flat_tree)
    }

    // pub async fn create_execution_plan(
    //     root: &LogicalPlan,
    //     state: &SessionState,
    // ) -> Result<Vec<AfsNode>> {
    //     // Note: below DFS/flatten implementation copied with minor adjustments from datafusion/core/src/physical_planner.rs
    //     // DFS the tree to flatten it into a Vec.
    //     // This will allow us to build the Physical Plan from the leaves up
    //     // to avoid recursion, and also to make it easier to build a valid
    //     // Physical Plan from the start and not rely on some intermediate
    //     // representation (since parents need to know their children at
    //     // construction time).
    //     let mut flat_tree = vec![];
    //     let mut dfs_visit_stack = vec![(None, root)];
    //     // Use this to be able to find the leaves to start construction bottom
    //     // up concurrently.
    //     let mut flat_tree_leaf_indices = vec![];
    //     while let Some((parent_index, node)) = dfs_visit_stack.pop() {
    //         let current_index = flat_tree.len();
    //         // Because of how we extend the visit stack here, we visit the children
    //         // in reverse order of how they appear, so later we need to reverse
    //         // the order of children when building the nodes.
    //         dfs_visit_stack.extend(node.inputs().iter().map(|&n| (Some(current_index), n)));
    //         let state = match node.inputs().len() {
    //             0 => {
    //                 flat_tree_leaf_indices.push(current_index);
    //                 NodeState::ZeroOrOneChild
    //             }
    //             1 => NodeState::ZeroOrOneChild,
    //             _ => {
    //                 let ready_children = Vec::with_capacity(node.inputs().len());
    //                 let ready_children = Mutex::new(ready_children);
    //                 NodeState::TwoOrMoreChildren(ready_children)
    //             }
    //         };
    //         let node = LogicalNode {
    //             node,
    //             parent_index,
    //             state,
    //         };
    //         flat_tree.push(node);
    //     }
    //     let flat_tree = Arc::new(flat_tree);

    //     let planning_concurrency = state.config_options().execution.planning_concurrency;
    //     // Can never spawn more tasks than leaves in the tree, as these tasks must
    //     // all converge down to the root node, which can only be processed by a
    //     // single task.
    //     let max_concurrency = planning_concurrency.min(flat_tree_leaf_indices.len());

    //     // Spawning tasks which will traverse leaf up to the root.
    //     let tasks = flat_tree_leaf_indices
    //         .into_iter()
    //         .map(|index| Self::convert_logical_plan_parent(index, flat_tree.clone(), state));
    //     let mut outputs = futures::stream::iter(tasks)
    //         .buffer_unordered(max_concurrency)
    //         .try_collect::<Vec<_>>()
    //         .await?
    //         .into_iter()
    //         .flatten()
    //         .collect::<Vec<_>>();
    //     // Ideally this never happens if we have a valid LogicalPlan tree
    //     if outputs.len() != 1 {
    //         return internal_err!(
    //             "Failed to convert LogicalPlan to ExecutionPlan: More than one root detected"
    //         );
    //     }
    //     let plan = outputs.pop().unwrap();
    //     Ok(plan)
    // }

    // async fn convert_logical_plan_parent<'a>(
    //     leaf_starter_index: usize,
    //     flat_tree: Arc<Vec<LogicalNode<'a>>>,
    //     state: &SessionState,
    // ) -> Result<Option<AfsNode>> {
    //     // We always start with a leaf, so can ignore status and pass empty children
    //     let mut node = flat_tree.get(leaf_starter_index).ok_or_else(|| {
    //         internal_datafusion_err!("Invalid index whilst creating initial physical plan")
    //     })?;
    //     let mut plan =
    //         Self::map_logical_plan_to_afs_node(node.node, state, ChildrenContainer::None).await?;
    //     let mut current_index = leaf_starter_index;
    //     // parent_index is None only for root
    //     while let Some(parent_index) = node.parent_index {
    //         node = flat_tree.get(parent_index).ok_or_else(|| {
    //             internal_datafusion_err!("Invalid index whilst creating initial physical plan")
    //         })?;
    //         match &node.state {
    //             NodeState::ZeroOrOneChild => {
    //                 plan = Self::map_logical_plan_to_afs_node(
    //                     node.node,
    //                     state,
    //                     ChildrenContainer::One(plan),
    //                 )
    //                 .await?;
    //             }
    //             // See if we have all children to build the node.
    //             NodeState::TwoOrMoreChildren(children) => {
    //                 let mut children: Vec<ExecutionPlanChild> = {
    //                     let mut guard = children.lock().await;
    //                     // Add our contribution to this parent node.
    //                     // Vec is pre-allocated so no allocation should occur here.
    //                     guard.push(ExecutionPlanChild {
    //                         index: current_index,
    //                         plan,
    //                     });
    //                     if guard.len() < node.node.inputs().len() {
    //                         // This node is not ready yet, still pending more children.
    //                         // This task is finished forever.
    //                         return Ok(None);
    //                     }

    //                     // With this task's contribution we have enough children.
    //                     // This task is the only one building this node now, and thus
    //                     // no other task will need the Mutex for this node, so take
    //                     // all children.
    //                     std::mem::take(guard.as_mut())
    //                 };

    //                 // Indices refer to position in flat tree Vec, which means they are
    //                 // guaranteed to be unique, hence unstable sort used.
    //                 //
    //                 // We reverse sort because of how we visited the node in the initial
    //                 // DFS traversal (see above).
    //                 children.sort_unstable_by_key(|epc| std::cmp::Reverse(epc.index));
    //                 let children = children.into_iter().map(|epc| epc.plan).collect();
    //                 let children = ChildrenContainer::Multiple(children);
    //                 plan = Self::map_logical_plan_to_afs_node(node.node, state, children).await?;
    //             }
    //         }
    //         current_index = parent_index;
    //     }
    //     // Only one task should ever reach this point for a valid LogicalPlan tree.
    //     Ok(Some(plan))
    // }

    // async fn map_logical_plan_to_afs_node(
    //     node: &LogicalPlan,
    //     state: &SessionState,
    //     children: ChildrenContainer,
    // ) -> Result<Arc<AfsNode>> {
    //     let afs_node = match children {
    //         ChildrenContainer::None => AfsNode::from(node, vec![]),
    //         ChildrenContainer::One(child) => AfsNode::from(node, vec![child]),
    //         ChildrenContainer::Multiple(children) => AfsNode::from(node, children),
    //     };
    //     Ok(Arc::new(afs_node))
    // }

    // pub fn convert_logical_tree(root: &LogicalPlan) -> Arc<AfsNode> {
    //     println!("convert_logical_tree Root: {:?}", root);
    //     fn dfs(node: &LogicalPlan) -> Arc<AfsNode> {
    //         let children: Vec<Arc<AfsNode>> =
    //             node.inputs().iter().map(|child| dfs(child)).collect();
    //         let afs_node = AfsNode::from(node, children);
    //         Arc::new(afs_node)
    //     }
    //     dfs(root)
    // }

    // pub fn flatten_tree(root: Arc<AfsNode>) -> Vec<Arc<AfsNode>> {
    //     println!("flatten_treeRoot: {:?}", root);
    //     let mut flat_plan = Vec::new();
    //     let mut stack = VecDeque::new();
    //     stack.push_back(root);

    //     while let Some(node) = stack.pop_front() {
    //         flat_plan.push(node.clone());
    //         for input in node.inputs() {
    //             stack.push_back(input.clone());
    //         }
    //     }
    //     println!("Flattened plan: {:?}", flat_plan);

    //     // flat_plan.reverse();
    //     flat_plan
    // }
}
