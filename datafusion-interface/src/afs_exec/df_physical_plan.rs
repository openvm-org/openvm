/// This file is currently unused
///
use std::{collections::VecDeque, sync::Arc};

use afs_stark_backend::config::StarkGenericConfig;
use datafusion::{
    arrow::array::RecordBatch,
    common::{internal_datafusion_err, internal_err},
    error::Result,
    execution::context::{SessionContext, SessionState},
    logical_expr::LogicalPlan,
};
use tokio::sync::Mutex;

use crate::{
    afs_expr::AfsExpr,
    afs_logical_plan::{
        plan::{Filter, PageScan},
        AfsNode,
    },
};

#[derive(Debug)]
enum NodeState {
    ZeroOrOneChild,
    /// Nodes with multiple children will have multiple tasks accessing it,
    /// and each task will append their contribution until the last task takes
    /// all the children to build the parent node.
    TwoOrMoreChildren(Mutex<Vec<ExecutionPlanChild>>),
}

/// To avoid needing to pass single child wrapped in a Vec for nodes
/// with only one child.
enum ChildrenContainer {
    None,
    One(Arc<dyn ExecutionPlan>),
    Multiple(Vec<Arc<dyn ExecutionPlan>>),
}

#[derive(Debug)]
struct LogicalNode<'a> {
    node: &'a LogicalPlan,
    // None if root
    parent_index: Option<usize>,
    state: NodeState,
}

impl AfsExec {
    pub async fn create_execution_plan(
        root: &LogicalPlan,
        state: &SessionState,
    ) -> Result<Vec<AfsNode>> {
        // Note: below DFS/flatten implementation copied with minor adjustments from datafusion/core/src/physical_planner.rs
        // DFS the tree to flatten it into a Vec.
        // This will allow us to build the Physical Plan from the leaves up
        // to avoid recursion, and also to make it easier to build a valid
        // Physical Plan from the start and not rely on some intermediate
        // representation (since parents need to know their children at
        // construction time).
        let mut flat_tree = vec![];
        let mut dfs_visit_stack = vec![(None, root)];
        // Use this to be able to find the leaves to start construction bottom
        // up concurrently.
        let mut flat_tree_leaf_indices = vec![];
        while let Some((parent_index, node)) = dfs_visit_stack.pop() {
            let current_index = flat_tree.len();
            // Because of how we extend the visit stack here, we visit the children
            // in reverse order of how they appear, so later we need to reverse
            // the order of children when building the nodes.
            dfs_visit_stack.extend(node.inputs().iter().map(|&n| (Some(current_index), n)));
            let state = match node.inputs().len() {
                0 => {
                    flat_tree_leaf_indices.push(current_index);
                    NodeState::ZeroOrOneChild
                }
                1 => NodeState::ZeroOrOneChild,
                _ => {
                    let ready_children = Vec::with_capacity(node.inputs().len());
                    let ready_children = Mutex::new(ready_children);
                    NodeState::TwoOrMoreChildren(ready_children)
                }
            };
            let node = LogicalNode {
                node,
                parent_index,
                state,
            };
            flat_tree.push(node);
        }
        let flat_tree = Arc::new(flat_tree);

        let planning_concurrency = state.config_options().execution.planning_concurrency;
        // Can never spawn more tasks than leaves in the tree, as these tasks must
        // all converge down to the root node, which can only be processed by a
        // single task.
        let max_concurrency = planning_concurrency.min(flat_tree_leaf_indices.len());

        // Spawning tasks which will traverse leaf up to the root.
        let tasks = flat_tree_leaf_indices
            .into_iter()
            .map(|index| Self::task_helper(index, flat_tree.clone(), state));
        let mut outputs = futures::stream::iter(tasks)
            .buffer_unordered(max_concurrency)
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        // Ideally this never happens if we have a valid LogicalPlan tree
        if outputs.len() != 1 {
            return internal_err!(
                "Failed to convert LogicalPlan to ExecutionPlan: More than one root detected"
            );
        }
        let plan = outputs.pop().unwrap();
        Ok(plan)
    }

    async fn task_helper(
        leaf_starter_index: usize,
        flat_tree: Arc<Vec<LogicalNode>>,
        state: &SessionState,
    ) -> Result<Option<Vec<LogicalPlan>>> {
        // We always start with a leaf, so can ignore status and pass empty children
        let mut node = flat_tree.get(leaf_starter_index).ok_or_else(|| {
            internal_datafusion_err!("Invalid index whilst creating initial physical plan")
        })?;
        let mut plan = Self::map_logical_plan_to_afs(node.node, state).await?;
        let mut current_index = leaf_starter_index;
        // parent_index is None only for root
        while let Some(parent_index) = node.parent_index {
            node = flat_tree.get(parent_index).ok_or_else(|| {
                internal_datafusion_err!("Invalid index whilst creating initial physical plan")
            })?;
            match &node.state {
                NodeState::ZeroOrOneChild => {
                    plan = Self::map_logical_plan_to_afs(
                        node.node,
                        state,
                        ChildrenContainer::One(plan),
                    );
                }
                // See if we have all children to build the node.
                NodeState::TwoOrMoreChildren(children) => {
                    let mut children: Vec<ExecutionPlanChild> = {
                        let mut guard = children.lock().await;
                        // Add our contribution to this parent node.
                        // Vec is pre-allocated so no allocation should occur here.
                        guard.push(ExecutionPlanChild {
                            index: current_index,
                            plan,
                        });
                        if guard.len() < node.node.inputs().len() {
                            // This node is not ready yet, still pending more children.
                            // This task is finished forever.
                            return Ok(None);
                        }

                        // With this task's contribution we have enough children.
                        // This task is the only one building this node now, and thus
                        // no other task will need the Mutex for this node, so take
                        // all children.
                        std::mem::take(guard.as_mut())
                    };

                    // Indices refer to position in flat tree Vec, which means they are
                    // guaranteed to be unique, hence unstable sort used.
                    //
                    // We reverse sort because of how we visited the node in the initial
                    // DFS traversal (see above).
                    children.sort_unstable_by_key(|epc| std::cmp::Reverse(epc.index));
                    let children = children.into_iter().map(|epc| epc.plan).collect();
                    let children = ChildrenContainer::Multiple(children);
                    plan = Self::map_logical_plan_to_afs(node.node, state, children).await?;
                }
            }
            current_index = parent_index;
        }
        // Only one task should ever reach this point for a valid LogicalPlan tree.
        Ok(Some(plan))
    }
}
