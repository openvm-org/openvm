use std::{collections::VecDeque, sync::Arc};

use datafusion::{
    arrow::array::RecordBatch, error::Result, execution::context::SessionContext,
    logical_expr::LogicalPlan,
};

use crate::afs_logical_plan::AfsLogicalPlan;

pub struct AfsExec {
    /// The session context
    pub ctx: SessionContext,
    /// AfsLogicalPlan tree flattened into a vec to be executed sequentially
    pub afs_execution_plan: Vec<Arc<AfsLogicalPlan>>,
}

impl AfsExec {
    pub fn new(ctx: SessionContext, root: LogicalPlan) -> Self {
        let root = ctx.state().optimize(&root).unwrap();
        let afs_logical_plan = Self::convert_logical_tree(&root);
        let afs_execution_plan = Self::flatten_tree(afs_logical_plan);
        Self {
            ctx,
            afs_execution_plan,
        }
    }

    pub fn execute(&self) -> Result<RecordBatch> {
        unimplemented!()
    }

    pub fn convert_logical_tree(root: &LogicalPlan) -> Arc<AfsLogicalPlan> {
        println!("convert_logical_tree Root: {:?}", root);
        fn dfs(node: &LogicalPlan) -> Arc<AfsLogicalPlan> {
            let children: Vec<Arc<AfsLogicalPlan>> =
                node.inputs().iter().map(|child| dfs(child)).collect();
            let afs_node = AfsLogicalPlan::from(node, children);
            Arc::new(afs_node)
        }
        dfs(root)
    }

    pub fn flatten_tree(root: Arc<AfsLogicalPlan>) -> Vec<Arc<AfsLogicalPlan>> {
        println!("flatten_treeRoot: {:?}", root);
        let mut flat_plan = Vec::new();
        let mut stack = VecDeque::new();
        stack.push_back(root);

        while let Some(node) = stack.pop_front() {
            flat_plan.push(node.clone());
            for input in node.inputs() {
                stack.push_back(input.clone());
            }
        }
        println!("Flattened plan: {:?}", flat_plan);

        // flat_plan.reverse();
        flat_plan
    }
}
