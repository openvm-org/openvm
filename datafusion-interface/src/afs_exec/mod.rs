use afs_stark_backend::config::StarkGenericConfig;
use color_eyre::eyre::Result;
use datafusion::{
    arrow::array::RecordBatch,
    execution::context::{SessionContext, SessionState},
    logical_expr::{LogicalPlan, TableScan},
};

use crate::afs_logical_plan::{plan::PageScan, AfsLogicalPlan};

pub struct AfsExec {
    pub ctx: SessionContext,
    // pub plan: LogicalPlan,
    pub afs_logical_plan: AfsLogicalPlan,
    pub afs_execution_plan: Vec<AfsLogicalPlan>,
}

impl AfsExec {
    pub fn new(ctx: SessionContext, plan: LogicalPlan) -> Self {
        let plan = ctx.state().optimize(&plan).unwrap();
        let afs_logical_plan = Self::create_execution_plan(&plan, &ctx.state());
        Self {
            ctx,
            afs_logical_plan,
            afs_execution_plan,
        }
    }

    pub fn execute(&self) -> Result<RecordBatch> {
        unimplemented!()
    }

    pub fn create_execution_plan(root: &LogicalPlan, state: &SessionState) {
        // Note: below DFS/flatten implementation copied from datafusion/core/src/physical_planner.rs
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
    }

    pub fn map_logical_plan_to_afs(
        node: &LogicalPlan,
        state: &SessionState,
    ) -> Result<AfsLogicalPlan> {
        let afs_node = match node {
            LogicalPlan::TableScan(table_scan) => {
                let page_id = table_scan.table_name.to_string();
                let source = table_scan.source.clone();
                AfsLogicalPlan::PageScan(PageScan { page_id, source })
            }
            _ => unimplemented!(),
        };
        Ok(afs_node)
    }
}
