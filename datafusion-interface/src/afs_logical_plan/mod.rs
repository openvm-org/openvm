use afs_stark_backend::config::StarkGenericConfig;
use datafusion::logical_expr::LogicalPlan;

use self::plan::{Filter, PageScan};

pub mod plan;

#[derive(Clone)]
pub enum AfsLogicalPlan {
    PageScan(PageScan),
    Filter(Filter),
}

impl AfsLogicalPlan {
    pub fn new(logical_plan: LogicalPlan) -> Self {
        todo!()
    }
}
