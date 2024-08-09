use std::sync::Arc;

use datafusion::logical_expr::TableSource;

use super::AfsLogicalPlan;
use crate::afs_expr::AfsExpr;

#[derive(Clone)]
pub struct PageScan {
    pub page_id: String,
    pub input: Arc<dyn TableSource>,
    pub output: Option<Arc<dyn TableSource>>,
}

#[derive(Clone)]
pub struct Filter {
    pub predicate: AfsExpr,
    pub input: Arc<AfsLogicalPlan>,
    pub output: Option<Arc<dyn TableSource>>,
}
