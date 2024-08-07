use std::sync::Arc;

use datafusion::logical_expr::TableSource;

#[derive(Clone)]
pub struct PageScan {
    pub page_id: String,
    pub source: Arc<dyn TableSource>,
}

#[derive(Debug, Clone)]
pub struct Filter {}
