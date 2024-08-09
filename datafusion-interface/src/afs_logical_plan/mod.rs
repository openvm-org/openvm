use std::{
    fmt::{self, Debug},
    sync::Arc,
};

use datafusion::logical_expr::LogicalPlan;

use self::plan::{Filter, PageScan};
use crate::afs_expr::AfsExpr;

pub mod plan;

#[derive(Clone)]
pub enum AfsLogicalPlan {
    PageScan(PageScan),
    Filter(Filter),
}

impl AfsLogicalPlan {
    pub fn from(logical_plan: &LogicalPlan, children: Vec<Arc<AfsLogicalPlan>>) -> Self {
        match logical_plan {
            LogicalPlan::TableScan(table_scan) => {
                let page_id = table_scan.table_name.to_string();
                let source = table_scan.source.clone();
                AfsLogicalPlan::PageScan(PageScan {
                    page_id,
                    input: source,
                    output: None,
                })
            }
            LogicalPlan::Filter(filter) => {
                let afs_expr = AfsExpr::from(&filter.predicate);
                let input = children.into_iter().next().unwrap();
                AfsLogicalPlan::Filter(Filter {
                    predicate: afs_expr,
                    input,
                    output: None,
                })
            }
            _ => panic!("Invalid node type: {:?}", logical_plan),
        }
    }

    pub fn inputs(&self) -> Vec<&Arc<AfsLogicalPlan>> {
        match self {
            AfsLogicalPlan::PageScan(_) => vec![],
            AfsLogicalPlan::Filter(filter) => vec![&filter.input],
        }
    }
}

impl Debug for AfsLogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AfsLogicalPlan::PageScan(_) => write!(f, "PageScan"),
            AfsLogicalPlan::Filter(_) => write!(f, "Filter"),
        }
    }
}
