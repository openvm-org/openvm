use std::{
    fmt::{self, Debug},
    sync::Arc,
};

use afs_stark_backend::config::StarkGenericConfig;
use datafusion::{error::Result, logical_expr::LogicalPlan};

use self::{filter::Filter, page_scan::PageScan};
use crate::{afs_exec::ChildrenContainer, afs_expr::AfsExpr};

pub mod filter;
pub mod page_scan;

macro_rules! delegate_to_node {
    ($self:ident, $method:ident) => {
        match $self {
            AfsNode::PageScan(ref mut page_scan) => page_scan.$method(),
            AfsNode::Filter(ref mut filter) => filter.$method(),
        }
    };
}

pub trait AfsNodeExecutable<SC: StarkGenericConfig> {
    /// Runs the node's execution logic without any cryptographic operations
    fn execute(&mut self) -> Result<()>;
    /// Generate the proving key for the node
    fn keygen(&mut self) -> Result<()>;
    /// Geenrate the STARK proof for the node
    fn prove(&mut self) -> Result<()>;
    /// Verify the STARK proof for the node
    fn verify(&self) -> Result<()>;
}

pub enum AfsNode<SC: StarkGenericConfig> {
    PageScan(PageScan<SC>),
    Filter(Filter<SC>),
}

impl<SC: StarkGenericConfig> AfsNode<SC> {
    pub fn from(logical_plan: &LogicalPlan, children: ChildrenContainer<SC>) -> Self {
        match logical_plan {
            LogicalPlan::TableScan(table_scan) => {
                let page_id = table_scan.table_name.to_string();
                let source = table_scan.source.clone();
                AfsNode::PageScan(PageScan {
                    page_id,
                    pk: None,
                    input: source,
                    output: None,
                })
            }
            LogicalPlan::Filter(filter) => {
                let afs_expr = AfsExpr::from(&filter.predicate);
                let input = match children {
                    ChildrenContainer::One(child) => child,
                    _ => panic!("Filter node expects exactly one child"),
                };
                AfsNode::Filter(Filter {
                    predicate: afs_expr,
                    pk: None,
                    input,
                    output: None,
                })
            }
            _ => panic!("Invalid node type: {:?}", logical_plan),
        }
    }

    pub fn inputs(&self) -> Vec<&Arc<AfsNode<SC>>> {
        match self {
            AfsNode::PageScan(_) => vec![],
            AfsNode::Filter(filter) => vec![&filter.input],
        }
    }

    pub fn execute(&mut self) -> Result<()> {
        delegate_to_node!(self, execute)
    }

    pub fn keygen(&mut self) -> Result<()> {
        delegate_to_node!(self, keygen)
    }

    pub fn prove(&mut self) -> Result<()> {
        delegate_to_node!(self, prove)
    }

    pub fn verify(&mut self) -> Result<()> {
        delegate_to_node!(self, verify)
    }
}

impl<SC: StarkGenericConfig> Debug for AfsNode<SC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AfsNode::PageScan(page_scan) => write!(f, "PageScan {:?}", page_scan.input.schema()),
            AfsNode::Filter(filter) => write!(f, "Filter {:?}", filter.input),
        }
    }
}
