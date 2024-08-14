use std::{
    fmt::{self, Debug},
    sync::Arc,
};

use afs_stark_backend::config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val};
use afs_test_utils::engine::StarkEngine;
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::LogicalPlan};
use p3_field::PrimeField64;
use serde::{de::DeserializeOwned, Serialize};

use self::{filter::Filter, page_scan::PageScan};
use crate::{afs_exec::ChildrenContainer, afs_expr::AfsExpr, committed_page::CommittedPage};

pub mod filter;
pub mod page_scan;

macro_rules! delegate_to_node {
    ($self:ident, $method:ident, $ctx:expr) => {
        match $self {
            AfsNode::PageScan(ref mut page_scan) => page_scan.$method($ctx).await,
            AfsNode::Filter(ref mut filter) => filter.$method($ctx).await,
        }
    };
    ($self:ident, $method:ident) => {
        match $self {
            AfsNode::PageScan(page_scan) => page_scan.$method(),
            AfsNode::Filter(filter) => filter.$method(),
        }
    };
}

pub trait AfsNodeExecutable<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    /// Runs the node's execution logic without any cryptographic operations
    async fn execute(&mut self, ctx: &SessionContext) -> Result<()>;
    /// Generate the proving key for the node
    async fn keygen(&mut self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Geenrate the STARK proof for the node
    async fn prove(&mut self, ctx: &SessionContext) -> Result<()>;
    /// Verify the STARK proof for the node
    async fn verify(&self, ctx: &SessionContext) -> Result<()>;
    /// Get the output of the node
    fn output(&self) -> Option<Arc<CommittedPage<SC>>>;
}

/// AfsNode is a wrapper around the node types that conform to the AfsNodeExecutable trait.
/// It provides conversion from DataFusion's LogicalPlan to the AfsNode type. AfsNodes are
/// meant to be executed by the AfsExec engine. They store the necessary information to handle
/// the cryptographic operations for each type of AfsNode operation.
pub enum AfsNode<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    PageScan(PageScan<SC, E>),
    Filter(Filter<SC, E>),
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> AfsNode<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    /// Converts a LogicalPlan tree to a flat AfsNode vec. Some LogicalPlan nodes may convert to
    /// multiple AfsNodes.
    pub fn from(logical_plan: &LogicalPlan, children: ChildrenContainer<SC, E>) -> Self {
        match logical_plan {
            LogicalPlan::TableScan(table_scan) => {
                let page_id = table_scan.table_name.to_string();
                let source = table_scan.source.clone();
                AfsNode::PageScan(PageScan::new(page_id, source))
            }
            LogicalPlan::Filter(filter) => {
                let afs_expr = AfsExpr::from(&filter.predicate);
                let input = match children {
                    ChildrenContainer::One(child) => {
                        let first_element = Arc::get_mut(&mut Arc::clone(&child))
                            .expect("No other references exist")
                            .remove(0);
                        Arc::new(first_element)
                    }
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

    /// Get the inputs to the node as a vector from left to right.
    pub fn inputs(&self) -> Vec<&Arc<AfsNode<SC, E>>> {
        match self {
            AfsNode::PageScan(_) => vec![],
            AfsNode::Filter(filter) => vec![&filter.input],
        }
    }

    pub async fn execute(&mut self, ctx: &SessionContext) -> Result<()> {
        delegate_to_node!(self, execute, ctx)
    }

    pub async fn keygen(&mut self, ctx: &SessionContext, engine: &E) -> Result<()> {
        // delegate_to_node!(self, keygen, ctx)
        match self {
            AfsNode::PageScan(page_scan) => page_scan.keygen(ctx, engine).await,
            AfsNode::Filter(filter) => filter.keygen(ctx, engine).await,
        }
    }

    pub async fn prove(&mut self, ctx: &SessionContext) -> Result<()> {
        delegate_to_node!(self, prove, ctx)
    }

    pub async fn verify(&mut self, ctx: &SessionContext) -> Result<()> {
        delegate_to_node!(self, verify, ctx)
    }

    pub fn output(&self) -> Option<Arc<CommittedPage<SC>>> {
        delegate_to_node!(self, output)
    }
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> Debug for AfsNode<SC, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AfsNode::PageScan(page_scan) => write!(f, "PageScan {:?}", page_scan.page_id),
            AfsNode::Filter(filter) => write!(f, "Filter {:?}", filter.predicate),
        }
    }
}
