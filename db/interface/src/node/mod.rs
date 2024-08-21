use std::{
    fmt::{self, Debug},
    sync::Arc,
};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    prover::types::Proof,
};
use afs_test_utils::engine::StarkEngine;
use async_trait::async_trait;
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::LogicalPlan};
use futures::lock::Mutex;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};

use self::{filter::Filter, page_scan::PageScan, projection::Projection};
use crate::{committed_page::CommittedPage, expr::AxdbExpr};

pub mod filter;
pub mod functionality;
pub mod page_scan;
pub mod projection;

macro_rules! delegate_to_node {
    ($self:ident, $method:ident, $ctx:expr, $engine:expr) => {
        match $self {
            AxdbNode::PageScan(ref mut page_scan) => page_scan.$method($ctx, $engine).await,
            AxdbNode::Projection(ref mut projection) => projection.$method($ctx, $engine).await,
            AxdbNode::Filter(ref mut filter) => filter.$method($ctx, $engine).await,
        }
    };
    ($self:ident, $method:ident) => {
        match $self {
            AxdbNode::PageScan(page_scan) => page_scan.$method(),
            AxdbNode::Projection(projection) => projection.$method(),
            AxdbNode::Filter(filter) => filter.$method(),
        }
    };
}

#[async_trait]
pub trait AxdbNodeExecutable<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
    /// Runs the node's execution logic without any cryptographic operations
    async fn execute(&mut self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Generate the proving key for the node
    async fn keygen(&mut self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Geenrate the STARK proof for the node
    async fn prove(&mut self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Verify the STARK proof for the node
    async fn verify(&self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Get the output of the node
    fn output(&self) -> &Option<CommittedPage<SC>>;
    /// Get the proof of the node
    fn proof(&self) -> &Option<Proof<SC>>;
    /// Get the string name of the node
    fn name(&self) -> &str;
}

/// AxdbNode is a wrapper around the node types that conform to the AxdbNodeExecutable trait.
/// It provides conversion from DataFusion's LogicalPlan to the AxdbNode type. AxdbNodes are
/// meant to be executed by the AxdbExec engine. They store the necessary information to handle
/// the cryptographic operations for each type of AxdbNode operation.
pub enum AxdbNode<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
    PageScan(PageScan<SC, E>),
    Projection(Projection<SC, E>),
    Filter(Filter<SC, E>),
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> AxdbNode<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    /// Converts a LogicalPlan tree to a flat AxdbNode vec. Some LogicalPlan nodes may convert to
    /// multiple AxdbNodes.
    pub fn from(logical_plan: &LogicalPlan, inputs: Vec<Arc<Mutex<AxdbNode<SC, E>>>>) -> Self {
        match logical_plan {
            LogicalPlan::TableScan(table_scan) => {
                let table_name = table_scan.table_name.to_string();
                let source = table_scan.source.clone();
                let filters = table_scan.filters.iter().map(AxdbExpr::from).collect();
                let projection = table_scan.projection.clone();
                AxdbNode::PageScan(PageScan::new(table_name, source, filters, projection))
            }
            LogicalPlan::Filter(filter) => {
                if inputs.len() != 1 {
                    panic!("Filter node expects exactly one input");
                }
                let afs_expr = AxdbExpr::from(&filter.predicate);
                let input = inputs[0].clone();
                AxdbNode::Filter(Filter {
                    input,
                    output: None,
                    predicate: afs_expr,
                    pk: None,
                    proof: None,
                })
            }
            _ => panic!("Invalid node type: {:?}", logical_plan),
        }
    }

    /// Get the inputs to the node as a vector from left to right.
    pub fn inputs(&self) -> Vec<&Arc<Mutex<AxdbNode<SC, E>>>> {
        match self {
            AxdbNode::PageScan(_) => vec![],
            AxdbNode::Projection(projection) => vec![&projection.input],
            AxdbNode::Filter(filter) => vec![&filter.input],
        }
    }

    pub async fn execute(&mut self, ctx: &SessionContext, engine: &E) -> Result<()> {
        delegate_to_node!(self, execute, ctx, engine)
    }

    pub async fn keygen(&mut self, ctx: &SessionContext, engine: &E) -> Result<()> {
        delegate_to_node!(self, keygen, ctx, engine)
    }

    pub async fn prove(&mut self, ctx: &SessionContext, engine: &E) -> Result<()> {
        delegate_to_node!(self, prove, ctx, engine)
    }

    pub async fn verify(&mut self, ctx: &SessionContext, engine: &E) -> Result<()> {
        delegate_to_node!(self, verify, ctx, engine)
    }

    pub fn output(&self) -> &Option<CommittedPage<SC>> {
        delegate_to_node!(self, output)
    }

    pub fn name(&self) -> &str {
        delegate_to_node!(self, name)
    }
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> Debug for AxdbNode<SC, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AxdbNode::PageScan(page_scan) => {
                write!(
                    f,
                    "PageScan {:?} {:?}",
                    page_scan.table_name, page_scan.output
                )
            }
            AxdbNode::Projection(projection) => {
                write!(
                    f,
                    "Projection {:?} {:?}",
                    projection.schema, projection.output
                )
            }
            AxdbNode::Filter(filter) => {
                write!(f, "Filter {:?} {:?}", filter.predicate, filter.output)
            }
        }
    }
}
