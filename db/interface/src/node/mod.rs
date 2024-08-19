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
use crate::{committed_page::CommittedPage, expr::AxiomDbExpr};

pub mod filter;
pub mod page_scan;
pub mod projection;

macro_rules! delegate_to_node {
    ($self:ident, $method:ident, $ctx:expr, $engine:expr) => {
        match $self {
            AxiomDbNode::PageScan(ref mut page_scan) => page_scan.$method($ctx, $engine).await,
            AxiomDbNode::Projection(ref mut projection) => projection.$method($ctx, $engine).await,
            AxiomDbNode::Filter(ref mut filter) => filter.$method($ctx, $engine).await,
        }
    };
    ($self:ident, $method:ident) => {
        match $self {
            AxiomDbNode::PageScan(page_scan) => page_scan.$method(),
            AxiomDbNode::Projection(projection) => projection.$method(),
            AxiomDbNode::Filter(filter) => filter.$method(),
        }
    };
}

#[async_trait]
pub trait AxiomDbNodeExecutable<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
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
}

/// AxiomDbNode is a wrapper around the node types that conform to the AxiomDbNodeExecutable trait.
/// It provides conversion from DataFusion's LogicalPlan to the AxiomDbNode type. AxiomDbNodes are
/// meant to be executed by the AxiomDbExec engine. They store the necessary information to handle
/// the cryptographic operations for each type of AxiomDbNode operation.
pub enum AxiomDbNode<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
    PageScan(PageScan<SC, E>),
    Projection(Projection<SC, E>),
    Filter(Filter<SC, E>),
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> AxiomDbNode<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    /// Converts a LogicalPlan tree to a flat AxiomDbNode vec. Some LogicalPlan nodes may convert to
    /// multiple AxiomDbNodes.
    pub fn from(logical_plan: &LogicalPlan, inputs: Vec<Arc<Mutex<AxiomDbNode<SC, E>>>>) -> Self {
        match logical_plan {
            LogicalPlan::TableScan(table_scan) => {
                let table_name = table_scan.table_name.to_string();
                let source = table_scan.source.clone();
                AxiomDbNode::PageScan(PageScan::new(table_name, source))
            }
            LogicalPlan::Filter(filter) => {
                if inputs.len() != 1 {
                    panic!("Filter node expects exactly one input");
                }
                let afs_expr = AxiomDbExpr::from(&filter.predicate);
                let input = inputs[0].clone();
                AxiomDbNode::Filter(Filter {
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
    pub fn inputs(&self) -> Vec<&Arc<Mutex<AxiomDbNode<SC, E>>>> {
        match self {
            AxiomDbNode::PageScan(_) => vec![],
            AxiomDbNode::Projection(projection) => vec![&projection.input],
            AxiomDbNode::Filter(filter) => vec![&filter.input],
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
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> Debug for AxiomDbNode<SC, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AxiomDbNode::PageScan(page_scan) => {
                write!(
                    f,
                    "PageScan {:?} {:?}",
                    page_scan.table_name, page_scan.output
                )
            }
            AxiomDbNode::Projection(projection) => {
                write!(
                    f,
                    "Projection {:?} {:?}",
                    projection.schema, projection.output
                )
            }
            AxiomDbNode::Filter(filter) => {
                write!(f, "Filter {:?} {:?}", filter.predicate, filter.output)
            }
        }
    }
}
