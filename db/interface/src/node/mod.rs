use std::{
    fmt::{self, Debug},
    sync::Arc,
};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    prover::types::Proof,
};
use async_trait::async_trait;
use ax_sdk::engine::StarkEngine;
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::LogicalPlan};
use enum_dispatch::enum_dispatch;
use futures::lock::Mutex;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};

use self::{filter::Filter, page_scan::PageScan, projection::Projection};
use crate::common::{cryptographic_object::CryptographicObject, expr::AxdbExpr};

pub mod filter;
pub mod functionality;
pub mod page_scan;
pub mod projection;

/// The AxdbNodeExecutable trait defines the methods that each AxdbNode type must implement.
/// These methods handle the execution, key generation, proof generation, verification of the node,
/// and provide access to the values in the node.
#[async_trait]
#[enum_dispatch]
pub trait AxdbNodeExecutable<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
    /// Generate the proving key for the node
    async fn keygen(&mut self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Runs the node's execution logic without any cryptographic operations
    async fn execute(&mut self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Geenrate the STARK proof for the node
    async fn prove(&mut self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Verify the STARK proof for the node
    async fn verify(&self, ctx: &SessionContext, engine: &E) -> Result<()>;
    /// Get the output of the node
    fn output(&self) -> &Option<CryptographicObject<SC>>;
    /// Get the proof of the node
    fn proof(&self) -> &Option<Proof<SC>>;
    /// Get the string name of the node
    fn name(&self) -> &str;
}

/// AxdbNode is a wrapper around the node types that conform to the AxdbNodeExecutable trait.
/// It provides conversion from DataFusion's LogicalPlan to the AxdbNode type. AxdbNodes are
/// meant to be executed by the AxdbController engine. They store the necessary information to handle
/// the cryptographic operations for each type of AxdbNode operation.
#[enum_dispatch(AxdbNodeExecutable<SC, E>)]
pub enum AxdbNode<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
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
    pub fn new(logical_plan: &LogicalPlan, inputs: Vec<Arc<Mutex<AxdbNode<SC, E>>>>) -> Self {
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
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> Debug for AxdbNode<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AxdbNode::PageScan(page_scan) => {
                write!(f, "PageScan {:?}", page_scan.table_name)
            }
            AxdbNode::Projection(projection) => {
                write!(f, "Projection {:?}", projection.schema)
            }
            AxdbNode::Filter(filter) => {
                write!(f, "Filter {:?}", filter.predicate)
            }
        }
    }
}
