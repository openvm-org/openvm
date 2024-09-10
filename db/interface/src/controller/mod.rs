use std::sync::Arc;

use afs_stark_backend::config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val};
use ax_sdk::engine::StarkEngine;
use datafusion::{
    arrow::array::RecordBatch, error::Result, execution::context::SessionContext,
    logical_expr::LogicalPlan,
};
use futures::lock::Mutex;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info;

use crate::{
    common::cryptographic_object::CryptographicObject,
    node::{AxdbNode, AxdbNodeExecutable},
};

macro_rules! run_execution_plan {
    ($self:ident, $method:ident, $ctx:expr, $engine:expr) => {
        for node in &mut $self.axdb_execution_plan {
            let mut node = node.lock().await;
            (&mut node).$method(&$self.ctx, &$self.engine).await?;
        }
    };
}

pub struct AxdbController<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    /// The session context from DataFusion
    pub ctx: SessionContext,
    /// STARK engine used for cryptographic operations
    pub engine: E,
    /// AxdbNode tree flattened into a vec to be executed sequentially
    pub axdb_execution_plan: Vec<Arc<Mutex<AxdbNode<SC, E>>>>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> AxdbController<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub async fn new(ctx: SessionContext, root: LogicalPlan, engine: E) -> Self {
        let root = ctx.state().optimize(&root).unwrap();
        let axdb_execution_plan = Self::create_execution_plan(root).await.unwrap();
        Self {
            ctx,
            engine,
            axdb_execution_plan,
        }
    }

    pub fn last_node(&self) -> Arc<Mutex<AxdbNode<SC, E>>> {
        self.axdb_execution_plan.last().unwrap().to_owned()
    }

    pub async fn output(&self) -> Result<RecordBatch> {
        let last_node = self.axdb_execution_plan.last().unwrap().to_owned();
        let last_node = last_node.lock().await;
        let output = last_node.output().as_ref().unwrap();
        match output {
            CryptographicObject::CommittedPage(cp) => Ok(cp.to_record_batch()),
            _ => panic!("output is not a CommittedPage"),
        }
    }

    pub async fn execute(&mut self) -> Result<()> {
        run_execution_plan!(self, execute, ctx, engine);
        Ok(())
    }

    pub async fn keygen(&mut self) -> Result<()> {
        run_execution_plan!(self, keygen, ctx, engine);
        Ok(())
    }

    pub async fn prove(&mut self) -> Result<()> {
        run_execution_plan!(self, prove, ctx, engine);
        Ok(())
    }

    pub async fn verify(&mut self) -> Result<()> {
        run_execution_plan!(self, verify, ctx, engine);
        Ok(())
    }

    /// Creates the flattened execution plan from a LogicalPlan tree root node.
    async fn create_execution_plan(root: LogicalPlan) -> Result<Vec<Arc<Mutex<AxdbNode<SC, E>>>>> {
        let mut flattened = vec![];
        Self::flatten_logical_plan_tree(&mut flattened, &root).await?;
        Ok(flattened)
    }

    /// Converts a LogicalPlan tree to a flat AxdbNode vec. Starts from the root (output) node and works backwards until it reaches the input(s).
    async fn flatten_logical_plan_tree(
        flattened: &mut Vec<Arc<Mutex<AxdbNode<SC, E>>>>,
        root: &LogicalPlan,
    ) -> Result<usize> {
        info!("flatten_logical_plan_tree {:?}", root);
        let current_index = flattened.len();

        let inputs = root.inputs();

        if inputs.is_empty() {
            let afs_node = Arc::new(Mutex::new(AxdbNode::new(root, vec![])));
            flattened.push(afs_node);
        } else {
            let mut input_indexes = vec![];
            for &input in inputs.iter() {
                let input_index =
                    Box::pin(Self::flatten_logical_plan_tree(flattened, input)).await?;
                input_indexes.push(input_index);
            }

            let input_pointers = input_indexes
                .iter()
                .map(|i| Arc::clone(&flattened[*i]))
                .collect::<Vec<Arc<Mutex<AxdbNode<SC, E>>>>>();
            let afs_node = Arc::new(Mutex::new(AxdbNode::new(root, input_pointers)));
            flattened.push(afs_node);
        }

        Ok(current_index)
    }
}
