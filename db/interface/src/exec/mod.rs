use std::sync::Arc;

use afs_stark_backend::config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val};
use afs_test_utils::engine::StarkEngine;
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::LogicalPlan};
use futures::lock::Mutex;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};

use crate::node::AxiomDbNode;

macro_rules! run_execution_plan {
    ($self:ident, $method:ident, $ctx:expr, $engine:expr) => {
        for node in &mut $self.afs_execution_plan {
            let mut node = node.lock().await;
            node.$method(&$self.ctx, &$self.engine).await?;
        }
    };
}

pub struct AxiomDbExec<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    /// The session context from DataFusion
    pub ctx: SessionContext,
    /// STARK engine used for cryptographic operations
    pub engine: E,
    /// AxiomDbNode tree flattened into a vec to be executed sequentially
    pub afs_execution_plan: Vec<Arc<Mutex<AxiomDbNode<SC, E>>>>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> AxiomDbExec<SC, E>
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
        let afs_execution_plan = Self::create_execution_plan(root).await.unwrap();
        Self {
            ctx,
            engine,
            afs_execution_plan,
        }
    }

    pub async fn last_node(&self) -> Result<Arc<Mutex<AxiomDbNode<SC, E>>>> {
        let last_node = self.afs_execution_plan.last().unwrap().to_owned();
        Ok(last_node)
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
    async fn create_execution_plan(
        root: LogicalPlan,
    ) -> Result<Vec<Arc<Mutex<AxiomDbNode<SC, E>>>>> {
        let mut flattened = vec![];
        Self::flatten_logical_plan_tree(&mut flattened, &root).await?;
        Ok(flattened)
    }

    /// Converts a LogicalPlan tree to a flat AxiomDbNode vec. Starts from the root (output) node and works backwards until it reaches the input(s).
    async fn flatten_logical_plan_tree(
        flattened: &mut Vec<Arc<Mutex<AxiomDbNode<SC, E>>>>,
        root: &LogicalPlan,
    ) -> Result<usize> {
        println!("flatten_logical_plan_tree {:?}", root);
        let current_index = flattened.len();

        let inputs = root.inputs();

        if inputs.is_empty() {
            let afs_node = Arc::new(Mutex::new(AxiomDbNode::from(root, vec![])));
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
                .collect::<Vec<Arc<Mutex<AxiomDbNode<SC, E>>>>>();
            let afs_node = Arc::new(Mutex::new(AxiomDbNode::from(root, input_pointers)));
            flattened.push(afs_node);
        }

        Ok(current_index)
    }
}
