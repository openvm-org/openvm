use std::{any::Any, result, sync::Arc};

use afs_stark_backend::config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val};
use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::SchemaRef,
    datasource::{TableProvider, TableType},
    error::DataFusionError,
    execution::context::SessionState,
    logical_expr::Expr,
    physical_plan::ExecutionPlan,
};
use p3_field::PrimeField64;
use serde::{de::DeserializeOwned, Serialize};

use super::{execution_plan::CommittedPageExec, CommittedPage};

pub type Result<T, E = DataFusionError> = result::Result<T, E>;

#[async_trait]
impl<SC: StarkGenericConfig + 'static> TableProvider for CommittedPage<SC>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(self.schema.clone())
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &SessionState,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let exec = CommittedPageExec::new(self.page.clone(), self.schema.clone());
        Ok(Arc::new(exec))
    }
}
