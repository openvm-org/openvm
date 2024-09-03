use std::{any::Any, sync::Arc};

use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::{Schema, SchemaRef},
    datasource::{TableProvider, TableType},
    error::DataFusionError,
    execution::context::SessionState,
    physical_plan::ExecutionPlan,
    prelude::Expr,
};
use enum_dispatch::enum_dispatch;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use super::{committed_page::CommittedPage, cryptographic_schema::CryptographicSchema};

pub type Result<T, E = DataFusionError> = std::result::Result<T, E>;

#[enum_dispatch]
pub trait CryptographicObjectTrait {
    fn schema(&self) -> Schema;
}

#[derive(Debug, Clone)]
#[enum_dispatch(CryptographicObjectTrait)]
pub enum CryptographicObject<SC: StarkGenericConfig> {
    CommittedPage(CommittedPage<SC>),
    CryptographicSchema(CryptographicSchema),
}

#[async_trait]
impl<SC: StarkGenericConfig + 'static> TableProvider for CryptographicObject<SC>
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
        let schema = self.schema();
        Arc::new(schema.clone())
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
