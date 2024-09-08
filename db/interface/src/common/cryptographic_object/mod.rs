use std::{any::Any, fmt::Debug, sync::Arc};

use afs_page::common::page::Page;
use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::{Schema, SchemaRef},
    datasource::{TableProvider, TableType},
    error::DataFusionError,
    execution::context::SessionState,
    physical_expr::EquivalenceProperties,
    physical_plan::{
        memory::MemoryStream, DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan,
        Partitioning, PlanProperties,
    },
    prelude::Expr,
};
use enum_dispatch::enum_dispatch;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use super::{
    committed_page::{utils::convert_to_record_batch, CommittedPage},
    cryptographic_schema::CryptographicSchema,
};

pub type Result<T, E = DataFusionError> = std::result::Result<T, E>;

#[enum_dispatch]
pub trait CryptographicObjectTrait {
    fn schema(&self) -> Schema;
    fn page(&self) -> Page;
}

#[derive(Clone)]
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
        let schema = CryptographicObjectTrait::schema(self);
        Arc::new(schema)
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
        let schema = CryptographicObjectTrait::schema(self);
        let page = CryptographicObjectTrait::page(self);
        let exec = CryptographicObjectExec::new(page, schema);
        Ok(Arc::new(exec))
    }
}

impl<SC: StarkGenericConfig> Debug for CryptographicObject<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CryptographicObject::CommittedPage(page) => {
                write!(f, "CryptographicObject::CommittedPage({:?})", page)
            }
            CryptographicObject::CryptographicSchema(schema) => {
                write!(f, "CryptographicObject::CryptographicSchema({:?})", schema)
            }
        }
    }
}

pub struct CryptographicObjectExec {
    pub page: Page,
    pub schema: Schema,
    properties: PlanProperties,
}

impl CryptographicObjectExec {
    pub fn new(page: Page, schema: Schema) -> Self {
        Self {
            page,
            schema: schema.clone(),
            properties: PlanProperties::new(
                EquivalenceProperties::new(Arc::new(schema)),
                Partitioning::UnknownPartitioning(1),
                ExecutionMode::Bounded,
            ),
        }
    }
}

impl ExecutionPlan for CryptographicObjectExec {
    fn name(&self) -> &str {
        "CryptographicObjectExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&std::sync::Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: std::sync::Arc<Self>,
        _children: Vec<std::sync::Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<std::sync::Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        let record_batch = convert_to_record_batch(self.page.clone(), self.schema.clone());
        Ok(Box::pin(MemoryStream::try_new(
            vec![record_batch],
            Arc::new(self.schema.clone()),
            None,
        )?))
    }
}

impl std::fmt::Debug for CryptographicObjectExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CryptographicObjectExec")
            .finish_non_exhaustive()
    }
}

impl DisplayAs for CryptographicObjectExec {
    fn fmt_as(&self, _t: DisplayFormatType, _f: &mut std::fmt::Formatter) -> std::fmt::Result {
        Ok(())
    }
}
