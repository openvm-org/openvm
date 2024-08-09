use std::sync::Arc;

use afs_page::common::page::Page;
use afs_stark_backend::config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val};
use datafusion::{
    arrow::{array::RecordBatch, datatypes::Schema},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        memory::MemoryStream, DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan,
        Partitioning, PlanProperties,
    },
};
use p3_field::PrimeField64;
use serde::{de::DeserializeOwned, Serialize};

use super::{utils::convert_to_record_batch, CommittedPage};

pub struct CommittedPageExec {
    // pub committed_page: Arc<CommittedPage<SC>>,
    pub page: Page,
    pub schema: Schema,
    // metrics: ExecutionPlanMetricsSet,
    properties: PlanProperties,
}

impl CommittedPageExec {
    // pub fn new(committed_page: Arc<CommittedPage<SC>>) -> Self {
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

impl ExecutionPlan for CommittedPageExec {
    fn name(&self) -> &str {
        "CommittedPageExec"
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
        children: Vec<std::sync::Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<std::sync::Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        let record_batch = convert_to_record_batch(self.page.clone(), self.schema.clone());
        // let committed_page = self.committed_page.clone();
        // let schema = self.committed_page.schema.clone();
        // let record_batch: RecordBatch = committed_page.to_record_batch();
        Ok(Box::pin(MemoryStream::try_new(
            vec![record_batch],
            Arc::new(self.schema.clone()),
            None,
        )?))
    }
}

impl std::fmt::Debug for CommittedPageExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommittedPageExec").finish_non_exhaustive()
    }
}

impl DisplayAs for CommittedPageExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                // write!(
                //     f,
                //     "StreamingTableExec: partition_sizes={:?}",
                //     self.partitions.len(),
                // )?;
                // if !self.projected_schema.fields().is_empty() {
                //     write!(
                //         f,
                //         ", projection={}",
                //         ProjectSchemaDisplay(&self.projected_schema)
                //     )?;
                // }
                // if self.infinite {
                //     write!(f, ", infinite_source=true")?;
                // }
                // if let Some(fetch) = self.limit {
                //     write!(f, ", fetch={fetch}")?;
                // }

                // display_orderings(f, &self.projected_output_ordering)?;

                Ok(())
            }
        }
    }
}
