use afs_page::common::page::Page;
use afs_stark_backend::{config::StarkGenericConfig, prover::trace::ProverTraceData};
use datafusion::arrow::datatypes::Schema;

pub mod table_provider;

#[derive(Clone)]
pub struct CommittedPage<SC: StarkGenericConfig> {
    pub page_id: String,
    pub schema: Schema,
    pub page: Page,
    pub cached_trace: Option<ProverTraceData<SC>>,
}

impl<SC: StarkGenericConfig> CommittedPage<SC> {
    pub fn new(
        page_id: String,
        schema: Schema,
        page: Page,
        cached_trace: Option<ProverTraceData<SC>>,
    ) -> Self {
        Self {
            page_id,
            schema,
            page,
            cached_trace,
        }
    }

    pub fn write_cached_trace(&mut self, trace: ProverTraceData<SC>) {
        self.cached_trace = Some(trace);
    }
}
