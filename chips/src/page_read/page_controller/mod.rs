use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_field::AbstractField;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use std::sync::{atomic::AtomicU32, Arc};

use super::PageReadAir;

#[cfg(test)]
pub mod tests;

pub mod trace;

pub struct PageController<SC: StarkGenericConfig> {
    pub page_read_air: PageReadAir,
    request_count: Vec<Arc<AtomicU32>>,
    page_trace: Option<DenseMatrix<Val<SC>>>,
    page_commitment: Option<Com<SC>>,
}

impl<SC: StarkGenericConfig> PageController<SC>
where
    Val<SC>: AbstractField,
{
    pub fn new(bus_index: usize) -> Self {
        PageController {
            page_read_air: PageReadAir::new(bus_index, 0, 0),
            request_count: vec![],
            page_trace: None,
            page_commitment: None,
        }
    }

    pub fn load_page(
        &mut self,
        trace_committer: &mut TraceCommitter<SC>,
        page: Vec<Vec<u32>>,
    ) -> (DenseMatrix<Val<SC>>, ProverTraceData<SC>) {
        let page_height = page.len();
        assert!(page_height > 0);
        let page_width = page[0].len();

        self.page_read_air =
            PageReadAir::new(self.page_read_air.bus_index(), page_width, page_height);

        let page_height = self.page_read_air.page_height();
        let page_width = self.page_read_air.page_width();
        self.request_count = (0..page_height)
            .map(|_| Arc::new(AtomicU32::new(0)))
            .collect();

        tracing::debug!("here: {:?}, {:?}", page_height, page_width);
        tracing::debug!("page: {:?}", page);

        self.page_trace = Some(RowMajorMatrix::new(
            page.clone()
                .into_iter()
                .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                .collect(),
            page_width,
        ));

        let prover_data = trace_committer.commit(vec![self.page_trace.clone().unwrap()]);
        self.page_commitment = Some(prover_data.commit.clone());

        (self.page_trace.clone().unwrap(), prover_data)
    }

    pub fn request(&self, page_index: usize) {
        assert!(page_index < self.request_count.len());
        self.request_count[page_index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
