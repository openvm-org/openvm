use std::sync::{atomic::AtomicU32, Arc};

use afs_stark_backend::{
    config::Com,
    prover::trace::{ProverTraceData, TraceCommitter},
};
use p3_field::AbstractField;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_uni_stark::{StarkGenericConfig, Val};

use super::PageAccessByRowIdAir;

#[cfg(test)]
pub mod tests;

pub mod trace;

pub struct PageController<SC: StarkGenericConfig> {
    pub page_access_air: PageAccessByRowIdAir,
    request_count: Vec<Arc<AtomicU32>>,
    page_trace: Option<DenseMatrix<Val<SC>>>,
    page_commitment: Option<Com<SC>>,
}

impl<SC: StarkGenericConfig> PageController<SC>
where
    Val<SC>: AbstractField,
{
    pub fn new(bus_index: usize, page_width: usize) -> Self {
        PageController {
            page_access_air: PageAccessByRowIdAir::new(bus_index, page_width),
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

        self.page_access_air =
            PageAccessByRowIdAir::new(self.page_access_air.bus_index(), page_width);

        let page_width = self.page_access_air.page_width();
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
