use crate::page_read::PageReadChip;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_field::AbstractField;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use parking_lot::Mutex;
use std::sync::{atomic::AtomicU32, Arc};

#[cfg(test)]
pub mod tests;

pub mod trace;

pub struct PageController<SC: StarkGenericConfig> {
    pub page_read_chip: Mutex<PageReadChip>,
    request_count: Vec<Arc<AtomicU32>>,
    page_trace: DenseMatrix<Val<SC>>,
    page_commitment: ProverTraceData<SC>,
}

impl<SC: StarkGenericConfig> PageController<SC>
where
    Val<SC>: AbstractField,
{
    pub fn new(bus_index: usize) -> Self {
        PageController {
            page_read_chip: Mutex::new(PageReadChip::new(bus_index, vec![vec![]])),
            request_count: vec![],
            page_trace: DenseMatrix::new_col(vec![]),
            page_commitment: ProverTraceData::new(),
        }
    }

    pub fn load_page(&mut self, trace_committer: &mut TraceCommitter<SC>, page: Vec<Vec<u32>>) {
        let mut page_read_chip_locked = self.page_read_chip.lock();

        *page_read_chip_locked = PageReadChip::new(page_read_chip_locked.bus_index(), page.clone());

        let page_height = page_read_chip_locked.page_height();
        let page_width = page_read_chip_locked.page_width();
        self.request_count = (0..page_height)
            .map(|_| Arc::new(AtomicU32::new(0)))
            .collect();

        self.page_trace = RowMajorMatrix::new(
            page.clone()
                .into_iter()
                .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                .collect(),
            page_width,
        );

        self.page_commitment = trace_committer.commit(vec![self.page_trace.clone()]);
    }

    pub fn get_page_commitment(&self) -> ProverTraceData<SC> {
        self.page_commitment.clone()
    }

    pub fn request(&self, page_index: usize) {
        assert!(page_index < self.request_count.len());
        self.request_count[page_index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
