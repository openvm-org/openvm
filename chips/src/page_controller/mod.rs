use crate::page_read::PageReadChip;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use parking_lot::Mutex;
use std::sync::{atomic::AtomicU32, Arc};

pub mod trace;

pub struct PageController {
    bus_index: usize,
    pub page_read_chip: Mutex<PageReadChip>,
    page_size: usize,
    request_count: Vec<Arc<AtomicU32>>,
}

impl PageController {
    pub fn new(bus_index: usize) -> Self {
        PageController {
            bus_index,
            page_read_chip: Mutex::new(PageReadChip::new(bus_index, vec![vec![]])),
            page_size: 0,
            request_count: vec![],
        }
    }

    fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn load_page<SC: StarkGenericConfig>(
        &mut self,
        trace_committer: &mut TraceCommitter<SC>,
        page: Vec<Vec<u32>>,
    ) -> (DenseMatrix<Val<SC>>, ProverTraceData<SC>) {
        let mut page_read_chip_locked = self.page_read_chip.lock();
        *page_read_chip_locked = PageReadChip::new(self.bus_index(), page.clone());

        self.page_size = page_read_chip_locked.page_size();
        self.request_count = (0..self.page_size)
            .map(|_| Arc::new(AtomicU32::new(0)))
            .collect();

        let page_trace = page_read_chip_locked.get_page_trace::<SC>();
        let commitment = trace_committer.commit(vec![page_trace.clone()]);

        (page_trace, commitment)
    }

    pub fn request(&self, page_index: usize) {
        assert!(page_index < self.page_size);
        self.request_count[page_index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
