use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_field::AbstractField;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{final_page_chip::FinalPageChip, offline_checker::OfflineChecker, page_chip::PageChip};

#[derive(PartialEq, Clone, Debug)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub clk: usize,
    pub idx: Vec<u32>,
    pub data: Vec<u32>,
    pub op_type: OpType,
}

impl Operation {
    pub fn new(clk: usize, idx: Vec<u32>, data: Vec<u32>, op_type: OpType) -> Self {
        Self {
            clk,
            idx,
            data,
            op_type,
        }
    }
}

pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub init_chip: PageChip,
    pub offline_checker: OfflineChecker,
    pub final_chip: FinalPageChip,

    init_chip_trace: Option<DenseMatrix<Val<SC>>>,
    pub offline_checker_trace: Option<DenseMatrix<Val<SC>>>,
    final_chip_trace: Option<DenseMatrix<Val<SC>>>,

    init_page_commitment: Option<Com<SC>>,
    final_page_commitment: Option<Com<SC>>,
}

impl<SC: StarkGenericConfig> PageController<SC> {
    pub fn new(
        page_bus_index: usize,
        sorted_bus_idx: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
    ) -> Self {
        Self {
            init_chip: PageChip::new(page_bus_index, idx_len, data_len),
            offline_checker: OfflineChecker::new(page_bus_index, idx_len, data_len),
            final_chip: FinalPageChip::new(
                page_bus_index,
                sorted_bus_idx,
                idx_len,
                data_len,
                idx_limb_bits,
                idx_decomp,
            ),

            init_chip_trace: None,
            offline_checker_trace: None,
            final_chip_trace: None,

            init_page_commitment: None,
            final_page_commitment: None,
        }
    }

    pub fn offline_checker_trace(&self) -> DenseMatrix<Val<SC>> {
        self.offline_checker_trace.clone().unwrap()
    }

    fn get_page_trace(&self, page: Vec<Vec<u32>>) -> DenseMatrix<Val<SC>> {
        self.init_chip.generate_trace::<SC>(page)
    }

    fn gen_ops_trace(
        &self,
        page: &mut Vec<Vec<u32>>,
        ops: &[Operation],
        trace_degree: usize,
    ) -> RowMajorMatrix<Val<SC>> {
        self.offline_checker
            .generate_trace::<SC>(page, ops.to_owned(), trace_degree)
    }

    pub fn load_page_and_ops(
        &mut self,
        mut page: Vec<Vec<u32>>,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
        ops: Vec<Operation>,
        trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (Vec<DenseMatrix<Val<SC>>>, Vec<ProverTraceData<SC>>) {
        assert!(!page.is_empty());
        self.init_chip_trace = Some(self.get_page_trace(page.clone()));

        // TODO: make bus_index public
        let page_bus_index = self.init_chip.bus_index();
        let sorted_bus_index = self.final_chip.sorted_bus_index;

        self.init_chip = PageChip::new(page_bus_index, idx_len, data_len);
        self.init_chip_trace = Some(self.get_page_trace(page.clone()));

        self.offline_checker = OfflineChecker::new(page_bus_index, idx_len, data_len);
        self.offline_checker_trace = Some(self.gen_ops_trace(&mut page, &ops, trace_degree));

        self.final_chip = FinalPageChip::new(
            page_bus_index,
            sorted_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        self.final_chip_trace = Some(self.get_page_trace(page.clone()));

        let prover_data = vec![
            trace_committer.commit(vec![self.init_chip_trace.clone().unwrap()]),
            trace_committer.commit(vec![self.final_chip_trace.clone().unwrap()]),
        ];

        self.init_page_commitment = Some(prover_data[0].commit.clone());
        self.final_page_commitment = Some(prover_data[1].commit.clone());

        tracing::debug!(
            "heights of all traces: {} {} {}",
            self.init_chip_trace.as_ref().unwrap().height(),
            self.offline_checker_trace.as_ref().unwrap().height(),
            self.final_chip_trace.as_ref().unwrap().height()
        );

        (
            vec![
                self.init_chip_trace.clone().unwrap(),
                self.final_chip_trace.clone().unwrap(),
            ],
            prover_data,
        )
    }
}
