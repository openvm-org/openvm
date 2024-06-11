use std::sync::Arc;

use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_field::{AbstractField, Field, PrimeField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{final_page::FinalPageAir, offline_checker::OfflineChecker, page::PageAir};
use crate::range_gate::RangeCheckerGateChip;

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
    pub init_chip: PageAir,
    pub offline_checker: OfflineChecker,
    pub final_chip: FinalPageAir,

    init_chip_trace: Option<DenseMatrix<Val<SC>>>,
    offline_checker_trace: Option<DenseMatrix<Val<SC>>>,
    final_chip_trace: Option<DenseMatrix<Val<SC>>>,
    final_page_aux_trace: Option<DenseMatrix<Val<SC>>>,

    init_page_commitment: Option<Com<SC>>,
    final_page_commitment: Option<Com<SC>>,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> PageController<SC> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_bus_index: usize,
        range_bus_index: usize,
        ops_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
    ) -> Self
    where
        Val<SC>: Field,
    {
        Self {
            init_chip: PageAir::new(page_bus_index, idx_len, data_len),
            offline_checker: OfflineChecker::new(
                page_bus_index,
                range_bus_index,
                ops_bus_index,
                idx_len,
                data_len,
                idx_limb_bits,
                Val::<SC>::bits() - 1,
                idx_decomp,
            ),
            final_chip: FinalPageAir::new(
                page_bus_index,
                range_bus_index,
                idx_len,
                data_len,
                idx_limb_bits,
                idx_decomp,
            ),

            init_chip_trace: None,
            offline_checker_trace: None,
            final_chip_trace: None,
            final_page_aux_trace: None,

            init_page_commitment: None,
            final_page_commitment: None,

            range_checker: Arc::new(RangeCheckerGateChip::new(range_bus_index, 1 << idx_decomp)),
        }
    }

    pub fn offline_checker_trace(&self) -> DenseMatrix<Val<SC>> {
        self.offline_checker_trace.clone().unwrap()
    }

    pub fn final_page_aux_trace(&self) -> DenseMatrix<Val<SC>> {
        self.final_page_aux_trace.clone().unwrap()
    }

    pub fn range_checker_trace(&self) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.range_checker.generate_trace()
    }

    fn get_page_trace(&self, page: Vec<Vec<u32>>) -> DenseMatrix<Val<SC>> {
        self.init_chip.generate_trace::<SC>(page)
    }

    fn gen_ops_trace(
        &self,
        page: &mut Vec<Vec<u32>>,
        ops: &[Operation],
        range_checker: Arc<RangeCheckerGateChip>,
        trace_degree: usize,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.offline_checker
            .generate_trace::<SC>(page, ops.to_owned(), range_checker, trace_degree)
    }

    pub fn update_range_checker(&mut self, idx_decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.bus_index(),
            1 << idx_decomp,
        ));
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> (Vec<DenseMatrix<Val<SC>>>, Vec<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        // idx_decomp can't change between different pages since range_checker depends on it
        assert!(1 << idx_decomp == self.range_checker.range_max());

        assert!(!page.is_empty());
        self.init_chip_trace = Some(self.get_page_trace(page.clone()));

        let page_bus_index = self.offline_checker.page_bus_index;
        let range_bus_index = self.offline_checker.range_bus_index;
        let ops_bus_index = self.offline_checker.ops_bus_index;

        self.init_chip = PageAir::new(page_bus_index, idx_len, data_len);
        self.init_chip_trace = Some(self.get_page_trace(page.clone()));

        self.offline_checker = OfflineChecker::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            Val::<SC>::bits() - 1,
            idx_decomp,
        );
        self.offline_checker_trace =
            Some(self.gen_ops_trace(&mut page, &ops, self.range_checker.clone(), trace_degree));

        // Sorting the page by (1-is_alloc, idx)
        page.sort_by_key(|row| (1 - row[0], row[1..1 + idx_len].to_vec()));

        // HashSet of all indices used in operations
        let internal_indices = ops.iter().map(|op| op.idx.clone()).collect();

        self.final_chip = FinalPageAir::new(
            page_bus_index,
            range_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        self.final_chip_trace = Some(self.get_page_trace(page.clone()));
        self.final_page_aux_trace = Some(self.final_chip.gen_aux_trace::<SC>(
            page.clone(),
            self.range_checker.clone(),
            internal_indices,
        ));

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
