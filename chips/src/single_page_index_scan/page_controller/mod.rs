use std::sync::Arc;

use afs_stark_backend::{
    config::Com,
    prover::trace::{ProverTraceData, TraceCommitter},
};
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_matrix::dense::DenseMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::range_gate::RangeCheckerGateChip;

use super::{
    page_index_scan_input::PageIndexScanInputChip, page_index_scan_output::PageIndexScanOutputChip,
};

pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub input_chip: PageIndexScanInputChip,
    pub output_chip: PageIndexScanOutputChip,

    input_chip_trace: Option<DenseMatrix<Val<SC>>>,
    input_chip_aux_trace: Option<DenseMatrix<Val<SC>>>,
    output_chip_trace: Option<DenseMatrix<Val<SC>>>,
    output_chip_aux_trace: Option<DenseMatrix<Val<SC>>>,

    input_commitment: Option<Com<SC>>,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> PageController<SC>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub fn new(
        bus_index: usize,
        idx_len: usize,
        data_len: usize,
        range_max: u32,
        idx_limb_bits: Vec<usize>,
        idx_decomp: usize,
    ) -> Self {
        let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, 1 << idx_decomp));
        Self {
            input_chip: PageIndexScanInputChip::new(
                bus_index,
                idx_len,
                data_len,
                range_max,
                idx_limb_bits.clone(),
                idx_decomp,
                range_checker.clone(),
            ),
            output_chip: PageIndexScanOutputChip::new(
                bus_index,
                idx_len,
                data_len,
                range_max,
                idx_limb_bits.clone(),
                idx_decomp,
                range_checker.clone(),
            ),
            input_chip_trace: None,
            input_chip_aux_trace: None,
            output_chip_trace: None,
            output_chip_aux_trace: None,
            input_commitment: None,
            range_checker,
        }
    }

    pub fn input_chip_trace(&self) -> DenseMatrix<Val<SC>> {
        self.input_chip_trace.clone().unwrap()
    }

    pub fn input_chip_aux_trace(&self) -> DenseMatrix<Val<SC>> {
        self.input_chip_aux_trace.clone().unwrap()
    }

    pub fn output_chip_trace(&self) -> DenseMatrix<Val<SC>> {
        self.output_chip_trace.clone().unwrap()
    }

    pub fn output_chip_aux_trace(&self) -> DenseMatrix<Val<SC>> {
        self.output_chip_aux_trace.clone().unwrap()
    }

    pub fn range_checker_trace(&self) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.range_checker.generate_trace()
    }

    pub fn update_range_checker(&mut self, idx_decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.bus_index(),
            1 << idx_decomp,
        ));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_page(
        &mut self,
        page: Vec<Vec<u32>>,
        x: Vec<u32>,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: Vec<usize>,
        idx_decomp: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (Vec<DenseMatrix<Val<SC>>>, Vec<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        // idx_decomp can't change between different pages since range_checker depends on it
        assert!(1 << idx_decomp == self.range_checker.range_max());

        assert!(!page.is_empty());

        let bus_index = self.input_chip.air.bus_index;

        self.input_chip = PageIndexScanInputChip::new(
            bus_index,
            idx_len,
            data_len,
            self.range_checker.range_max(),
            idx_limb_bits.clone(),
            idx_decomp,
            self.range_checker.clone(),
        );
        self.input_chip_trace = Some(self.input_chip.gen_page_trace::<SC>(page.clone()));
        self.input_chip_aux_trace =
            Some(self.input_chip.gen_aux_trace::<SC>(page.clone(), x.clone()));

        self.output_chip = PageIndexScanOutputChip::new(
            bus_index,
            idx_len,
            data_len,
            self.range_checker.range_max(),
            idx_limb_bits.clone(),
            idx_decomp,
            self.range_checker.clone(),
        );

        let page_result = self.input_chip.gen_output(page.clone(), x.clone());

        println!("page_result: {:?}", page_result);

        self.output_chip_trace = Some(self.output_chip.gen_page_trace::<SC>(page_result.clone()));
        self.output_chip_aux_trace =
            Some(self.output_chip.gen_aux_trace::<SC>(page_result.clone()));

        let prover_data =
            vec![trace_committer.commit(vec![self.input_chip_trace.clone().unwrap()])];

        self.input_commitment = Some(prover_data[0].commit.clone());

        tracing::debug!(
            "heights of all traces: {} {} {} {}",
            self.input_chip_trace.as_ref().unwrap().height(),
            self.input_chip_aux_trace.as_ref().unwrap().height(),
            self.output_chip_trace.as_ref().unwrap().height(),
            self.output_chip_aux_trace.as_ref().unwrap().height()
        );

        (vec![self.input_chip_trace.clone().unwrap()], prover_data)
    }
}
