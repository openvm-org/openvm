use std::sync::Arc;

use afs_stark_backend::{
    config::Com,
    prover::trace::{ProverTraceData, TraceCommitter},
};
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::range_gate::RangeCheckerGateChip;

use super::{
    page_index_scan_input::{Comp, PageIndexScanInputChip},
    page_index_scan_output::PageIndexScanOutputChip,
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
    output_commitment: Option<Com<SC>>,

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
        cmp: Comp,
    ) -> Self {
        let range_checker = Arc::new(RangeCheckerGateChip::new(bus_index, range_max));
        Self {
            input_chip: PageIndexScanInputChip::new(
                bus_index,
                idx_len,
                data_len,
                idx_limb_bits.clone(),
                idx_decomp,
                range_checker.clone(),
                cmp,
            ),
            output_chip: PageIndexScanOutputChip::new(
                bus_index,
                idx_len,
                data_len,
                idx_limb_bits.clone(),
                idx_decomp,
                range_checker.clone(),
            ),
            input_chip_trace: None,
            input_chip_aux_trace: None,
            output_chip_trace: None,
            output_chip_aux_trace: None,
            input_commitment: None,
            output_commitment: None,
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

    pub fn gen_output(
        &self,
        page: Vec<Vec<u32>>,
        x: Vec<u32>,
        idx_len: usize,
        page_width: usize,
        cmp: Comp,
    ) -> Vec<Vec<u32>> {
        let mut output: Vec<Vec<u32>> = vec![];

        for page_row in &page {
            let is_alloc = page_row[0];
            let idx = page_row[1..1 + idx_len].to_vec();
            let data = page_row[1 + idx_len..].to_vec();

            match cmp {
                Comp::Lt => {
                    let mut less_than = false;
                    for (&idx_val, &x_val) in idx.iter().zip(x.iter()) {
                        use std::cmp::Ordering;
                        match idx_val.cmp(&x_val) {
                            Ordering::Less => {
                                less_than = true;
                                break;
                            }
                            Ordering::Greater => {
                                break;
                            }
                            Ordering::Equal => {}
                        }
                    }
                    if less_than {
                        output.push(
                            vec![is_alloc]
                                .into_iter()
                                .chain(idx.iter().cloned())
                                .chain(data.iter().cloned())
                                .collect(),
                        );
                    }
                }
                Comp::Lte => {
                    let mut less_than = false;
                    for (&idx_val, &x_val) in idx.iter().zip(x.iter()) {
                        use std::cmp::Ordering;
                        match idx_val.cmp(&x_val) {
                            Ordering::Less => {
                                less_than = true;
                                break;
                            }
                            Ordering::Greater => {
                                break;
                            }
                            Ordering::Equal => {}
                        }
                    }

                    let mut eq = true;
                    for (&idx_val, &x_val) in idx.iter().zip(x.iter()) {
                        if idx_val != x_val {
                            eq = false;
                            break;
                        }
                    }

                    if less_than || eq {
                        output.push(
                            vec![is_alloc]
                                .into_iter()
                                .chain(idx.iter().cloned())
                                .chain(data.iter().cloned())
                                .collect(),
                        );
                    }
                }
                Comp::Eq => {
                    let mut eq = true;
                    for (&idx_val, &x_val) in idx.iter().zip(x.iter()) {
                        if idx_val != x_val {
                            eq = false;
                            break;
                        }
                    }
                    if eq {
                        output.push(
                            vec![is_alloc]
                                .into_iter()
                                .chain(idx.iter().cloned())
                                .chain(data.iter().cloned())
                                .collect(),
                        );
                    }
                }
                Comp::Gte => {
                    let mut greater_than = false;
                    for (&idx_val, &x_val) in idx.iter().zip(x.iter()) {
                        use std::cmp::Ordering;
                        match idx_val.cmp(&x_val) {
                            Ordering::Greater => {
                                greater_than = true;
                                break;
                            }
                            Ordering::Less => {
                                break;
                            }
                            Ordering::Equal => {}
                        }
                    }

                    let mut eq = true;
                    for (&idx_val, &x_val) in idx.iter().zip(x.iter()) {
                        if idx_val != x_val {
                            eq = false;
                            break;
                        }
                    }

                    if greater_than || eq {
                        output.push(
                            vec![is_alloc]
                                .into_iter()
                                .chain(idx.iter().cloned())
                                .chain(data.iter().cloned())
                                .collect(),
                        );
                    }
                }
                Comp::Gt => {
                    let mut greater_than = false;
                    for (&idx_val, &x_val) in idx.iter().zip(x.iter()) {
                        use std::cmp::Ordering;
                        match idx_val.cmp(&x_val) {
                            Ordering::Greater => {
                                greater_than = true;
                                break;
                            }
                            Ordering::Less => {
                                break;
                            }
                            Ordering::Equal => {}
                        }
                    }
                    if greater_than {
                        output.push(
                            vec![is_alloc]
                                .into_iter()
                                .chain(idx.iter().cloned())
                                .chain(data.iter().cloned())
                                .collect(),
                        );
                    }
                }
            }
        }

        let num_remaining = page.len() - output.len();

        output.extend((0..num_remaining).map(|_| vec![0; page_width]));

        output
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_page(
        &mut self,
        page_input: Vec<Vec<u32>>,
        page_output: Vec<Vec<u32>>,
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

        assert!(!page_input.is_empty());

        let bus_index = self.input_chip.air.bus_index;

        self.input_chip = PageIndexScanInputChip::new(
            bus_index,
            idx_len,
            data_len,
            idx_limb_bits.clone(),
            idx_decomp,
            self.range_checker.clone(),
            self.input_chip.cmp.clone(),
        );
        self.input_chip_trace = Some(self.input_chip.gen_page_trace::<SC>(page_input.clone()));
        self.input_chip_aux_trace = Some(
            self.input_chip
                .gen_aux_trace::<SC>(page_input.clone(), x.clone()),
        );

        self.output_chip = PageIndexScanOutputChip::new(
            bus_index,
            idx_len,
            data_len,
            idx_limb_bits.clone(),
            idx_decomp,
            self.range_checker.clone(),
        );

        self.output_chip_trace = Some(self.output_chip.gen_page_trace::<SC>(page_output.clone()));
        self.output_chip_aux_trace =
            Some(self.output_chip.gen_aux_trace::<SC>(page_output.clone()));

        let prover_data = vec![
            trace_committer.commit(vec![self.input_chip_trace.clone().unwrap()]),
            trace_committer.commit(vec![self.output_chip_trace.clone().unwrap()]),
        ];

        self.input_commitment = Some(prover_data[0].commit.clone());
        self.output_commitment = Some(prover_data[1].commit.clone());

        (
            vec![
                self.input_chip_trace.clone().unwrap(),
                self.output_chip_trace.clone().unwrap(),
            ],
            prover_data,
        )
    }
}
