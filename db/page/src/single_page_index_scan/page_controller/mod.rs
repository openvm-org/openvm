use std::sync::Arc;

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    prover::{
        trace::{ProverTraceData, TraceCommitmentBuilder, TraceCommitter},
        types::Proof,
    },
    verifier::VerificationError,
};
use afs_test_utils::engine::StarkEngine;
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::info_span;

use crate::common::page::Page;

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

    page_traces: Vec<DenseMatrix<Val<SC>>>,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> PageController<SC>
where
    Val<SC>: AbstractField + PrimeField64,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_bus_index: usize,
        range_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        range_max: u32,
        idx_limb_bits: usize,
        idx_decomp: usize,
        cmp: Comp,
    ) -> Self {
        let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus_index, range_max));
        Self {
            input_chip: PageIndexScanInputChip::new(
                page_bus_index,
                idx_len,
                data_len,
                idx_limb_bits,
                idx_decomp,
                range_checker.clone(),
                cmp,
            ),
            output_chip: PageIndexScanOutputChip::new(
                page_bus_index,
                idx_len,
                data_len,
                idx_limb_bits,
                idx_decomp,
                range_checker.clone(),
            ),
            input_chip_trace: None,
            input_chip_aux_trace: None,
            output_chip_trace: None,
            output_chip_aux_trace: None,
            input_commitment: None,
            output_commitment: None,
            page_traces: vec![],
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

    pub fn gen_output(&self, page: Page, x: Vec<u32>, page_width: usize, cmp: Comp) -> Page {
        let mut output: Vec<Vec<u32>> = vec![];

        for page_row in page.iter() {
            let is_alloc = page_row.is_alloc;
            let idx = page_row.idx.clone();
            let data = page_row.data.clone();

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

        let num_remaining = page.height() - output.len();

        output.extend((0..num_remaining).map(|_| vec![0; page_width]));

        Page::from_2d_vec(&output, page.idx_len(), page.data_len())
    }

    pub fn set_up_keygen_builder<'a>(
        &'a self,
        keygen_builder: &mut MultiStarkKeygenBuilder<'a, SC>,
        page_width: usize,
        idx_len: usize,
    ) {
        let input_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
        let output_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
        let input_page_aux_ptr = keygen_builder.add_main_matrix(self.input_chip.air.aux_width());
        let output_page_aux_ptr = keygen_builder.add_main_matrix(self.output_chip.air.aux_width());
        let range_checker_ptr = keygen_builder.add_main_matrix(self.range_checker.air_width());

        keygen_builder.add_partitioned_air(
            &self.input_chip.air,
            idx_len,
            vec![input_page_ptr, input_page_aux_ptr],
        );

        keygen_builder.add_partitioned_air(
            &self.output_chip.air,
            0,
            vec![output_page_ptr, output_page_aux_ptr],
        );

        keygen_builder.add_partitioned_air(&self.range_checker.air, 0, vec![range_checker_ptr]);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prove(
        &mut self,
        engine: &dyn StarkEngine<SC>,
        pk: &MultiStarkProvingKey<SC>,
        trace_builder: &mut TraceCommitmentBuilder<SC>,
        input_prover_data: Arc<ProverTraceData<SC>>,
        output_prover_data: Arc<ProverTraceData<SC>>,
        x: Vec<u32>,
        idx_decomp: usize,
    ) -> Proof<SC>
    where
        Val<SC>: PrimeField,
        Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        let page_traces = self.page_traces.clone();

        let input_chip_aux_trace = self.input_chip_aux_trace();
        let output_chip_aux_trace = self.output_chip_aux_trace();
        let range_checker_trace = self.range_checker_trace();

        // Clearing the range_checker counts
        self.update_range_checker(idx_decomp);

        trace_builder.clear();

        trace_builder.load_cached_trace(
            page_traces[0].clone(),
            match Arc::try_unwrap(input_prover_data) {
                Ok(data) => data,
                Err(_) => panic!("Prover data should have only one owner"),
            },
        );
        trace_builder.load_cached_trace(
            page_traces[1].clone(),
            match Arc::try_unwrap(output_prover_data) {
                Ok(data) => data,
                Err(_) => panic!("Prover data should have only one owner"),
            },
        );
        trace_builder.load_trace(input_chip_aux_trace);
        trace_builder.load_trace(output_chip_aux_trace);
        trace_builder.load_trace(range_checker_trace);

        tracing::info_span!("Prove trace commitment").in_scope(|| trace_builder.commit_current());

        let vk = pk.vk();

        let main_trace_data = trace_builder.view(
            &vk,
            vec![
                &self.input_chip.air,
                &self.output_chip.air,
                &self.range_checker.air,
            ],
        );

        let pis = vec![
            x.iter()
                .map(|x| Val::<SC>::from_canonical_u32(*x))
                .collect(),
            vec![],
            vec![],
        ];

        let prover = engine.prover();
        let mut challenger = engine.new_challenger();

        prover.prove(&mut challenger, pk, main_trace_data, &pis)
    }

    /// This function takes a proof (returned by the prove function) and verifies it
    pub fn verify(
        &self,
        engine: &dyn StarkEngine<SC>,
        vk: MultiStarkVerifyingKey<SC>,
        proof: Proof<SC>,
        x: Vec<u32>,
    ) -> Result<(), VerificationError>
    where
        Val<SC>: PrimeField,
    {
        let verifier = engine.verifier();

        let pis = vec![
            x.iter()
                .map(|x| Val::<SC>::from_canonical_u32(*x))
                .collect(),
            vec![],
            vec![],
        ];

        let mut challenger = engine.new_challenger();
        verifier.verify(
            &mut challenger,
            &vk,
            vec![
                &self.input_chip.air,
                &self.output_chip.air,
                &self.range_checker.air,
            ],
            &proof,
            &pis,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_page(
        &mut self,
        page_input: Page,
        page_output: Page,
        page_input_pdata: Option<Arc<ProverTraceData<SC>>>,
        page_output_pdata: Option<Arc<ProverTraceData<SC>>>,
        x: Vec<u32>,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (Arc<ProverTraceData<SC>>, Arc<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        // idx_decomp can't change between different pages since range_checker depends on it
        assert!(1 << idx_decomp == self.range_checker.range_max());

        assert!(!page_input.is_empty());

        let trace_span = info_span!("Load page trace generation").entered();
        let bus_index = self.input_chip.air.page_bus_index;

        self.input_chip = PageIndexScanInputChip::new(
            bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
            self.range_checker.clone(),
            self.input_chip.cmp.clone(),
        );
        self.input_chip_trace = Some(self.input_chip.gen_page_trace::<SC>(&page_input));
        self.input_chip_aux_trace =
            Some(self.input_chip.gen_aux_trace::<SC>(&page_input, x.clone()));

        self.output_chip = PageIndexScanOutputChip::new(
            bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
            self.range_checker.clone(),
        );

        self.output_chip_trace = Some(self.output_chip.gen_page_trace::<SC>(&page_output));
        self.output_chip_aux_trace = Some(self.output_chip.gen_aux_trace::<SC>(&page_output));
        trace_span.exit();

        let trace_commit_span = info_span!("Load page trace commitment").entered();
        let page_input_prover_data = match page_input_pdata {
            Some(pdata) => pdata,
            None => Arc::new(trace_committer.commit(vec![self.input_chip_trace.clone().unwrap()])),
        };
        let page_output_prover_data = match page_output_pdata {
            Some(pdata) => pdata,
            None => Arc::new(trace_committer.commit(vec![self.output_chip_trace.clone().unwrap()])),
        };
        trace_commit_span.exit();

        self.input_commitment = Some(page_input_prover_data.commit.clone());
        self.output_commitment = Some(page_output_prover_data.commit.clone());

        self.page_traces = vec![
            self.input_chip_trace.clone().unwrap(),
            self.output_chip_trace.clone().unwrap(),
        ];

        (page_input_prover_data, page_output_prover_data)
    }
}
