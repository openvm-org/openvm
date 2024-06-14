use crate::group_by::final_page::MyFinalPageAir;
use crate::group_by::group_by_input::GroupByAir;
use crate::range_gate::RangeCheckerGateChip;
use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::ProverTraceData;
use afs_stark_backend::prover::trace::TraceCommitter;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use std::sync::Arc;

pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub group_by: GroupByAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub final_chip: MyFinalPageAir,

    group_by_trace: Option<DenseMatrix<Val<SC>>>,
    final_page_trace: Option<DenseMatrix<Val<SC>>>,
    final_page_aux_trace: Option<DenseMatrix<Val<SC>>>,

    group_by_commitment: Option<Com<SC>>,
    final_page_commitment: Option<Com<SC>>,
}

impl<SC: StarkGenericConfig> PageController<SC> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_width: usize,
        group_by_cols: Vec<usize>,
        aggregated_col: usize,
        internal_bus: usize,
        output_bus: usize,
        range_bus: usize,
        limb_bits: usize,
        decomp: usize,
    ) -> Self {
        let group_by = GroupByAir::new(
            page_width,
            group_by_cols.clone(),
            aggregated_col,
            internal_bus,
            output_bus,
        );
        let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus, 1 << decomp));
        let final_chip = MyFinalPageAir::new(
            output_bus,
            range_bus,
            group_by_cols.len() + 1,
            1,
            limb_bits,
            decomp,
        );
        Self {
            group_by,
            range_checker,
            final_chip,
            group_by_trace: None,
            final_page_trace: None,
            final_page_aux_trace: None,
            group_by_commitment: None,
            final_page_commitment: None,
        }
    }

    pub fn load_page(
        &mut self,
        page: Vec<Vec<u32>>,
        trace_committer: &TraceCommitter<SC>,
    ) -> (Vec<DenseMatrix<Val<SC>>>, Vec<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        self.group_by_trace = Some(self.group_by.gen_page_trace(page.clone()));
        self.final_page_trace = Some(self.final_chip.gen_page_trace::<SC>(page.clone()));
        self.final_page_aux_trace = Some(
            self.final_chip
                .gen_aux_trace::<SC>(page, self.range_checker.clone()),
        );

        let prover_data = vec![
            trace_committer.commit(vec![self.group_by_trace.clone().unwrap()]),
            trace_committer.commit(vec![self.final_page_trace.clone().unwrap()]),
        ];

        self.group_by_commitment = Some(prover_data[0].commit.clone());
        self.final_page_commitment = Some(prover_data[1].commit.clone());

        (
            vec![
                self.group_by_trace.clone().unwrap(),
                self.final_page_trace.clone().unwrap(),
            ],
            prover_data,
        )
    }
}
