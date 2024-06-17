use crate::common::page::Page;
use crate::group_by::final_page::MyFinalPageAir;
use crate::group_by::group_by_input::GroupByAir;
use crate::range_gate::RangeCheckerGateChip;
use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::ProverTraceData;
use afs_stark_backend::prover::trace::TraceCommitter;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use std::marker::PhantomData;
use std::sync::Arc;

pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub group_by: GroupByAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub final_chip: MyFinalPageAir,
    _marker: PhantomData<SC>,
}

pub struct GroupByTraces<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub group_by_trace: DenseMatrix<Val<SC>>,
    pub group_by_aux_trace: DenseMatrix<Val<SC>>,
    pub final_page_trace: DenseMatrix<Val<SC>>,
    pub final_page_aux_trace: DenseMatrix<Val<SC>>,
}

pub struct GroupByCommitments<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub group_by_commitment: Com<SC>,
    pub final_page_commitment: Com<SC>,
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
            group_by_cols.len(),
            1,
            limb_bits,
            decomp,
        );
        Self {
            group_by,
            range_checker,
            final_chip,
            _marker: PhantomData,
        }
    }

    pub fn load_page(
        &mut self,
        page: &Page,
        trace_committer: &TraceCommitter<SC>,
    ) -> (
        GroupByTraces<SC>,
        GroupByCommitments<SC>,
        Vec<ProverTraceData<SC>>,
    )
    where
        Val<SC>: PrimeField,
    {
        let group_by_trace = page.gen_trace();

        let grouped_page = self.group_by.request(page);
        let group_by_aux_trace: DenseMatrix<Val<SC>> = self.group_by.gen_aux_trace(&grouped_page);

        let final_page_trace = grouped_page.gen_trace();
        let final_page_aux_trace = self
            .final_chip
            .gen_aux_trace::<SC>(&grouped_page, self.range_checker.clone());

        let prover_data = vec![
            trace_committer.commit(vec![group_by_trace.clone()]),
            trace_committer.commit(vec![final_page_trace.clone()]),
        ];

        let group_by_commitment = prover_data[0].commit.clone();
        let final_page_commitment = prover_data[1].commit.clone();

        (
            GroupByTraces {
                group_by_trace,
                group_by_aux_trace,
                final_page_trace,
                final_page_aux_trace,
            },
            GroupByCommitments {
                group_by_commitment,
                final_page_commitment,
            },
            prover_data,
        )
    }
}
