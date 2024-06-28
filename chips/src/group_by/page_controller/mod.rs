use crate::common::page::Page;
use crate::group_by::final_page::MyFinalPageAir;
use crate::group_by::group_by_input::GroupByAir;
use crate::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{
        types::{MultiStarkPartialProvingKey, MultiStarkPartialVerifyingKey},
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
use std::marker::PhantomData;
use std::sync::Arc;

/// Main struct for group-by integration testing with `MyFinalPage`
///
/// It has a `GroupByAir` for the group-by operation, followed by a
/// `MyFinalPageAir` for the final page operation which uses a
/// `RangeCheckerGateChip` for the range check.
///
/// The `load_page` function is the main entry point for loading a page into the
/// controller, and it purely returns all necessary traces and commitments for
/// the group-by operation.
///
/// The `refresh_range_checker` function is used to reset the `range_checker` to
/// all zeros.
pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub group_by: GroupByAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub final_chip: MyFinalPageAir,
    _marker: PhantomData<SC>,
}

/// Container struct for all traces generated by `load_page`.
/// Note importantly that the `range_checker` trace is generated automatically.
pub struct GroupByTraces<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub group_by_trace: DenseMatrix<Val<SC>>,
    pub group_by_aux_trace: DenseMatrix<Val<SC>>,
    pub final_page_trace: DenseMatrix<Val<SC>>,
    pub final_page_aux_trace: DenseMatrix<Val<SC>>,
}

/// Container struct for all commitments generated by `load_page`.
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

    /// Load a `page: &Page` into the controller and generate all necessary traces and
    /// commitments for the group-by operation.
    ///
    /// Returns a tuple of `GroupByTraces`, `GroupByCommitments`, and a vector of
    /// `ProverTraceData`.
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

    pub fn refresh_range_checker(&mut self) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.air.bus_index,
            self.range_checker.air.range_max,
        ));
    }

    /// Set up the keygen builder for the group-by test case by querying trace widths.
    pub fn set_up_keygen_builder(
        &self,
        keygen_builder: &mut MultiStarkKeygenBuilder<SC>,
        height: usize,
        range_checker_height: usize,
    ) where
        Val<SC>: PrimeField64,
    {
        let group_by_ptr = keygen_builder.add_cached_main_matrix(self.group_by.page_width);
        let final_page_ptr = keygen_builder.add_cached_main_matrix(self.final_chip.page_width());
        let group_by_aux_ptr = keygen_builder.add_main_matrix(self.group_by.aux_width());
        let final_page_aux_ptr = keygen_builder.add_main_matrix(self.final_chip.aux_width());
        let range_checker_ptr = keygen_builder.add_main_matrix(self.range_checker.air_width());

        keygen_builder.add_partitioned_air(
            &self.group_by,
            height,
            0,
            vec![group_by_ptr, group_by_aux_ptr],
        );

        keygen_builder.add_partitioned_air(
            &self.final_chip,
            height,
            0,
            vec![final_page_ptr, final_page_aux_ptr],
        );

        keygen_builder.add_partitioned_air(
            &self.range_checker.air,
            range_checker_height,
            0,
            vec![range_checker_ptr],
        );
    }

    pub fn prove(
        &self,
        engine: &impl StarkEngine<SC>,
        partial_pk: &MultiStarkPartialProvingKey<SC>,
        trace_builder: &mut TraceCommitmentBuilder<SC>,
        group_by_traces: GroupByTraces<SC>,
        mut cached_traces_prover_data: Vec<ProverTraceData<SC>>,
    ) -> Proof<SC>
    where
        Val<SC>: PrimeField64,
        Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        assert!(cached_traces_prover_data.len() == 2);

        let range_checker_trace = self.range_checker.generate_trace();

        trace_builder.clear();

        trace_builder.load_cached_trace(
            group_by_traces.group_by_trace,
            cached_traces_prover_data.remove(0),
        );
        trace_builder.load_cached_trace(
            group_by_traces.final_page_trace,
            cached_traces_prover_data.remove(0),
        );

        trace_builder.load_trace(group_by_traces.group_by_aux_trace);
        trace_builder.load_trace(group_by_traces.final_page_aux_trace);
        trace_builder.load_trace(range_checker_trace);

        trace_builder.commit_current();

        let partial_vk = partial_pk.partial_vk();

        let main_trace_data = trace_builder.view(
            &partial_vk,
            vec![&self.group_by, &self.final_chip, &self.range_checker.air],
        );

        let pis = vec![vec![]; partial_vk.per_air.len()];

        let prover = engine.prover();
        let mut challenger = engine.new_challenger();
        prover.prove(&mut challenger, partial_pk, main_trace_data, &pis)
    }

    /// This function takes a proof (returned by the prove function) and verifies it
    pub fn verify(
        &self,
        engine: &impl StarkEngine<SC>,
        partial_vk: MultiStarkPartialVerifyingKey<SC>,
        proof: Proof<SC>,
    ) -> Result<(), VerificationError>
    where
        Val<SC>: PrimeField64,
    {
        let verifier = engine.verifier();

        let pis = vec![vec![]; partial_vk.per_air.len()];

        let mut challenger = engine.new_challenger();
        verifier.verify(
            &mut challenger,
            partial_vk,
            vec![&self.group_by, &self.final_chip, &self.range_checker.air],
            proof,
            &pis,
        )
    }
}
