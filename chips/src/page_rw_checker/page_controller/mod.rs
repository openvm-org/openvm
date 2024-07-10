use std::collections::HashSet;
use std::sync::Arc;

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
    rap::AnyRap,
    verifier::VerificationError,
};
use afs_test_utils::engine::StarkEngine;
use p3_field::{AbstractField, Field, PrimeField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::info_span;

use super::{
    final_page::IndexedPageWriteAir, initial_page::PageReadAir, offline_checker::OfflineChecker,
};
use crate::common::page::Page;
use crate::range_gate::RangeCheckerGateChip;

#[derive(PartialEq, Clone, Debug, Copy)]
pub enum OpType {
    Read = 0,
    Write = 1,
    Delete = 2,
}

#[derive(Clone, Debug, derive_new::new)]
pub struct Operation {
    pub clk: usize,
    pub idx: Vec<u32>,
    pub data: Vec<u32>,
    pub op_type: OpType,
}

struct PageRWTraces<F> {
    init_page_trace: RowMajorMatrix<F>,
    final_page_trace: RowMajorMatrix<F>,
    final_page_aux_trace: RowMajorMatrix<F>,
    offline_checker_trace: RowMajorMatrix<F>,
}

#[allow(dead_code)]
struct PageCommitments<SC: StarkGenericConfig> {
    init_page_commitment: Com<SC>,
    final_page_commitment: Com<SC>,
}

/// This is a controller for read/write/delete for one page. Here's an outline of how it works
/// It owns three chips: a init_chip (MyInitialPageAir), offline_checker (OfflineChecker), and final_chip (MyFinalPageAir)
/// The only trace partition of init_chip is the initial page and a trace partition of final_chip is the final page. The goal of
/// those chips and the offline_checker is to prove that the difference between the initial and final pages is exactly
/// the list of operations sent on the ops_bus to the offline_checker.
///
/// High-level overview:
/// We do this by imposing certain constraints on the traces and interactions between the chips. First, we use page_bus to
/// send (idx, data) from the init_chip. If the index appears in an operation, this data is intercepted by the offline_checker,
/// and the new (idx, data) after applying all the operations is sent by the offline_checker (or nothing is sent if the index got deleted).
/// Finally, the final_chip receives all the (idx, data) whether they were intercepted by the offline_checker or not. In the trace of
/// the offline checker, we verify that the operations are applied correctly for every index.
/// We assume that the initial page in a proper format (allocated rows come first, indices in allocated rows are sorted and distinct,
/// and unallocated rows are all zeros), but we enforce that this is the case on the final page.
///
/// Exact protocol:
/// Initial page chip.
///     - This will start with the commitment of the initial page, send (idx, data) to the page bus for every allocated row with multiplicity 1.
///     - We assume that the initial page is in the proper format
/// Offline Checker.
///     - This is the chip that has the list of all operations and imposes sorting ordering.
///     - Every row in the trace has fout bits: is_initial, is_final_write, is_final_delete, and is_internal.
///         - Exactly one of those bits should be on in 1 row.
///         - is_initial is on to indicate that (idx, data) in the row is part of the initial page
///         - is_internal is on when the row refers to an internal R/W/D operation
///         - is_final_write and is_final_delete indicate that the row is last row for the idx block
///             - is_final_write indicates that (idx, data) in that row is part of the final page
///             - is_final_delete indicates that the idx was deleted after all operations
///     - Receives (idx, data) for every row tagged is_initial on page_bus with multiplicity 1
///     - Sends (idx, data) for every row tagged is_final_write on page_bus with multiplicity 3
///     - Receives (clk, idx, data,  R or W) for every row tagged is_internal on ops_bus with multiplicity 1
///     - Every key block must end in an is_final_write or an is_final_delete operation
///         - The offline checker sends (idx, data) to the final chip only in case of is_final_write.
///     - Moreover, every key block must either start with an is_initial row (so the row comes from the initial page chip)
///       or with a write operation (in case the operations allocate a row in the page with a new index)
///     - Furthermore, the row above is_final_write or is_final_delete must be is_internal
///         - The row above is_final_write must be a delete operation
///     - The row below is_final_delete must have a different idx
///     - As mentioned above, we must ensure that the rows are sorted by (key, clk). We use the range_bus to enforce the sorting
///     - For every read operation, the data must match the data in the previous operation on the same idx
/// Final page chip.
///     - This should be essentially the same as initial page chip that starts with the commitment of the final page
///     - Receives (idx, data) on the page bus for every row with multiplicity
///         - 0 for unallocated rows/deleted indices
///         - 1 for allocated indicies that don’t appear in the operations
///         - 3 for allocated indicies that appear in the operations and were not deleted by the end
///     - Those three options for the multiplicity should be enforced in the AIR constraints
///     - This chip should enforce that the page is sorted properly (allocated rows come first and allocated rows are sorted by key).
///       We use the range_bus to enforce this.
///
/// Pseudo-proof that the diff between initial and final page is exactly the internal operations:
///
/// Assuming that the Offline Checker is constrained correctly, we only need to prove that the interactions between
/// the different chips work as intended. We will show this now.
///
/// Let idx be any index. On the page_bus, this index will be sent a times from the initial page chip, received b times by the
/// Offline Checker, sent c times by the Offline Checker, and received d times by the final page chip.
///
/// If the interactions work out, we get that a-b+c-d=0 must be true.
///
/// Moreover, by the AIR constraints, we know that a ∈ {0, 1}, b ∈ {0, 1}, c ∈ {0, 3}, d ∈ {0, 1, 3}. The only tuples that satisfy a-b+c-d=0
/// and b>0 => c>0 are the following:
///     - (a, b, c, d) = (0, 0, 0, 0), which corresponds to the case where idx does not appear anywhere or has been inserted and then deleted
///     - (a, b, c, d) = (1, 1, 0, 0), which corresponds to the case where idx is present in the initial page and was deleted in the operations
///     - (a, b, c, d) = (0, 0, 3, 3), which corresponds to the case where idx is not present in the initial page but
///       is inserted in the operations in the final page
///     - (a, b, c, d) = (1, 0, 0, 1), which corresponds to the case where idx is present in the initial page but not in the
///       list of operations, so it’s just received by the final page chip
///     - (a, b, c, d) = (1, 1, 3, 3), which corresponds to the case where idx is present in the initial page and appears in
///       an operation
/// Note that in all of those cases b>0 => a=b and c>0 => d=c as wanted. The above tuples cover exactly all cases we support.
///
/// This proves that the list of operations gets us from the initial page to the final page exactly, which is all we want.
pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    init_chip: PageReadAir,
    offline_checker: OfflineChecker,
    final_chip: IndexedPageWriteAir,

    traces: Option<PageRWTraces<Val<SC>>>,
    page_commitments: Option<PageCommitments<SC>>,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> PageController<SC> {
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
            init_chip: PageReadAir::new(page_bus_index, idx_len, data_len),
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
            final_chip: IndexedPageWriteAir::new(
                page_bus_index,
                range_bus_index,
                idx_len,
                data_len,
                idx_limb_bits,
                idx_decomp,
            ),

            traces: None,
            page_commitments: None,

            range_checker: Arc::new(RangeCheckerGateChip::new(range_bus_index, 1 << idx_decomp)),
        }
    }

    pub fn range_checker_trace(&self) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.range_checker.generate_trace()
    }

    pub fn reset_range_checker(&mut self, idx_decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.air.bus_index,
            1 << idx_decomp,
        ));
    }

    pub fn load_page_and_ops(
        &mut self,
        page: &Page,
        init_page_pdata: Option<Arc<ProverTraceData<SC>>>,
        final_page_pdata: Option<Arc<ProverTraceData<SC>>>,
        ops: Vec<Operation>,
        trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (Arc<ProverTraceData<SC>>, Arc<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        let trace_span = info_span!("Load page trace generation").entered();
        let mut page = page.clone();

        assert!(!page.rows.is_empty());
        let init_page_trace = self.gen_page_trace(&page);

        let offline_checker_trace =
            self.gen_ops_trace(&mut page, &ops, self.range_checker.clone(), trace_degree);

        // HashSet of all indices intercepted by Offline Checker to be written to the final page
        let mut final_write_indices = HashSet::new();
        // HashSet of all indices intercepted by Offline Checker to be deleted from the final page
        let mut final_delete_indices = HashSet::new();
        for op in ops.iter().rev() {
            if final_write_indices.contains(&op.idx) || final_delete_indices.contains(&op.idx) {
                continue;
            }

            if op.op_type == OpType::Delete {
                final_delete_indices.insert(op.idx.clone());
            } else {
                // Can be read or write operation as either will be part of the Offline Checker trace
                final_write_indices.insert(op.idx.clone());
            }
        }

        let final_page_trace = self.gen_page_trace(&page);
        let final_page_aux_trace = self.final_chip.gen_aux_trace::<SC>(
            &page,
            self.range_checker.clone(),
            final_write_indices,
        );
        trace_span.exit();

        let trace_commit_span = info_span!("Load page trace commitment").entered();
        let init_page_pdata = match init_page_pdata {
            Some(prover_data) => prover_data,
            None => Arc::new(trace_committer.commit(vec![init_page_trace.clone()])),
        };

        let final_page_pdata = match final_page_pdata {
            Some(prover_data) => prover_data,
            None => Arc::new(trace_committer.commit(vec![final_page_trace.clone()])),
        };
        trace_commit_span.exit();

        self.traces = Some(PageRWTraces {
            init_page_trace,
            final_page_trace,
            final_page_aux_trace,
            offline_checker_trace,
        });

        self.page_commitments = Some(PageCommitments {
            init_page_commitment: init_page_pdata.commit.clone(),
            final_page_commitment: final_page_pdata.commit.clone(),
        });

        (init_page_pdata, final_page_pdata)
    }

    /// Sets up keygen with the different trace partitions for the chips
    /// init_chip, final_chip, offline_checker, range_checker, and the
    /// ops_sender, which is passed in
    pub fn set_up_keygen_builder(
        &self,
        keygen_builder: &mut MultiStarkKeygenBuilder<SC>,
        page_height: usize,
        offline_checker_trace_degree: usize,
        ops_sender: &dyn AnyRap<SC>,
        ops_sender_trace_degree: usize,
    ) where
        Val<SC>: PrimeField,
    {
        let init_page_ptr = keygen_builder.add_cached_main_matrix(self.init_chip.air_width());
        let final_page_ptr = keygen_builder.add_cached_main_matrix(self.final_chip.page_width());
        let final_page_aux_ptr = keygen_builder.add_main_matrix(self.final_chip.aux_width());

        keygen_builder.add_partitioned_air(&self.init_chip, page_height, 0, vec![init_page_ptr]);

        keygen_builder.add_partitioned_air(
            &self.final_chip,
            page_height,
            0,
            vec![final_page_ptr, final_page_aux_ptr],
        );

        keygen_builder.add_air(&self.offline_checker, offline_checker_trace_degree, 0);

        keygen_builder.add_air(
            &self.range_checker.air,
            self.range_checker.range_max() as usize,
            0,
        );

        keygen_builder.add_air(ops_sender, ops_sender_trace_degree, 0);
    }

    /// This function clears the trace_builder, loads in the traces for all involved chips
    /// (including the range_checker and the ops_sender, which is passed in along with its trace),
    /// commits them, and then generates the proof.
    /// cached_traces_prover_data is a vector of ProverTraceData object for the cached pages
    /// (init_page, final_page), which is returned by load_page_and_ops
    #[allow(clippy::too_many_arguments)]
    pub fn prove(
        &self,
        engine: &impl StarkEngine<SC>,
        partial_pk: &MultiStarkPartialProvingKey<SC>,
        trace_builder: &mut TraceCommitmentBuilder<SC>,
        init_page_pdata: Arc<ProverTraceData<SC>>,
        final_page_pdata: Arc<ProverTraceData<SC>>,
        ops_sender: &dyn AnyRap<SC>,
        ops_sender_trace: DenseMatrix<Val<SC>>,
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
        let traces = self.traces.as_ref().unwrap();

        trace_builder.clear();

        trace_builder.load_cached_trace(
            traces.init_page_trace.clone(),
            match Arc::try_unwrap(init_page_pdata) {
                Ok(data) => data,
                Err(_) => panic!("Prover data should have only one owner"),
            },
        );
        trace_builder.load_cached_trace(
            traces.final_page_trace.clone(),
            match Arc::try_unwrap(final_page_pdata) {
                Ok(data) => data,
                Err(_) => panic!("Prover data should have only one owner"),
            },
        );
        trace_builder.load_trace(traces.final_page_aux_trace.clone());
        trace_builder.load_trace(traces.offline_checker_trace.clone());
        trace_builder.load_trace(self.range_checker.generate_trace());
        trace_builder.load_trace(ops_sender_trace);

        tracing::info_span!("Prove trace commitment").in_scope(|| trace_builder.commit_current());

        let partial_vk = partial_pk.partial_vk();

        let main_trace_data = trace_builder.view(
            &partial_vk,
            vec![
                &self.init_chip,
                &self.final_chip,
                &self.offline_checker,
                &self.range_checker.air,
                ops_sender,
            ],
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
        ops_sender: &dyn AnyRap<SC>,
    ) -> Result<(), VerificationError>
    where
        Val<SC>: PrimeField,
    {
        let verifier = engine.verifier();

        let pis = vec![vec![]; partial_vk.per_air.len()];

        let mut challenger = engine.new_challenger();
        verifier.verify(
            &mut challenger,
            partial_vk,
            vec![
                &self.init_chip,
                &self.final_chip,
                &self.offline_checker,
                &self.range_checker.air,
                ops_sender,
            ],
            proof,
            &pis,
        )
    }

    fn gen_page_trace(&self, page: &Page) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        page.gen_trace()
    }

    fn gen_ops_trace(
        &self,
        page: &mut Page,
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
}
