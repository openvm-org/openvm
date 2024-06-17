use std::sync::Arc;

use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_field::{AbstractField, Field, PrimeField};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{
    my_final_page::MyFinalPageAir, my_initial_page::MyInitialPageAir,
    offline_checker::OfflineChecker,
};
use crate::common::page::Page;
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

/// This is a controller read/write for one page. Here's an outline of how it works
/// It owns three chips: a init_chip (MyInitialPageAir), offline_checker (OfflineChecker), and final_chip (MyFinalPageAir)
/// The trace partition of init_chip is the initial page and a trace partition of final_chip is the final page. The goal of
/// those chips and the offline_checker is to prove that the difference between the initial and final pages is exactly
/// the list of operations send on the ops_bus to the offline_checker.
///
/// High level overview:
/// We do this by imposing certain constraints on the traces and interactions between the chips. First, we use page_bus to
/// send (idx, data) from the init_chip. If the index appears in an operation, this data is intercepted by the offline_checker,
/// and the new (idx, data) after applying all the operations is sent by the offline_checker. Finally, the final_chip receives
/// all the (idx, data) whether they were intercepted by the offline_checker or not. In the trace of the offline checker, we
/// verify that the operations are applied correctly for every index.
/// We assume that the initial page in a proper format (allocated rows come first, indices in allocated rows are sorted and distinct,
/// and unallocated rows are all zeros), but we enforce that this is the case on the final page.
///
/// Exact protocol:
/// Initial page chip.
///     - This will start with the commitment of the initial page, send (idx, data) to the page bus for every allocated row with multiplicity 1.
///     - We assume that the initial page is in the proper format
/// Offline Checker.
///     - This is the chip that has the list of all operations and imposes sorting ordering.
///     - Every row in the trace has three bits: is_initial, is_final, and is_internal. Exactly one of those bits should be on in 1 row.
///         - is_initial is on to indicate that (idx, data) in the row is part of the initial page
///         - is_internal is on when the row refers to an internal R/W operation
///         - is_final is on to indicate that (idx, data) in the row is part of the final page.
///     - Receives (idx, data) for every row tagged is_initial on page_bus with multiplicity 1
///     - Sends (idx, data) for every row tagged is_final on page_bus with multiplicity 3
///     - Receives (clk, idx, data,  R or W) for every row tagged is_internal on ops_bus with multiplicity 1
///     - Every key block must end in an is_final operation so that (idx, data) is sent to the final chip.
///     - Moreover, every key block must either start with an is_initial row (so the row comes from the initial page chip)
///       or with a write operation (in case the operations allocate a row in the page with a new index)
///     - Furthermore, the row above is_final must be is_internal
///     - As mentioned above, we must ensure that the rows are sorted by (key, clk). We use the range_bus to enforce the sorting
///     - For every read operation, the data must match the data in the previous operation on the same idx
/// Final page chip.
///     - This should be essentially the same as initial page chip that starts with the commitment of the final page
///     - Receives (idx, data) on the page bus for every row with multiplicity
///         - 0 for unallocated rows
///         - 1 for allocated rows that don’t appear in the operations
///         - 3 for allocated rows that appear in the operations
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
/// If the interactions work out, we get that a-b+c-d=0 must be true. Moreover, we know that if idx has an is_initial row in the Offline Checker
/// (so it’s received by the Offline Checker), its block must end in an is_final row which implies that idx will be sent by the Offline Checker.
/// In other words, the constraints imply that b>0 => c>0.
///
/// Moreover, by the AIR constraints, we know that a ∈ {0, 1}, b ∈ {0, 1}, c ∈ {0, 3}, d ∈ {0, 1, 3}. The only tuples that satisfy a-b+c-d=0
/// and b>0 => c>0 are the following:
///     - (a, b, c, d) = (0, 0, 0, 0), which corresponds to the case of idx not appearing anywhere
///     - (a, b, c, d) = (0, 0, 3, 3), which corresponds to the case where idx is not present in the initial page but
///       is inserted in the operations in the final page
///     - (a, b, c, d) = (1, 0, 0, 1), which corresponds to the case where idx is present in the initial page but not in the
///       list of operations, so it’s just received by the final page chip
///     - (a, b, c, d) = (1, 1, 3, 3), which corresponds to the case where idx is present in the initial page and appears in
///       an operation
/// Note that in all of those cases b>0 => a=b and c>0 => d=c as wanted. The above tuples cover exactly all cases we support.
///
/// This proves that the list of operations gets us from the initial page to the final page exactly, which is all that we want.
pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub init_chip: MyInitialPageAir,
    pub offline_checker: OfflineChecker,
    pub final_chip: MyFinalPageAir,

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
            init_chip: MyInitialPageAir::new(page_bus_index, idx_len, data_len),
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
            final_chip: MyFinalPageAir::new(
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

    pub fn update_range_checker(&mut self, idx_decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.air.bus_index,
            1 << idx_decomp,
        ));
    }

    pub fn load_page_and_ops(
        &mut self,
        page: &Page,
        ops: Vec<Operation>,
        trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (Vec<DenseMatrix<Val<SC>>>, Vec<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        let mut page = page.clone();

        assert!(!page.rows.is_empty());
        self.init_chip_trace = Some(self.gen_page_trace(&page));

        self.init_chip_trace = Some(self.gen_page_trace(&page));

        self.offline_checker_trace =
            Some(self.gen_ops_trace(&mut page, &ops, self.range_checker.clone(), trace_degree));

        // Sorting the page by (1-is_alloc, idx)
        page.rows
            .sort_by_key(|row| (1 - row.is_alloc, row.idx.clone()));

        // HashSet of all indices used in operations
        let internal_indices = ops.iter().map(|op| op.idx.clone()).collect();

        self.final_chip_trace = Some(self.gen_page_trace(&page));
        self.final_page_aux_trace = Some(self.final_chip.gen_aux_trace::<SC>(
            &page,
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
