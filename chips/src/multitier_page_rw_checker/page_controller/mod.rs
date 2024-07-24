use std::collections::HashSet;
use std::sync::Arc;

use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use itertools::Itertools;
use p3_field::{AbstractField, Field, PrimeField64};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};

use p3_uni_stark::{StarkGenericConfig, Val};

use crate::common::page::Page;
use crate::page_rw_checker::offline_checker::PageOfflineChecker;
use crate::page_rw_checker::page_controller::Operation;
use crate::range_gate::RangeCheckerGateChip;

use self::page_tree_graph::PageTreeGraph;

use super::internal_page_air::InternalPageAir;
use super::leaf_page_air::LeafPageAir;
use super::root_signal_air::RootSignalAir;

pub mod page_tree_graph;

#[derive(Clone)]
pub struct PageTreeParams {
    pub path_bus_index: usize,
    pub leaf_cap: usize,
    pub internal_cap: usize,
    pub leaf_page_height: usize,
    pub internal_page_height: usize,
}

#[derive(Clone)]
pub struct MyLessThanTupleParams {
    pub limb_bits: usize,
    pub decomp: usize,
}

struct TreeProducts<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub root: RootProducts<SC, COMMITMENT_LEN>,
    pub leaf: NodeProducts<SC, COMMITMENT_LEN>,
    pub internal: NodeProducts<SC, COMMITMENT_LEN>,
}

struct RootProducts<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub main_traces: DenseMatrix<Val<SC>>,
    pub commitments: Com<SC>,
}

struct NodeProducts<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub data_traces: Vec<DenseMatrix<Val<SC>>>,
    pub main_traces: Vec<DenseMatrix<Val<SC>>>,
    pub prover_data: Vec<ProverTraceData<SC>>,
    pub commitments: Vec<Com<SC>>,
}

#[derive(Clone)]
pub struct PageControllerDataTrace<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub init_leaf_chip_traces: Vec<DenseMatrix<Val<SC>>>,
    pub init_internal_chip_traces: Vec<DenseMatrix<Val<SC>>>,
    pub final_leaf_chip_traces: Vec<DenseMatrix<Val<SC>>>,
    pub final_internal_chip_traces: Vec<DenseMatrix<Val<SC>>>,
}

#[derive(Clone)]
pub struct PageControllerMainTrace<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub init_root_signal_trace: DenseMatrix<Val<SC>>,
    pub init_leaf_chip_main_traces: Vec<DenseMatrix<Val<SC>>>,
    pub init_internal_chip_main_traces: Vec<DenseMatrix<Val<SC>>>,
    pub offline_checker_trace: DenseMatrix<Val<SC>>,
    pub final_root_signal_trace: DenseMatrix<Val<SC>>,
    pub final_leaf_chip_main_traces: Vec<DenseMatrix<Val<SC>>>,
    pub final_internal_chip_main_traces: Vec<DenseMatrix<Val<SC>>>,
}

pub struct PageControllerProverData<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub init_leaf_page: Vec<ProverTraceData<SC>>,
    pub init_internal_page: Vec<ProverTraceData<SC>>,
    pub final_leaf_page: Vec<ProverTraceData<SC>>,
    pub final_internal_page: Vec<ProverTraceData<SC>>,
}

#[derive(Clone)]
pub struct PageControllerCommit<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub init_leaf_page_commitments: Vec<Com<SC>>,
    pub init_internal_page_commitments: Vec<Com<SC>>,
    pub init_root_commitment: Com<SC>,
    pub final_leaf_page_commitments: Vec<Com<SC>>,
    pub final_internal_page_commitments: Vec<Com<SC>>,
    pub final_root_commitment: Com<SC>,
}

#[derive(Clone)]
pub struct PageControllerParams {
    pub idx_len: usize,
    pub data_len: usize,
    pub commitment_len: usize,
    pub init_tree_params: PageTreeParams,
    pub final_tree_params: PageTreeParams,
}

pub struct PageController<const COMMITMENT_LEN: usize> {
    pub init_root_signal: RootSignalAir<COMMITMENT_LEN>,
    pub init_leaf_chips: Vec<LeafPageAir<COMMITMENT_LEN>>,
    pub init_internal_chips: Vec<InternalPageAir<COMMITMENT_LEN>>,
    pub offline_checker: PageOfflineChecker,
    pub final_root_signal: RootSignalAir<COMMITMENT_LEN>,
    pub final_leaf_chips: Vec<LeafPageAir<COMMITMENT_LEN>>,
    pub final_internal_chips: Vec<InternalPageAir<COMMITMENT_LEN>>,
    pub params: PageControllerParams,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

#[allow(clippy::too_many_arguments)]
impl<const COMMITMENT_LEN: usize> PageController<COMMITMENT_LEN> {
    pub fn new<SC: StarkGenericConfig>(
        data_bus_index: usize,
        internal_data_bus_index: usize,
        ops_bus_index: usize,
        lt_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        init_param: PageTreeParams,
        final_param: PageTreeParams,
        less_than_tuple_param: MyLessThanTupleParams,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self
    where
        Val<SC>: AbstractField + PrimeField64,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        Self {
            init_leaf_chips: (0..init_param.leaf_cap)
                .map(|i| {
                    LeafPageAir::new(
                        init_param.path_bus_index,
                        data_bus_index,
                        less_than_tuple_param.clone(),
                        lt_bus_index,
                        idx_len,
                        data_len,
                        true,
                        i as u32,
                    )
                })
                .collect_vec(),
            init_internal_chips: (0..init_param.internal_cap)
                .map(|i| {
                    InternalPageAir::new(
                        init_param.path_bus_index,
                        internal_data_bus_index,
                        less_than_tuple_param.clone(),
                        lt_bus_index,
                        idx_len,
                        true,
                        i as u32,
                    )
                })
                .collect_vec(),
            offline_checker: PageOfflineChecker::new(
                data_bus_index,
                lt_bus_index,
                ops_bus_index,
                idx_len,
                data_len,
                less_than_tuple_param.limb_bits,
                Val::<SC>::bits() - 1,
                less_than_tuple_param.decomp,
            ),
            final_leaf_chips: (0..final_param.leaf_cap)
                .map(|i| {
                    LeafPageAir::new(
                        final_param.path_bus_index,
                        data_bus_index,
                        less_than_tuple_param.clone(),
                        lt_bus_index,
                        idx_len,
                        data_len,
                        false,
                        i as u32,
                    )
                })
                .collect_vec(),
            final_internal_chips: (0..final_param.internal_cap)
                .map(|i| {
                    InternalPageAir::new(
                        final_param.path_bus_index,
                        internal_data_bus_index,
                        less_than_tuple_param.clone(),
                        lt_bus_index,
                        idx_len,
                        false,
                        i as u32,
                    )
                })
                .collect_vec(),
            init_root_signal: RootSignalAir::new(init_param.path_bus_index, true, idx_len),
            final_root_signal: RootSignalAir::new(final_param.path_bus_index, false, idx_len),
            params: PageControllerParams {
                idx_len,
                data_len,
                commitment_len: COMMITMENT_LEN,
                init_tree_params: init_param,
                final_tree_params: final_param,
            },
            range_checker,
        }
    }

    fn gen_ops_trace<SC: StarkGenericConfig>(
        &self,
        mega_page: &mut Page,
        ops: &[Operation],
        range_checker: Arc<RangeCheckerGateChip>,
        trace_degree: usize,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        self.offline_checker.generate_trace::<SC>(
            mega_page,
            ops.to_owned(),
            range_checker,
            trace_degree,
        )
    }

    pub fn load_page_and_ops<SC: StarkGenericConfig>(
        &mut self,
        init_leaf_pages: Vec<Vec<Vec<u32>>>,
        init_internal_pages: Vec<Vec<Vec<u32>>>,
        init_root_is_leaf: bool,
        init_root_idx: usize,
        final_leaf_pages: Vec<Vec<Vec<u32>>>,
        final_internal_pages: Vec<Vec<Vec<u32>>>,
        final_root_is_leaf: bool,
        final_root_idx: usize,
        ops: Vec<Operation>,
        trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (
        PageControllerDataTrace<SC, COMMITMENT_LEN>,
        PageControllerMainTrace<SC, COMMITMENT_LEN>,
        PageControllerCommit<SC>,
        PageControllerProverData<SC>,
    )
    where
        Val<SC>: AbstractField + PrimeField64,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        let init_leaf_height = self.params.init_tree_params.leaf_page_height;
        let init_internal_height = self.params.init_tree_params.internal_page_height;
        let final_leaf_height = self.params.final_tree_params.leaf_page_height;
        let final_internal_height = self.params.final_tree_params.internal_page_height;

        let blank_init_leaf_row = vec![0; 1 + self.params.idx_len + self.params.data_len];
        let blank_init_leaf = vec![blank_init_leaf_row.clone(); init_leaf_height];

        let mut blank_init_internal_row = vec![2];
        blank_init_internal_row.resize(2 + 2 * self.params.idx_len + self.params.commitment_len, 0);
        let blank_init_internal = vec![blank_init_internal_row; init_internal_height];

        let blank_final_leaf_row = vec![0; 1 + self.params.idx_len + self.params.data_len];
        let blank_final_leaf = vec![blank_final_leaf_row.clone(); final_leaf_height];

        let mut blank_final_internal_row = vec![2];
        blank_final_internal_row
            .resize(2 + 2 * self.params.idx_len + self.params.commitment_len, 0);
        let blank_final_internal = vec![blank_final_internal_row; final_internal_height];
        let internal_indices = ops.iter().map(|op| op.idx.clone()).collect();
        let (init_tree_products, mega_page) = make_tree_products(
            trace_committer,
            &init_leaf_pages,
            &self.init_leaf_chips,
            &blank_init_leaf,
            &init_internal_pages,
            &self.init_internal_chips,
            &blank_init_internal,
            &self.init_root_signal,
            &self.params.init_tree_params,
            init_root_is_leaf,
            init_root_idx,
            self.params.idx_len,
            self.params.data_len,
            self.range_checker.clone(),
            &internal_indices,
            true,
        );
        let mut mega_page = mega_page.unwrap();
        mega_page.resize(
            mega_page.len() + 3 * ops.len(),
            vec![0; 1 + self.params.idx_len + self.params.data_len],
        );
        let (final_tree_products, _) = make_tree_products(
            trace_committer,
            &final_leaf_pages,
            &self.final_leaf_chips,
            &blank_final_leaf,
            &final_internal_pages,
            &self.final_internal_chips,
            &blank_final_internal,
            &self.final_root_signal,
            &self.params.final_tree_params,
            final_root_is_leaf,
            final_root_idx,
            self.params.idx_len,
            self.params.data_len,
            self.range_checker.clone(),
            &internal_indices,
            false,
        );
        let mut mega_page =
            Page::from_2d_vec(&mega_page, self.params.idx_len, self.params.data_len);
        let offline_checker_trace = self.gen_ops_trace::<SC>(
            &mut mega_page,
            &ops,
            self.range_checker.clone(),
            trace_degree,
        );

        let data_trace = PageControllerDataTrace {
            init_leaf_chip_traces: init_tree_products.leaf.data_traces,
            init_internal_chip_traces: init_tree_products.internal.data_traces,
            final_leaf_chip_traces: final_tree_products.leaf.data_traces,
            final_internal_chip_traces: final_tree_products.internal.data_traces,
        };
        let main_trace = PageControllerMainTrace {
            init_root_signal_trace: init_tree_products.root.main_traces,
            init_leaf_chip_main_traces: init_tree_products.leaf.main_traces,
            init_internal_chip_main_traces: init_tree_products.internal.main_traces,
            offline_checker_trace,
            final_root_signal_trace: final_tree_products.root.main_traces,
            final_leaf_chip_main_traces: final_tree_products.leaf.main_traces,
            final_internal_chip_main_traces: final_tree_products.internal.main_traces,
        };
        let commitments = PageControllerCommit {
            init_leaf_page_commitments: init_tree_products.leaf.commitments,
            init_internal_page_commitments: init_tree_products.internal.commitments,
            init_root_commitment: init_tree_products.root.commitments,
            final_leaf_page_commitments: final_tree_products.leaf.commitments,
            final_internal_page_commitments: final_tree_products.internal.commitments,
            final_root_commitment: final_tree_products.root.commitments,
        };
        let prover_data = PageControllerProverData {
            init_leaf_page: init_tree_products.leaf.prover_data,
            init_internal_page: init_tree_products.internal.prover_data,
            final_leaf_page: final_tree_products.leaf.prover_data,
            final_internal_page: final_tree_products.internal.prover_data,
        };
        (data_trace, main_trace, commitments, prover_data)
    }
}

#[allow(clippy::too_many_arguments)]
/// internal_indices are relevant for final page generation only
fn make_tree_products<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>(
    committer: &mut TraceCommitter<SC>,
    leaf_pages: &[Vec<Vec<u32>>],
    leaf_chips: &[LeafPageAir<COMMITMENT_LEN>],
    blank_leaf_page: &[Vec<u32>],
    internal_pages: &[Vec<Vec<u32>>],
    internal_chips: &[InternalPageAir<COMMITMENT_LEN>],
    blank_internal_page: &[Vec<u32>],
    root_signal: &RootSignalAir<COMMITMENT_LEN>,
    params: &PageTreeParams,
    root_is_leaf: bool,
    root_idx: usize,
    idx_len: usize,
    data_len: usize,
    range_checker: Arc<RangeCheckerGateChip>,
    internal_indices: &HashSet<Vec<u32>>,
    make_mega_page: bool,
) -> (TreeProducts<SC, COMMITMENT_LEN>, Option<Vec<Vec<u32>>>)
where
    Val<SC>: AbstractField + PrimeField64,
    Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
{
    let mut leaf_pages = leaf_pages.to_vec();
    let mut internal_pages = internal_pages.to_vec();
    leaf_pages.resize(params.leaf_cap, blank_leaf_page.to_vec());
    internal_pages.resize(params.internal_cap, blank_internal_page.to_vec());
    let leaf_trace = leaf_pages
        .iter()
        .zip(leaf_chips.iter())
        .map(|(page, chip)| {
            chip.generate_cached_trace::<Val<SC>>(Page::from_2d_vec(page, idx_len, data_len))
        })
        .collect::<Vec<_>>();

    let internal_trace = internal_pages
        .iter()
        .zip(internal_chips.iter())
        .map(|(page, chip)| chip.generate_cached_trace::<Val<SC>>(page.clone()))
        .collect::<Vec<_>>();

    let mut leaf_prods = gen_products(committer, leaf_trace);
    let mut internal_prods = gen_products(committer, internal_trace);

    let tree = PageTreeGraph::<SC, COMMITMENT_LEN>::new(
        &leaf_pages,
        &internal_pages,
        &leaf_prods.commitments,
        &internal_prods.commitments,
        (root_is_leaf, root_idx),
        idx_len,
    );
    for i in 0..leaf_prods.commitments.len() {
        let page = leaf_pages[i].clone();
        let range = tree.leaf_ranges[i].clone();
        let page = Page::from_2d_vec(&page, idx_len, data_len);
        let tmp = leaf_chips[i].generate_main_trace::<SC>(
            &page,
            range,
            range_checker.clone(),
            internal_indices,
        );
        leaf_prods.main_traces.push(tmp);
    }

    for i in 0..internal_prods.commitments.len() {
        let page = internal_pages[i].clone();
        let range = tree.internal_ranges[i].clone();
        let mults = tree.mults[i].clone();
        let child_ids = tree.child_ids[i].clone();
        let tmp = internal_chips[i].generate_main_trace::<Val<SC>>(
            page,
            child_ids,
            mults,
            range,
            range_checker.clone(),
        );
        internal_prods.main_traces.push(tmp);
    }
    let root_commitment = if root_is_leaf {
        leaf_prods.commitments[root_idx].clone()
    } else {
        internal_prods.commitments[root_idx].clone()
    };
    let root_signal_trace = root_signal.generate_trace::<Val<SC>>(
        root_idx as u32,
        tree.root_mult - 1,
        tree.root_range.clone(),
    );
    let root_prods = RootProducts {
        main_traces: root_signal_trace,
        commitments: root_commitment,
    };
    let mega_page = if make_mega_page {
        Some(tree.mega_page)
    } else {
        None
    };
    (
        TreeProducts {
            root: root_prods,
            leaf: leaf_prods,
            internal: internal_prods,
        },
        mega_page,
    )
}

fn gen_products<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>(
    committer: &mut TraceCommitter<SC>,
    trace: Vec<RowMajorMatrix<Val<SC>>>,
) -> NodeProducts<SC, COMMITMENT_LEN>
where
    Val<SC>: AbstractField + PrimeField64,
    Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
{
    let prover_data = data_from_trace(committer, &trace);

    let commitments = commitment_from_data(&prover_data);

    NodeProducts {
        data_traces: trace,
        main_traces: vec![],
        prover_data,
        commitments,
    }
}

fn data_from_trace<SC: StarkGenericConfig>(
    committer: &mut TraceCommitter<SC>,
    traces: &[RowMajorMatrix<Val<SC>>],
) -> Vec<ProverTraceData<SC>> {
    traces
        .iter()
        .map(|trace| committer.commit(vec![trace.clone()]))
        .collect::<Vec<_>>()
}

fn commitment_from_data<SC: StarkGenericConfig>(data: &[ProverTraceData<SC>]) -> Vec<Com<SC>> {
    data.iter()
        .map(|data| data.commit.clone())
        .collect::<Vec<_>>()
}
