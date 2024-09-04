use std::{collections::HashSet, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    engine::StarkEngine,
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    prover::{
        trace::{ProverTraceData, TraceCommitmentBuilder, TraceCommitter},
        types::Proof,
    },
    rap::AnyRap,
    verifier::VerificationError,
};
use itertools::Itertools;
use p3_field::{AbstractField, Field, PrimeField, PrimeField64};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::info_span;

use self::page_tree_graph::PageTreeGraph;
use super::{
    internal_page_air::InternalPageAir, leaf_page_air::LeafPageAir, root_signal_air::RootSignalAir,
};
use crate::{
    common::{page::Page, page_cols::PageCols},
    page_rw_checker::{offline_checker::PageOfflineChecker, page_controller::Operation},
};

pub mod page_tree_graph;

#[derive(Clone)]
pub struct PageTreeParams {
    pub path_bus_index: usize,
    pub leaf_cap: Option<usize>,
    pub internal_cap: Option<usize>,
    pub leaf_page_height: usize,
    pub internal_page_height: usize,
}

#[derive(Clone)]
pub struct MyLessThanTupleParams {
    pub limb_bits: usize,
    pub decomp: usize,
}

pub struct MultitierCapacities {
    pub init_leaf_cap: Option<usize>,
    pub init_internal_cap: Option<usize>,
    pub final_leaf_cap: Option<usize>,
    pub final_internal_cap: Option<usize>,
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

pub struct NodeProducts<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>
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

pub struct PageController<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub init_root_signal: RootSignalAir<COMMITMENT_LEN>,
    pub init_leaf_chips: Vec<LeafPageAir<COMMITMENT_LEN>>,
    pub init_internal_chips: Vec<InternalPageAir<COMMITMENT_LEN>>,
    pub offline_checker: PageOfflineChecker,
    pub final_root_signal: RootSignalAir<COMMITMENT_LEN>,
    pub final_leaf_chips: Vec<LeafPageAir<COMMITMENT_LEN>>,
    pub final_internal_chips: Vec<InternalPageAir<COMMITMENT_LEN>>,
    pub params: PageControllerParams,
    pub range_checker: Arc<RangeCheckerGateChip>,
    main_traces: Option<PageControllerMainTrace<SC, COMMITMENT_LEN>>,
    data_traces: Option<PageControllerDataTrace<SC, COMMITMENT_LEN>>,
    commits: Option<PageControllerCommit<SC>>,
}

#[allow(clippy::too_many_arguments)]
impl<SC: StarkGenericConfig, const COMMITMENT_LEN: usize> PageController<SC, COMMITMENT_LEN>
where
    Val<SC>: AbstractField + PrimeField64,
    Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
{
    pub fn new(
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
    ) -> Self {
        Self {
            init_leaf_chips: (0..init_param.leaf_cap.unwrap_or(1))
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
            init_internal_chips: (0..init_param.internal_cap.unwrap_or(1))
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
            final_leaf_chips: (0..final_param.leaf_cap.unwrap_or(1))
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
            final_internal_chips: (0..final_param.internal_cap.unwrap_or(1))
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
            main_traces: None,
            data_traces: None,
            commits: None,
        }
    }

    fn gen_ops_trace(
        &self,
        mega_page: &mut Page,
        ops: &[Operation],
        range_checker: Arc<RangeCheckerGateChip>,
        trace_degree: usize,
    ) -> RowMajorMatrix<Val<SC>> {
        self.offline_checker.generate_trace::<SC>(
            mega_page,
            ops.to_owned(),
            range_checker,
            trace_degree,
        )
    }

    pub fn load_page_and_ops(
        &mut self,
        init_leaf_pages: Vec<Vec<Vec<u32>>>,
        init_internal_pages: Vec<Vec<Vec<u32>>>,
        init_root_is_leaf: bool,
        init_root_idx: usize,
        final_leaf_pages: Vec<Vec<Vec<u32>>>,
        final_internal_pages: Vec<Vec<Vec<u32>>>,
        final_root_is_leaf: bool,
        final_root_idx: usize,
        ops: &[Operation],
        trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
        init_cached_data: Option<(
            NodeProducts<SC, COMMITMENT_LEN>,
            NodeProducts<SC, COMMITMENT_LEN>,
        )>,
        final_cached_data: Option<(
            NodeProducts<SC, COMMITMENT_LEN>,
            NodeProducts<SC, COMMITMENT_LEN>,
        )>,
    ) -> PageControllerProverData<SC> {
        let trace_span = info_span!("Load page trace generation").entered();
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
        let init_leaf_pages = init_leaf_pages
            .into_iter()
            .map(|p| Page::from_2d_vec_consume(p, self.params.idx_len, self.params.data_len))
            .collect_vec();
        let final_leaf_pages = final_leaf_pages
            .into_iter()
            .map(|p| Page::from_2d_vec_consume(p, self.params.idx_len, self.params.data_len))
            .collect_vec();
        let blank_init_leaf =
            Page::from_2d_vec_consume(blank_init_leaf, self.params.idx_len, self.params.data_len);
        let blank_final_leaf =
            Page::from_2d_vec_consume(blank_final_leaf, self.params.idx_len, self.params.data_len);
        let (init_tree_products, mega_page) = make_tree_products(
            trace_committer,
            init_leaf_pages,
            &mut self.init_leaf_chips,
            blank_init_leaf,
            init_internal_pages,
            &mut self.init_internal_chips,
            blank_init_internal,
            &self.init_root_signal,
            &self.params.init_tree_params,
            init_root_is_leaf,
            init_root_idx,
            self.params.idx_len,
            self.params.data_len,
            self.range_checker.clone(),
            &internal_indices,
            true,
            init_cached_data,
        );
        let mut mega_page = mega_page.unwrap();
        mega_page.rows.resize(
            mega_page.rows.len() + 3 * ops.len(),
            PageCols::new(
                0,
                vec![0; self.params.idx_len],
                vec![0; self.params.data_len],
            ),
        );
        let (final_tree_products, _) = make_tree_products(
            trace_committer,
            final_leaf_pages,
            &mut self.final_leaf_chips,
            blank_final_leaf,
            final_internal_pages,
            &mut self.final_internal_chips,
            blank_final_internal,
            &self.final_root_signal,
            &self.params.final_tree_params,
            final_root_is_leaf,
            final_root_idx,
            self.params.idx_len,
            self.params.data_len,
            self.range_checker.clone(),
            &internal_indices,
            false,
            final_cached_data,
        );
        let offline_checker_span = info_span!("Ops Trace Generation").entered();
        let offline_checker_trace = self.gen_ops_trace(
            &mut mega_page,
            ops,
            self.range_checker.clone(),
            trace_degree,
        );
        offline_checker_span.exit();

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
        trace_span.exit();
        self.main_traces = Some(main_trace);
        self.data_traces = Some(data_trace);
        self.commits = Some(commitments);
        prover_data
    }

    pub fn set_up_keygen_builder<'a>(
        &'a self,
        keygen_builder: &mut MultiStarkKeygenBuilder<'a, SC>,
        ops_sender: &'a dyn AnyRap<SC>,
    ) {
        let mut init_leaf_data_ptrs = vec![];

        let mut init_internal_data_ptrs = vec![];
        let mut init_internal_main_ptrs = vec![];

        let mut final_leaf_data_ptrs = vec![];
        let mut final_leaf_main_ptrs = vec![];

        let mut final_internal_data_ptrs = vec![];
        let mut final_internal_main_ptrs = vec![];

        for _ in 0..self.init_leaf_chips.len() {
            init_leaf_data_ptrs.push(
                keygen_builder.add_cached_main_matrix(self.init_leaf_chips[0].cached_width()),
            );
        }

        for _ in 0..self.init_internal_chips.len() {
            init_internal_data_ptrs.push(
                keygen_builder.add_cached_main_matrix(self.init_internal_chips[0].cached_width()),
            );
        }

        for _ in 0..self.final_leaf_chips.len() {
            final_leaf_data_ptrs.push(
                keygen_builder.add_cached_main_matrix(self.final_leaf_chips[0].cached_width()),
            );
        }

        for _ in 0..self.final_internal_chips.len() {
            final_internal_data_ptrs.push(
                keygen_builder.add_cached_main_matrix(self.final_internal_chips[0].cached_width()),
            );
        }

        for _ in 0..self.init_internal_chips.len() {
            init_internal_main_ptrs
                .push(keygen_builder.add_main_matrix(self.init_internal_chips[0].main_width()));
        }

        for _ in 0..self.final_leaf_chips.len() {
            final_leaf_main_ptrs
                .push(keygen_builder.add_main_matrix(self.final_leaf_chips[0].main_width()));
        }

        for _ in 0..self.final_internal_chips.len() {
            final_internal_main_ptrs
                .push(keygen_builder.add_main_matrix(self.final_internal_chips[0].main_width()));
        }

        let ops_ptr = keygen_builder.add_main_matrix(self.offline_checker.air_width());

        let init_root_ptr = keygen_builder.add_main_matrix(self.init_root_signal.air_width());
        let final_root_ptr = keygen_builder.add_main_matrix(self.final_root_signal.air_width());

        for (chip, ptr) in self
            .init_leaf_chips
            .iter()
            .zip(init_leaf_data_ptrs.into_iter())
        {
            keygen_builder.add_partitioned_air(chip, COMMITMENT_LEN, vec![ptr]);
        }

        for i in 0..self.init_internal_chips.len() {
            keygen_builder.add_partitioned_air(
                &self.init_internal_chips[i],
                COMMITMENT_LEN,
                vec![init_internal_data_ptrs[i], init_internal_main_ptrs[i]],
            );
        }

        for i in 0..self.final_leaf_chips.len() {
            keygen_builder.add_partitioned_air(
                &self.final_leaf_chips[i],
                COMMITMENT_LEN,
                vec![final_leaf_data_ptrs[i], final_leaf_main_ptrs[i]],
            );
        }

        for i in 0..self.final_internal_chips.len() {
            keygen_builder.add_partitioned_air(
                &self.final_internal_chips[i],
                COMMITMENT_LEN,
                vec![final_internal_data_ptrs[i], final_internal_main_ptrs[i]],
            );
        }

        keygen_builder.add_partitioned_air(&self.offline_checker, 0, vec![ops_ptr]);

        keygen_builder.add_partitioned_air(
            &self.init_root_signal,
            COMMITMENT_LEN,
            vec![init_root_ptr],
        );

        keygen_builder.add_partitioned_air(
            &self.final_root_signal,
            COMMITMENT_LEN,
            vec![final_root_ptr],
        );

        keygen_builder.add_air(&self.range_checker.air, 0);

        keygen_builder.add_air(ops_sender, 0);
    }
    /// This function clears the trace_builder, loads in the traces for all involved chips
    /// (including the range_checker and the ops_sender, which is passed in along with its trace),
    /// commits them, and then generates the proof.
    /// cached_traces_prover_data is a vector of ProverTraceData object for the cached pages
    /// (init_page, final_page), which is returned by load_page_and_ops
    #[allow(clippy::too_many_arguments)]
    pub fn prove(
        &mut self,
        engine: &impl StarkEngine<SC>,
        pk: &MultiStarkProvingKey<SC>,
        trace_builder: &mut TraceCommitmentBuilder<SC>,
        prover_data: PageControllerProverData<SC>,
        ops_sender: &dyn AnyRap<SC>,
        ops_sender_trace: DenseMatrix<Val<SC>>,
    ) -> (Proof<SC>, Vec<Vec<Val<SC>>>)
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
        let main_trace = self.main_traces.take().unwrap();
        let data_trace = self.data_traces.take().unwrap();
        let commits = self.commits.take().unwrap();

        trace_builder.clear();

        let offline_checker_trace = main_trace.offline_checker_trace;
        let init_root = main_trace.init_root_signal_trace;
        let final_root = main_trace.final_root_signal_trace;
        let range_trace = self.range_checker.generate_trace();

        for (tr, pd) in data_trace
            .init_leaf_chip_traces
            .into_iter()
            .zip_eq(prover_data.init_leaf_page)
        {
            trace_builder.load_cached_trace(tr, pd);
        }

        for (tr, pd) in data_trace
            .init_internal_chip_traces
            .into_iter()
            .zip_eq(prover_data.init_internal_page)
        {
            trace_builder.load_cached_trace(tr, pd);
        }

        for (tr, pd) in data_trace
            .final_leaf_chip_traces
            .into_iter()
            .zip_eq(prover_data.final_leaf_page)
        {
            trace_builder.load_cached_trace(tr, pd);
        }

        for (tr, pd) in data_trace
            .final_internal_chip_traces
            .into_iter()
            .zip_eq(prover_data.final_internal_page)
        {
            trace_builder.load_cached_trace(tr, pd);
        }
        for tr in main_trace.init_internal_chip_main_traces.into_iter() {
            trace_builder.load_trace(tr);
        }

        for tr in main_trace.final_leaf_chip_main_traces.into_iter() {
            trace_builder.load_trace(tr);
        }

        for tr in main_trace.final_internal_chip_main_traces.into_iter() {
            trace_builder.load_trace(tr);
        }
        trace_builder.load_trace(offline_checker_trace);
        trace_builder.load_trace(init_root);
        trace_builder.load_trace(final_root);
        trace_builder.load_trace(range_trace);
        trace_builder.load_trace(ops_sender_trace);

        tracing::info_span!("Prove trace commitment").in_scope(|| trace_builder.commit_current());

        let mut airs: Vec<&dyn AnyRap<SC>> = vec![];
        for chip in &self.init_leaf_chips {
            airs.push(chip);
        }
        for chip in &self.init_internal_chips {
            airs.push(chip);
        }
        for chip in &self.final_leaf_chips {
            airs.push(chip);
        }
        for chip in &self.final_internal_chips {
            airs.push(chip);
        }
        airs.push(&self.offline_checker);
        airs.push(&self.init_root_signal);
        airs.push(&self.final_root_signal);
        airs.push(&self.range_checker.air);
        airs.push(ops_sender);
        let vk = pk.vk();
        let main_trace_data = trace_builder.view(&vk, airs.clone());

        let mut pis = vec![];
        for c in commits.init_leaf_page_commitments {
            let c: [Val<SC>; COMMITMENT_LEN] = c.into();
            pis.push(c.to_vec());
        }
        for c in commits.init_internal_page_commitments {
            let c: [Val<SC>; COMMITMENT_LEN] = c.into();
            pis.push(c.to_vec());
        }
        for c in commits.final_leaf_page_commitments {
            let c: [Val<SC>; COMMITMENT_LEN] = c.into();
            pis.push(c.to_vec());
        }
        for c in commits.final_internal_page_commitments {
            let c: [Val<SC>; COMMITMENT_LEN] = c.into();
            pis.push(c.to_vec());
        }
        pis.push(vec![]);
        {
            let c: [Val<SC>; COMMITMENT_LEN] = commits.init_root_commitment.into();
            pis.push(c.to_vec());
        }
        {
            let c: [Val<SC>; COMMITMENT_LEN] = commits.final_root_commitment.into();
            pis.push(c.to_vec());
        }
        pis.push(vec![]);
        pis.push(vec![]);
        let prover = engine.prover();
        let mut challenger = engine.new_challenger();
        (
            prover.prove(&mut challenger, pk, main_trace_data, &pis),
            pis,
        )
    }

    /// This function takes a proof (returned by the prove function) and verifies it
    pub fn verify(
        &self,
        engine: &impl StarkEngine<SC>,
        vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
        pis: &[Vec<Val<SC>>],
    ) -> Result<(), VerificationError>
    where
        Val<SC>: PrimeField,
    {
        let verifier = engine.verifier();

        let mut challenger = engine.new_challenger();
        verifier.verify(&mut challenger, vk, proof, pis)
    }

    pub fn airs<'a>(&'a self, ops_sender: &'a dyn AnyRap<SC>) -> Vec<&'a dyn AnyRap<SC>> {
        let mut airs: Vec<&dyn AnyRap<SC>> = vec![];
        for chip in &self.init_leaf_chips {
            airs.push(chip);
        }
        for chip in &self.init_internal_chips {
            airs.push(chip);
        }
        for chip in &self.final_leaf_chips {
            airs.push(chip);
        }
        for chip in &self.final_internal_chips {
            airs.push(chip);
        }
        airs.push(&self.offline_checker);
        airs.push(&self.init_root_signal);
        airs.push(&self.final_root_signal);
        airs.push(&self.range_checker.air);
        airs.push(ops_sender);
        airs
    }
}

#[allow(clippy::too_many_arguments)]
/// internal_indices are relevant for final page generation only
fn make_tree_products<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>(
    committer: &mut TraceCommitter<SC>,
    leaf_pages: Vec<Page>,
    leaf_chips: &mut Vec<LeafPageAir<COMMITMENT_LEN>>,
    blank_leaf_page: Page,
    internal_pages: Vec<Vec<Vec<u32>>>,
    internal_chips: &mut Vec<InternalPageAir<COMMITMENT_LEN>>,
    blank_internal_page: Vec<Vec<u32>>,
    root_signal: &RootSignalAir<COMMITMENT_LEN>,
    params: &PageTreeParams,
    root_is_leaf: bool,
    root_idx: usize,
    idx_len: usize,
    data_len: usize,
    range_checker: Arc<RangeCheckerGateChip>,
    internal_indices: &HashSet<Vec<u32>>,
    make_mega_page: bool,
    cached_data: Option<(
        NodeProducts<SC, COMMITMENT_LEN>,
        NodeProducts<SC, COMMITMENT_LEN>,
    )>,
) -> (TreeProducts<SC, COMMITMENT_LEN>, Option<Page>)
where
    Val<SC>: AbstractField + PrimeField64,
    Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
{
    let mut leaf_pages = leaf_pages;
    let mut internal_pages = internal_pages;
    if let (Some(leaf_cap), Some(internal_cap)) = (params.leaf_cap, params.internal_cap) {
        leaf_pages.resize(leaf_cap, blank_leaf_page.clone());
        internal_pages.resize(internal_cap, blank_internal_page.to_vec());
    } else {
        if internal_pages.is_empty() {
            internal_pages.push(blank_internal_page.to_vec());
        }
        leaf_chips.truncate(leaf_pages.len());
        for i in leaf_chips.len()..leaf_pages.len() {
            leaf_chips.push(leaf_chips[0].clone_with_id(i as u32));
        }
        for i in internal_chips.len()..internal_pages.len() {
            internal_chips.push(internal_chips[0].clone_with_id(i as u32));
        }
    }

    let leaf_trace = leaf_pages
        .iter()
        .zip(leaf_chips.iter())
        .map(|(page, chip)| chip.generate_cached_trace_from_page::<Val<SC>>(page))
        .collect::<Vec<_>>();

    let internal_trace = internal_pages
        .iter()
        .zip(internal_chips.iter())
        .map(|(page, chip)| chip.generate_cached_trace::<Val<SC>>(page))
        .collect::<Vec<_>>();
    let (mut leaf_prods, mut internal_prods) = if cached_data.is_some() {
        let mut data = cached_data.unwrap();
        data.0.data_traces = leaf_trace;
        data.1.data_traces = internal_trace;
        (data.0, data.1)
    } else {
        (
            gen_products(committer, leaf_trace),
            gen_products(committer, internal_trace),
        )
    };
    let tree_span = info_span!("Tree DFS").entered();
    let tree = PageTreeGraph::<SC, COMMITMENT_LEN>::new(
        &leaf_pages,
        &internal_pages,
        internal_indices,
        &leaf_prods.commitments,
        &internal_prods.commitments,
        (root_is_leaf, root_idx),
        idx_len,
        data_len,
    );
    tree_span.exit();
    let main_trace_span = info_span!("Main Trace Generation").entered();
    for (i, page) in leaf_pages.into_iter().enumerate() {
        let range = tree.leaf_ranges[i].clone();
        let tmp = leaf_chips[i].generate_main_trace::<SC>(
            page,
            range,
            range_checker.clone(),
            internal_indices,
        );
        leaf_prods.main_traces.push(tmp);
    }

    for (i, page) in internal_pages.into_iter().enumerate() {
        let range = tree.internal_ranges[i].clone();
        let tmp = internal_chips[i].generate_main_trace::<Val<SC>>(
            page,
            &tree.child_ids[i],
            &tree.mults[i],
            range,
            range_checker.clone(),
        );
        internal_prods.main_traces.push(tmp);
    }
    main_trace_span.exit();
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

pub fn gen_some_products_from_prover_data<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>(
    data: Vec<ProverTraceData<SC>>,
) -> NodeProducts<SC, COMMITMENT_LEN>
where
    Val<SC>: AbstractField + PrimeField64,
    Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
{
    let commitments = commitment_from_data(&data);

    NodeProducts {
        data_traces: vec![],
        main_traces: vec![],
        prover_data: data,
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

pub fn commitment_from_data<SC: StarkGenericConfig>(data: &[ProverTraceData<SC>]) -> Vec<Com<SC>> {
    data.iter()
        .map(|data| data.commit.clone())
        .collect::<Vec<_>>()
}
