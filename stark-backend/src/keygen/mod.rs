use itertools::Itertools;
use p3_air::BaseAir;
use p3_field::AbstractExtensionField;
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::instrument;

pub mod types;

use self::types::{
    create_commit_to_air_graph, MultiStarkProvingKey, ProverOnlySinglePreprocessedData,
    StarkProvingKey, StarkVerifyingKey, TraceWidth, VerifierSinglePreprocessedData,
};
use crate::{
    air_builders::symbolic::{get_symbolic_builder, SymbolicRapBuilder},
    commit::{MatrixCommitmentPointers, SingleMatrixCommitPtr},
    prover::trace::TraceCommitter,
    rap::AnyRap,
};

/// Stateful builder to create multi-stark proving and verifying keys
/// for system of multiple RAPs with multiple multi-matrix commitments
pub struct MultiStarkKeygenBuilder<'a, SC: StarkGenericConfig> {
    pub config: &'a SC,
    /// `placeholder_main_matrix_in_commit[commit_idx][mat_idx] =` matrix width, it is used to store
    /// a placeholder of a main trace matrix that must be committed during proving
    placeholder_main_matrix_in_commit: Vec<Vec<usize>>,
    /// Information for partitioned AIRs
    partitioned_airs: Vec<(&'a dyn AnyRap<SC>, usize, Vec<SingleMatrixCommitPtr>)>,
    /// Number of interactions to bundle in permutation trace
    interaction_chunk_size: Option<usize>,
}

impl<'a, SC: StarkGenericConfig> MultiStarkKeygenBuilder<'a, SC> {
    pub fn new(config: &'a SC) -> Self {
        Self {
            config,
            placeholder_main_matrix_in_commit: vec![vec![]],
            partitioned_airs: vec![],
            interaction_chunk_size: None,
        }
    }

    /// Set the number of interactions to bundle in permutation trace
    pub fn set_interaction_chunk_size(&mut self, size: usize) {
        self.interaction_chunk_size = Some(size);
    }

    /// Generates proving key, resetting the state of the builder.
    /// The verifying key can be obtained from the proving key.
    pub fn generate_pk(&mut self) -> MultiStarkProvingKey<SC> {
        if self.interaction_chunk_size.is_none() {
            // If this interaction_chunk_size is not set, use the following as a default
            // so that logup constraints degree matches max AIR constraint degree, assuming
            // `fields` and `count` are of degree 1 in all interactions
            self.interaction_chunk_size = Some(self.all_airs_max_constraint_degree() - 1);
        }

        let interaction_chunk_size = self.interaction_chunk_size.unwrap();
        let mut multi_pk = MultiStarkProvingKey::empty();

        let partitioned_airs = std::mem::take(&mut self.partitioned_airs);
        for (air, num_public_values, partitioned_main_ptrs) in partitioned_airs.into_iter() {
            let (prep_prover_data, prep_verifier_data, symbolic_builder) = self
                .get_prep_data_and_symbolic_builder(
                    air,
                    num_public_values,
                    &partitioned_main_ptrs,
                    interaction_chunk_size,
                );

            let params = symbolic_builder.params();
            let symbolic_constraints = symbolic_builder.constraints();

            let log_quotient_degree = symbolic_constraints.get_log_quotient_degree();
            let quotient_degree = 1 << log_quotient_degree;

            let vk = StarkVerifyingKey {
                preprocessed_data: prep_verifier_data,
                params,
                symbolic_constraints,
                main_graph: MatrixCommitmentPointers::new(partitioned_main_ptrs),
                quotient_degree,
                interaction_chunk_size,
            };
            let pk = StarkProvingKey {
                air_name: air.name(),
                vk,
                preprocessed_data: prep_prover_data,
                interaction_chunk_size,
            };

            multi_pk.per_air.push(pk);
        }

        // Determine global num challenges to sample
        let num_phases = multi_pk
            .per_air
            .iter()
            .map(|pk| {
                // Consistency check
                let num = pk.vk.width().after_challenge.len();
                assert_eq!(num, pk.vk.params.num_challenges_to_sample.len());
                assert_eq!(num, pk.vk.params.num_exposed_values_after_challenge.len());
                num
            })
            .max()
            .unwrap_or(0);
        multi_pk.num_challenges_to_sample = (0..num_phases)
            .map(|phase_idx| {
                multi_pk
                    .per_air
                    .iter()
                    .map(|pk| {
                        *pk.vk
                            .params
                            .num_challenges_to_sample
                            .get(phase_idx)
                            .unwrap_or(&0)
                    })
                    .max()
                    .unwrap_or_else(|| panic!("No challenges used in challenge phase {phase_idx}"))
            })
            .collect();

        if matches!(self.placeholder_main_matrix_in_commit.last(), Some(mats) if mats.is_empty()) {
            self.placeholder_main_matrix_in_commit.pop();
        }
        multi_pk.num_main_trace_commitments = self.placeholder_main_matrix_in_commit.len();
        // Build commit->air graph
        let air_matrices = multi_pk
            .per_air
            .iter()
            .map(|pk| pk.vk.main_graph.clone())
            .collect_vec();
        multi_pk.main_commit_to_air_graph =
            create_commit_to_air_graph(&air_matrices, multi_pk.num_main_trace_commitments);
        // reset state
        self.placeholder_main_matrix_in_commit = vec![vec![]];
        self.interaction_chunk_size = None;

        for pk in multi_pk.per_air.iter() {
            let width = pk.vk.width();
            tracing::info!("{:<20} | Quotient Deg = {:<2} | Prep Cols = {:<2} | Main Cols = {:<8} | Perm Cols = {:<4} | {:<4} Constraints | {:<3} Interactions On Buses {:?}",
                pk.air_name,
                pk.vk.quotient_degree,
                width.preprocessed.unwrap_or(0),
                format!("{:?}",width.partitioned_main),
                format!("{:?}",width.after_challenge.iter().map(|&x| x * <SC::Challenge as AbstractExtensionField<Val<SC>>>::D).collect_vec()),
                pk.vk.symbolic_constraints.constraints.len(),
                pk.vk.symbolic_constraints.interactions.len(),
                pk.vk
                    .symbolic_constraints
                    .interactions
                    .iter()
                    .map(|i| i.bus_index)
                    .collect_vec()
            );
        }

        multi_pk
    }

    /// Creates abstract placeholder matrix and adds to current last trace commitment
    pub fn add_main_matrix(&mut self, width: usize) -> SingleMatrixCommitPtr {
        let commit_idx = self.placeholder_main_matrix_in_commit.len() - 1;
        let mats = self.placeholder_main_matrix_in_commit.last_mut().unwrap();
        let matrix_idx = mats.len();
        mats.push(width);
        SingleMatrixCommitPtr::new(commit_idx, matrix_idx)
    }

    /// Seals the current main trace commitment and starts a new one
    pub fn seal_current_main_commitment(&mut self) {
        self.placeholder_main_matrix_in_commit.push(vec![]);
    }

    /// Adds a single matrix to dedicated commitment and starts new commitment
    pub fn add_cached_main_matrix(&mut self, width: usize) -> SingleMatrixCommitPtr {
        assert!(
            matches!(self.placeholder_main_matrix_in_commit.last(), Some(mats) if mats.is_empty()),
            "Current commitment non-empty: cache may not have desired effect"
        );
        let ptr = self.add_main_matrix(width);
        self.seal_current_main_commitment();
        ptr
    }

    /// Default way to add a single Interactive AIR.
    /// DO NOT use this if the main trace needs to be partitioned.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - Adds main trace to the last main trace commitment.
    #[instrument(level = "debug", skip_all)]
    pub fn add_air(&mut self, air: &'a dyn AnyRap<SC>, num_public_values: usize) {
        let main_width = <dyn AnyRap<SC> as BaseAir<Val<SC>>>::width(air);
        let ptr = self.add_main_matrix(main_width);
        self.add_partitioned_air(air, num_public_values, vec![ptr]);
    }

    /// Add a single Interactive AIR with partitioned main trace.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - The matrix pointers for partitioned main trace must be manually created ahead of time.
    /// - `partitioned_main` is a list of (width, matrix_ptr) pairs.
    #[instrument(level = "debug", skip_all)]
    pub fn add_partitioned_air(
        &mut self,
        air: &'a dyn AnyRap<SC>,
        num_public_values: usize,
        partitioned_main_ptrs: Vec<SingleMatrixCommitPtr>,
    ) {
        self.partitioned_airs
            .push((air, num_public_values, partitioned_main_ptrs));
    }

    /// Default way to add a single Interactive AIR.
    /// DO NOT use this if the main trace needs to be partitioned.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - Adds main trace to the default shared main trace commitment.
    #[instrument(level = "debug", skip_all)]
    pub fn get_single_preprocessed_data(
        &self,
        air: &dyn AnyRap<SC>,
    ) -> Option<(
        ProverOnlySinglePreprocessedData<SC>,
        VerifierSinglePreprocessedData<SC>,
    )> {
        let pcs = self.config.pcs();
        let preprocessed_trace = air.preprocessed_trace();
        preprocessed_trace.map(|trace| {
            let trace_committer = TraceCommitter::<SC>::new(pcs);
            let data = trace_committer.commit(vec![trace.clone()]);
            let vdata = VerifierSinglePreprocessedData {
                commit: data.commit,
            };
            let pdata = ProverOnlySinglePreprocessedData {
                trace,
                data: data.data,
            };
            (pdata, vdata)
        })
    }

    fn all_airs_max_constraint_degree(&mut self) -> usize {
        let mut max_constraint_degree = 0;
        for (air, num_public_values, partitioned_main_ptrs) in self.partitioned_airs.iter() {
            let (_, _, symbolic_builder) = self.get_prep_data_and_symbolic_builder(
                *air,
                *num_public_values,
                partitioned_main_ptrs,
                1,
            );

            let symbolic_constraints = symbolic_builder.constraints();

            max_constraint_degree =
                max_constraint_degree.max(symbolic_constraints.max_constraint_degree());
        }

        max_constraint_degree
    }

    #[allow(clippy::type_complexity)]
    fn get_prep_data_and_symbolic_builder(
        &self,
        air: &dyn AnyRap<SC>,
        num_public_values: usize,
        partitioned_main_ptrs: &[SingleMatrixCommitPtr],
        interaction_chunk_size: usize,
    ) -> (
        Option<ProverOnlySinglePreprocessedData<SC>>,
        Option<VerifierSinglePreprocessedData<SC>>,
        SymbolicRapBuilder<Val<SC>>,
    ) {
        let (prep_prover_data, prep_verifier_data): (Option<_>, Option<_>) =
            self.get_single_preprocessed_data(air).unzip();
        let preprocessed_width = prep_prover_data.as_ref().map(|d| d.trace.width());

        let main_widths = partitioned_main_ptrs
            .iter()
            .map(|ptr| self.placeholder_main_matrix_in_commit[ptr.commit_index][ptr.matrix_index])
            .collect();
        let width = TraceWidth {
            preprocessed: preprocessed_width,
            partitioned_main: main_widths,
            after_challenge: vec![],
        };
        let symbolic_builder = get_symbolic_builder(
            air,
            &width,
            num_public_values,
            &[],
            &[],
            interaction_chunk_size,
        );

        (prep_prover_data, prep_verifier_data, symbolic_builder)
    }
}
