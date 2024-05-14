use itertools::Itertools;
use p3_matrix::Matrix;
use p3_uni_stark::StarkGenericConfig;
use tracing::instrument;

pub mod types;

use crate::{
    air_builders::symbolic::get_log_quotient_degree,
    commit::{MatrixCommitmentGraph, SingleMatrixCommitPtr},
    prover::trace::TraceCommitter,
};

use self::types::{
    create_commit_to_air_graph, MultiStarkProvingKey, ProverOnlySinglePreprocessedData,
    StarkProvingKey, StarkVerifyingKey, SymbolicRap, TraceWidth, VerifierSinglePreprocessedData,
};

/// Constants for interactive AIRs
const NUM_PERM_CHALLENGES: usize = 2;
const NUM_PERM_EXPOSED_VALUES: usize = 1;

/// Stateful builder to create multi-stark proving and verifying keys
/// for system of multiple RAPs with multiple multi-matrix commitments
pub struct MultiStarkKeygenBuilder<'a, SC: StarkGenericConfig> {
    pub config: &'a SC,
    /// Tracks how many matrices are in the i-th main trace commitment
    num_mats_in_main_commit: Vec<usize>,
    pk: MultiStarkProvingKey<SC>,
}

impl<'a, SC: StarkGenericConfig> MultiStarkKeygenBuilder<'a, SC> {
    pub fn new(config: &'a SC) -> Self {
        Self {
            config,
            pk: MultiStarkProvingKey::empty(),
            num_mats_in_main_commit: vec![0],
        }
    }

    /// Generates proving key, resetting the state of the builder.
    /// The verifying key can be obtained from the proving key.
    pub fn generate_pk(&mut self) -> MultiStarkProvingKey<SC> {
        let mut pk = std::mem::take(&mut self.pk);
        // Determine global num challenges to sample
        let num_phases = pk
            .per_air
            .iter()
            .map(|pk| {
                // Consistency check
                let num = pk.vk.width.after_challenge.len();
                assert_eq!(num, pk.vk.num_challenges_to_sample.len());
                assert_eq!(num, pk.vk.num_exposed_values_after_challenge.len());
                num
            })
            .max()
            .unwrap_or(0);
        pk.num_challenges_to_sample = (0..num_phases)
            .map(|phase_idx| {
                pk.per_air
                    .iter()
                    .map(|pk| *pk.vk.num_challenges_to_sample.get(phase_idx).unwrap_or(&0))
                    .max()
                    .unwrap_or_else(|| panic!("No challenges used in challenge phase {phase_idx}"))
            })
            .collect();

        if let Some(&0) = self.num_mats_in_main_commit.last() {
            self.num_mats_in_main_commit.pop();
        }
        pk.num_main_trace_commitments = self.num_mats_in_main_commit.len();
        // Build commit->air graph
        let air_matrices = pk
            .per_air
            .iter()
            .map(|pk| pk.vk.main_graph.clone())
            .collect_vec();
        pk.main_commit_to_air_graph =
            create_commit_to_air_graph(&air_matrices, pk.num_main_trace_commitments);
        // reset state
        self.num_mats_in_main_commit = vec![0];

        pk
    }

    /// Creates abstract placeholder matrix and adds to current last trace commitment
    pub fn add_main_matrix(&mut self) -> SingleMatrixCommitPtr {
        let commit_idx = self.num_mats_in_main_commit.len() - 1;
        let matrix_idx = self.num_mats_in_main_commit[commit_idx];
        self.num_mats_in_main_commit[commit_idx] += 1;
        SingleMatrixCommitPtr::new(commit_idx, matrix_idx)
    }

    /// Seals the current main trace commitment and starts a new one
    pub fn seal_current_main_commitment(&mut self) {
        self.num_mats_in_main_commit.push(0);
    }

    /// Adds a single matrix to dedicated commitment and starts new commitment
    pub fn add_cached_main_matrix(&mut self) -> SingleMatrixCommitPtr {
        assert_eq!(
            self.num_mats_in_main_commit.last(),
            Some(&0),
            "Current commitment non-empty: cache may not have desired effect"
        );
        let ptr = self.add_main_matrix();
        self.seal_current_main_commitment();
        ptr
    }

    /// Default way to add a single Interactive AIR.
    /// DO NOT use this if the main trace needs to be partitioned.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - Adds main trace to the last main trace commitment.
    #[instrument(level = "debug", skip_all)]
    pub fn add_air(&mut self, air: &dyn SymbolicRap<SC>, degree: usize, num_public_values: usize) {
        let main_width = air.width();
        let ptr = self.add_main_matrix();
        self.add_partitioned_air(air, degree, num_public_values, vec![(main_width, ptr)]);
    }

    /// Add a single Interactive AIR with partitioned main trace.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - The matrix pointers for partitioned main trace must be manually created ahead of time.
    /// - `partitioned_main` is a list of (width, matrix_ptr) pairs.
    #[instrument(level = "debug", skip_all)]
    pub fn add_partitioned_air(
        &mut self,
        air: &dyn SymbolicRap<SC>,
        degree: usize,
        num_public_values: usize,
        partitioned_main: Vec<(usize, SingleMatrixCommitPtr)>,
    ) {
        let (prep_prover_data, prep_verifier_data): (Option<_>, Option<_>) =
            self.get_single_preprocessed_data(air).unzip();
        let preprocessed_width = prep_prover_data.as_ref().map(|d| d.trace.width());
        let perm_width = air.permutation_width();
        let (main_widths, main_ptrs): (Vec<_>, Vec<_>) = partitioned_main.into_iter().unzip();
        let width = TraceWidth {
            preprocessed: preprocessed_width,
            partitioned_main: main_widths,
            after_challenge: perm_width.into_iter().collect(),
        };
        let num_challenges_to_sample = if width.after_challenge.is_empty() {
            vec![]
        } else {
            vec![NUM_PERM_CHALLENGES]
        };
        let num_exposed_values = if width.after_challenge.is_empty() {
            vec![]
        } else {
            vec![NUM_PERM_EXPOSED_VALUES]
        };
        let log_quotient_degree = get_log_quotient_degree(
            air,
            &width,
            &num_challenges_to_sample,
            num_public_values,
            &num_exposed_values,
        );
        let quotient_degree = 1 << log_quotient_degree;
        let vk = StarkVerifyingKey {
            degree,
            preprocessed_data: prep_verifier_data,
            width,
            main_graph: MatrixCommitmentGraph::new(main_ptrs),
            quotient_degree,
            num_public_values,
            num_exposed_values_after_challenge: num_exposed_values,
            num_challenges_to_sample,
        };
        let pk = StarkProvingKey {
            vk,
            preprocessed_data: prep_prover_data,
        };

        self.pk.per_air.push(pk);
    }

    /// Default way to add a single Interactive AIR.
    /// DO NOT use this if the main trace needs to be partitioned.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - Adds main trace to the default shared main trace commitment.
    #[instrument(level = "debug", skip_all)]
    pub fn get_single_preprocessed_data(
        &mut self,
        air: &dyn SymbolicRap<SC>,
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
}
