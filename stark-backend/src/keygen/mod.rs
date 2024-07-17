use itertools::Itertools;
use p3_air::BaseAir;
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::instrument;

pub mod types;

use crate::{
    air_builders::symbolic::get_log_quotient_degree,
    commit::{MatrixCommitmentPointers, SingleMatrixCommitPtr},
    interaction::AirBridge,
    prover::trace::TraceCommitter,
    rap::AnyRap,
};

use self::types::{
    create_commit_to_air_graph, MultiStarkPartialProvingKey, ProverOnlySinglePreprocessedData,
    StarkPartialProvingKey, StarkPartialVerifyingKey, TraceWidth, VerifierSinglePreprocessedData,
};

/// Constants for interactive AIRs
const NUM_PERM_CHALLENGES: usize = 2;
const NUM_PERM_EXPOSED_VALUES: usize = 1;

/// Stateful builder to create multi-stark proving and verifying keys
/// for system of multiple RAPs with multiple multi-matrix commitments
pub struct MultiStarkKeygenBuilder<'a, SC: StarkGenericConfig> {
    pub config: &'a SC,
    /// `placeholder_main_matrix_in_commit[commit_idx][mat_idx] =` matrix width, it is used to store
    /// a placeholder of a main trace matrix that must be committed during proving
    placeholder_main_matrix_in_commit: Vec<Vec<usize>>,
    partial_pk: MultiStarkPartialProvingKey<SC>,
}

impl<'a, SC: StarkGenericConfig> MultiStarkKeygenBuilder<'a, SC> {
    pub fn new(config: &'a SC) -> Self {
        Self {
            config,
            partial_pk: MultiStarkPartialProvingKey::empty(),
            placeholder_main_matrix_in_commit: vec![vec![]],
        }
    }

    /// Generates proving key, resetting the state of the builder.
    /// The verifying key can be obtained from the proving key.
    pub fn generate_partial_pk(&mut self) -> MultiStarkPartialProvingKey<SC> {
        let mut pk = std::mem::take(&mut self.partial_pk);
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

        if matches!(self.placeholder_main_matrix_in_commit.last(), Some(mats) if mats.is_empty()) {
            self.placeholder_main_matrix_in_commit.pop();
        }
        pk.num_main_trace_commitments = self.placeholder_main_matrix_in_commit.len();
        // Build commit->air graph
        let air_matrices = pk
            .per_air
            .iter()
            .map(|pk| pk.vk.main_graph.clone())
            .collect_vec();
        pk.main_commit_to_air_graph =
            create_commit_to_air_graph(&air_matrices, pk.num_main_trace_commitments);
        // reset state
        self.placeholder_main_matrix_in_commit = vec![vec![]];

        pk
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
    pub fn add_air(&mut self, air: &dyn AnyRap<SC>, num_public_values: usize) {
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
        air: &dyn AnyRap<SC>,
        num_public_values: usize,
        partitioned_main_ptrs: Vec<SingleMatrixCommitPtr>,
    ) {
        let (prep_prover_data, prep_verifier_data): (Option<_>, Option<_>) =
            self.get_single_preprocessed_data(air).unzip();
        let preprocessed_width = prep_prover_data.as_ref().map(|d| d.trace.width());
        let perm_width = <dyn AnyRap<SC> as AirBridge<Val<SC>>>::permutation_width(air);
        let main_widths = partitioned_main_ptrs
            .iter()
            .map(|ptr| self.placeholder_main_matrix_in_commit[ptr.commit_index][ptr.matrix_index])
            .collect();
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
        let vk = StarkPartialVerifyingKey {
            preprocessed_data: prep_verifier_data,
            width,
            main_graph: MatrixCommitmentPointers::new(partitioned_main_ptrs),
            quotient_degree,
            num_public_values,
            num_exposed_values_after_challenge: num_exposed_values,
            num_challenges_to_sample,
        };
        let pk = StarkPartialProvingKey {
            vk,
            preprocessed_data: prep_prover_data,
        };

        self.partial_pk.per_air.push(pk);
    }

    /// Default way to add a single Interactive AIR.
    /// DO NOT use this if the main trace needs to be partitioned.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - Adds main trace to the default shared main trace commitment.
    #[instrument(level = "debug", skip_all)]
    pub fn get_single_preprocessed_data(
        &mut self,
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
}
