//! Traits and types describing the core interfaces of the verifier sub-circuit. The verifier
//! sub-circuit verifies multiple proofs for the same child verifying key. It supports **recursive**
//! verification, where the child verifying key is equal to the verifying key of the verifier
//! circuit itself.
use std::{iter, sync::Arc};

use openvm_stark_backend::{AirRef, interaction::BusIndex};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::BasedVectorSpace;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    EF, F, SC, StarkEngineV2,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::{
        CHUNK, WIDTH,
        sponge::{FiatShamirTranscript, TranscriptHistory, TranscriptLog},
    },
    proof::{Proof, TraceVData},
    prover::{
        AirProvingContextV2, ColMajorMatrix, CommittedTraceDataV2, CpuBackendV2, ProverBackendV2,
    },
};

use crate::{
    batch_constraint::{BatchConstraintModule, LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX},
    bus::{
        AirShapeBus, BatchConstraintModuleBus, CachedCommitBus, ColumnClaimsBus, CommitmentsBus,
        ConstraintSumcheckRandomnessBus, DagCommitBus, EqNegBaseRandBus, EqNegResultBus,
        ExpressionClaimNMaxBus, FractionFolderInputBus, GkrModuleBus, HyperdimBus,
        LiftedHeightsBus, MerkleVerifyBus, Poseidon2CompressBus, Poseidon2PermuteBus,
        PublicValuesBus, SelUniBus, StackingIndicesBus, StackingModuleBus, TranscriptBus,
        WhirModuleBus, WhirOpeningPointBus, XiRandomnessBus,
    },
    gkr::GkrModule,
    primitives::{
        bus::{ExpBitsLenBus, PowerCheckerBus, RangeCheckerBus},
        exp_bits_len::{ExpBitsLenAir, ExpBitsLenTraceGenerator},
        pow::{PowerCheckerAir, PowerCheckerTraceGenerator},
    },
    proof_shape::ProofShapeModule,
    stacking::StackingModule,
    transcript::TranscriptModule,
    utils::poseidon2_hash_slice_with_states,
    whir::{WhirModule, folding::FoldRecord},
};

mod dummy;
pub(crate) mod frame;

const BATCH_CONSTRAINT_MOD_IDX: usize = 0;
const POW_CHECKER_HEIGHT: usize = 32;

// Trait to make tracegen functions generic on ProverBackendV2
pub trait VerifierTraceGen<PB: ProverBackendV2> {
    fn new(
        child_mvk: Arc<MultiStarkVerifyingKeyV2>,
        continuations_enabled: bool,
        has_cached: bool,
    ) -> Self;

    fn commit_child_vk<E: StarkEngineV2<SC = SC, PB = PB>>(
        &self,
        engine: &E,
        child_vk: &MultiStarkVerifyingKeyV2,
    ) -> CommittedTraceDataV2<PB>;

    /// The generic `TS` allows using different transcript implementations for debugging purposes.
    /// The default type to use is `DuplexSpongeRecorder`.
    fn generate_proving_ctxs<TS: FiatShamirTranscript + TranscriptHistory + Default>(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        child_vk_pcs_data: CommittedTraceDataV2<PB>,
        proofs: &[Proof],
        external_poseidon2_compress_inputs: &Vec<[PB::Val; WIDTH]>,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContextV2<PB>>>;

    fn generate_proving_ctxs_base<TS: FiatShamirTranscript + TranscriptHistory + Default>(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        child_vk_pcs_data: CommittedTraceDataV2<PB>,
        proofs: &[Proof],
    ) -> Vec<AirProvingContextV2<PB>> {
        self.generate_proving_ctxs::<TS>(child_vk, child_vk_pcs_data, proofs, &vec![], None)
            .unwrap()
    }
}

// Trait to help make AIR generation generic
pub trait AggregationSubCircuit {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>>;
    fn bus_inventory(&self) -> &BusInventory;
    fn next_bus_idx(&self) -> BusIndex;
    fn max_num_proofs(&self) -> usize;
}

// TODO[jpw]: make this generic in <SC: StarkGenericConfig>
pub trait AirModule {
    fn num_airs(&self) -> usize;
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>>;
}

/// Trait defining the types for the global input shared across modules for trace generation. These
/// types are specialized per hardware backend.
pub trait GlobalTraceGenCtx {
    /// Verifying key of the child proof to be verified. This is a multi-trace verifying key.
    type ChildVerifyingKey;
    /// Type for a collection of proofs.
    type MultiProof: ?Sized;
    /// Preflight records corresponding to an instance of `MultiProof`.
    // NOTE[jpw]: we can add lifetimes if necessary
    type PreflightRecords: ?Sized;
}

/// Trait for generating the trace matrices, on device, for a given AIR module.
/// The module has a view of all proofs being verified as well as the global preflight records from
/// each proof.
///
/// This function should be expected to be called in parallel, one logical thread per module.
pub trait TraceGenModule<GC: GlobalTraceGenCtx, PB: ProverBackendV2>: Send + Sync {
    type ModuleSpecificCtx;

    fn generate_proving_ctxs(
        &self,
        child_vk: &GC::ChildVerifyingKey,
        proofs: &GC::MultiProof,
        preflights: &GC::PreflightRecords,
        ctx: &Self::ModuleSpecificCtx,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContextV2<PB>>>;
}

pub struct GlobalCtxCpu;

impl GlobalTraceGenCtx for GlobalCtxCpu {
    type ChildVerifyingKey = MultiStarkVerifyingKeyV2;
    type MultiProof = [Proof];
    type PreflightRecords = [Preflight];
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BusIndexManager {
    /// All existing buses use indices in [0, bus_idx_max)
    bus_idx_max: BusIndex,
}

impl BusIndexManager {
    pub fn new() -> Self {
        Self { bus_idx_max: 0 }
    }

    pub fn new_bus_idx(&mut self) -> BusIndex {
        let idx = self.bus_idx_max;
        self.bus_idx_max = self.bus_idx_max.checked_add(1).unwrap();
        idx
    }
}

#[derive(Clone, Debug)]
pub struct BusInventory {
    // Control flow buses
    pub transcript_bus: TranscriptBus,
    pub poseidon2_permute_bus: Poseidon2PermuteBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub merkle_verify_bus: MerkleVerifyBus,
    pub gkr_module_bus: GkrModuleBus,
    pub bc_module_bus: BatchConstraintModuleBus,
    pub stacking_module_bus: StackingModuleBus,
    pub whir_module_bus: WhirModuleBus,

    // Data buses
    pub air_shape_bus: AirShapeBus,
    pub hyperdim_bus: HyperdimBus,
    pub lifted_heights_bus: LiftedHeightsBus,
    pub stacking_indices_bus: StackingIndicesBus,
    pub commitments_bus: CommitmentsBus,
    pub public_values_bus: PublicValuesBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub range_checker_bus: RangeCheckerBus,
    pub power_checker_bus: PowerCheckerBus,
    pub expression_claim_n_max_bus: ExpressionClaimNMaxBus,
    pub fraction_folder_input_bus: FractionFolderInputBus,

    // Randomness buses
    pub xi_randomness_bus: XiRandomnessBus,
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,

    // Compute buses
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub sel_uni_bus: SelUniBus,
    pub eq_neg_result_bus: EqNegResultBus,
    pub eq_neg_base_rand_bus: EqNegBaseRandBus,

    // Continuations buses
    pub cached_commit_bus: CachedCommitBus,
    pub dag_commit_bus: DagCommitBus,
}

/// The records from global recursion preflight on CPU for verifying a single proof.
#[derive(Clone, Debug, Default)]
pub struct Preflight {
    /// The concatenated sequence of observes/samples. Not available during preflight; populated
    /// after.
    pub transcript: TranscriptLog,
    // TODO[jpw]: flatten and remove these preflight types if they are mostly trivial
    pub proof_shape: ProofShapePreflight,
    pub gkr: GkrPreflight,
    pub batch_constraint: BatchConstraintPreflight,
    pub stacking: StackingPreflight,
    pub whir: WhirPreflight,
    pub poseidon2_perm_inputs: Vec<[F; WIDTH]>,
    pub poseidon2_compress_inputs: Vec<[F; WIDTH]>,
    pub initial_row_states: Vec<Vec<Vec<Vec<[F; WIDTH]>>>>,
    /// Indexed by [round][query][coset]. Stores post-permutation state.
    pub codeword_states: Vec<Vec<Vec<[F; WIDTH]>>>,
}

#[derive(Clone, Debug, Default)]
pub struct ProofShapePreflight {
    pub sorted_trace_vdata: Vec<(usize, TraceVData)>,
    pub starting_tidx: Vec<usize>,
    pub pvs_tidx: Vec<usize>,
    pub post_tidx: usize,
    pub n_max: usize,
    pub n_logup: usize,
    pub l_skip: usize,
}

impl ProofShapePreflight {
    pub fn n_global(&self) -> usize {
        self.n_max.max(self.n_logup)
    }
}

#[derive(Clone, Debug, Default)]
pub struct GkrPreflight {
    pub post_tidx: usize,
    pub xi: Vec<(usize, EF)>,
}

#[derive(Clone, Debug, Default)]
pub struct BatchConstraintPreflight {
    pub lambda_tidx: usize,
    pub tidx_before_univariate: usize,
    pub tidx_before_multilinear: usize,
    pub tidx_before_column_openings: usize,
    pub post_tidx: usize,
    pub xi: Vec<EF>,
    pub sumcheck_rnd: Vec<EF>,
    pub eq_ns: Vec<EF>,
    pub eq_sharp_ns: Vec<EF>,
    pub eq_ns_frontloaded: Vec<EF>,
    pub eq_sharp_ns_frontloaded: Vec<EF>,
}

#[derive(Clone, Debug, Default)]
pub struct StackingPreflight {
    pub intermediate_tidx: [usize; 3],
    pub post_tidx: usize,
    pub univariate_poly_rand_eval: EF,
    pub stacking_batching_challenge: EF,
    pub lambda: EF,
    pub sumcheck_rnd: Vec<EF>,
}

#[derive(Clone, Debug, Default)]
pub struct WhirPreflight {
    pub num_queries_per_round: Vec<usize>,
    pub query_offsets: Vec<usize>,
    pub alphas: Vec<EF>,
    pub z0s: Vec<EF>,
    pub zj_roots: Vec<Vec<F>>,
    pub zjs: Vec<Vec<F>>,
    pub yjs: Vec<Vec<EF>>,
    pub gammas: Vec<EF>,
    pub folding_pow_samples: Vec<F>,
    pub query_pow_samples: Vec<F>,
    pub queries: Vec<F>,
    pub tidx_per_round: Vec<usize>,
    pub query_tidx_per_round: Vec<usize>,
    pub initial_claim_per_round: Vec<EF>,
    pub post_sumcheck_claims: Vec<EF>,
    pub pre_query_claims: Vec<EF>,
    pub eq_partials: Vec<EF>,
    pub fold_records: Vec<FoldRecord>,
    pub final_poly_at_u: EF,
}

impl BusInventory {
    pub(crate) fn new(b: &mut BusIndexManager) -> Self {
        Self {
            transcript_bus: TranscriptBus::new(b.new_bus_idx()),
            poseidon2_permute_bus: Poseidon2PermuteBus::new(b.new_bus_idx()),
            poseidon2_compress_bus: Poseidon2CompressBus::new(b.new_bus_idx()),
            merkle_verify_bus: MerkleVerifyBus::new(b.new_bus_idx()),

            // Control flow buses
            gkr_module_bus: GkrModuleBus::new(b.new_bus_idx()),
            bc_module_bus: BatchConstraintModuleBus::new(b.new_bus_idx()),
            stacking_module_bus: StackingModuleBus::new(b.new_bus_idx()),
            whir_module_bus: WhirModuleBus::new(b.new_bus_idx()),

            // Data buses
            air_shape_bus: AirShapeBus::new(b.new_bus_idx()),
            hyperdim_bus: HyperdimBus::new(b.new_bus_idx()),
            lifted_heights_bus: LiftedHeightsBus::new(b.new_bus_idx()),
            stacking_indices_bus: StackingIndicesBus::new(b.new_bus_idx()),
            commitments_bus: CommitmentsBus::new(b.new_bus_idx()),
            public_values_bus: PublicValuesBus::new(b.new_bus_idx()),
            sel_uni_bus: SelUniBus::new(b.new_bus_idx()),
            range_checker_bus: RangeCheckerBus::new(b.new_bus_idx()),
            power_checker_bus: PowerCheckerBus::new(b.new_bus_idx()),
            expression_claim_n_max_bus: ExpressionClaimNMaxBus::new(b.new_bus_idx()),
            fraction_folder_input_bus: FractionFolderInputBus::new(b.new_bus_idx()),

            // Randomness buses
            xi_randomness_bus: XiRandomnessBus::new(b.new_bus_idx()),
            constraint_randomness_bus: ConstraintSumcheckRandomnessBus::new(b.new_bus_idx()),
            whir_opening_point_bus: WhirOpeningPointBus::new(b.new_bus_idx()),

            // Claims buses
            column_claims_bus: ColumnClaimsBus::new(b.new_bus_idx()),

            exp_bits_len_bus: ExpBitsLenBus::new(b.new_bus_idx()),
            eq_neg_base_rand_bus: EqNegBaseRandBus::new(b.new_bus_idx()),
            eq_neg_result_bus: EqNegResultBus::new(b.new_bus_idx()),

            // Continuation buses
            cached_commit_bus: CachedCommitBus::new(b.new_bus_idx()),
            dag_commit_bus: DagCommitBus::new(b.new_bus_idx()),
        }
    }
}

/// A pre-state/post-state pair for a single Poseidon permutation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PoseidonStatePair {
    pub pre_state: [F; WIDTH],
    pub post_state: [F; WIDTH],
}

struct MerklePrecomputation {
    poseidon2_perm_inputs: Vec<[F; WIDTH]>,
    poseidon2_compress_inputs: Vec<[F; WIDTH]>,
    initial_row_states: Vec<Vec<Vec<Vec<[F; WIDTH]>>>>,
    codeword_states: Vec<Vec<Vec<[F; WIDTH]>>>,
}

#[derive(Clone, Copy, strum_macros::Display)]
enum TraceModuleRef<'a> {
    Transcript(&'a TranscriptModule),
    ProofShape(&'a ProofShapeModule),
    Gkr(&'a GkrModule),
    BatchConstraint(&'a BatchConstraintModule),
    Stacking(&'a StackingModule),
    Whir(&'a WhirModule),
}

impl<'a> TraceModuleRef<'a> {
    #[tracing::instrument(
        name = "wrapper.run_preflight",
        level = "trace",
        skip_all,
        fields(air_module = %self)
    )]
    fn run_preflight<TS>(
        self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight,
        sponge: &mut TS,
    ) where
        TS: FiatShamirTranscript + TranscriptHistory,
    {
        match self {
            TraceModuleRef::ProofShape(module) => {
                module.run_preflight(child_vk, proof, preflight, sponge)
            }
            TraceModuleRef::Gkr(module) => module.run_preflight(proof, preflight, sponge),
            TraceModuleRef::BatchConstraint(module) => {
                module.run_preflight(child_vk, proof, preflight, sponge)
            }
            TraceModuleRef::Stacking(module) => module.run_preflight(proof, preflight, sponge),
            TraceModuleRef::Whir(module) => module.run_preflight(proof, preflight, sponge),
            _ => panic!("TraceModuleRef::run_preflight called with invalid module"),
        }
    }

    #[tracing::instrument(
        name = "wrapper.generate_proving_ctxs",
        level = "trace",
        skip_all,
        fields(air_module = %self)
    )]
    fn generate_cpu_ctxs(
        self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
        exp_bits_len_gen: &ExpBitsLenTraceGenerator,
        external_poseidon2_compress_inputs: &Vec<[F; WIDTH]>,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContextV2<CpuBackendV2>>> {
        match self {
            TraceModuleRef::Transcript(module) => module.generate_proving_ctxs(
                child_vk,
                proofs,
                preflights,
                external_poseidon2_compress_inputs,
                required_heights,
            ),
            TraceModuleRef::ProofShape(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, &(), required_heights)
            }
            TraceModuleRef::Gkr(module) => module.generate_proving_ctxs(
                child_vk,
                proofs,
                preflights,
                exp_bits_len_gen,
                required_heights,
            ),
            TraceModuleRef::BatchConstraint(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, &(), required_heights)
            }
            TraceModuleRef::Stacking(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, &(), required_heights)
            }
            TraceModuleRef::Whir(module) => module.generate_proving_ctxs(
                child_vk,
                proofs,
                preflights,
                exp_bits_len_gen,
                required_heights,
            ),
        }
    }
}

/// The recursive verifier sub-circuit consists of multiple chips, grouped into **modules**.
///
/// This struct is stateful.
pub struct VerifierSubCircuit<const MAX_NUM_PROOFS: usize> {
    bus_inventory: BusInventory,
    bus_idx_manager: BusIndexManager,

    transcript: TranscriptModule,
    proof_shape: ProofShapeModule,
    gkr: GkrModule,
    batch_constraint: BatchConstraintModule,
    stacking: StackingModule,
    whir: WhirModule,

    power_checker_air: Arc<PowerCheckerAir<2, POW_CHECKER_HEIGHT>>,
    power_checker_trace: Arc<PowerCheckerTraceGenerator<2, POW_CHECKER_HEIGHT>>,
}

impl<const MAX_NUM_PROOFS: usize> VerifierSubCircuit<MAX_NUM_PROOFS> {
    pub fn new(child_mvk: Arc<MultiStarkVerifyingKeyV2>) -> Self {
        Self::new_with_options(child_mvk, false, true)
    }

    pub fn new_with_options(
        child_mvk: Arc<MultiStarkVerifyingKeyV2>,
        continuations_enabled: bool,
        has_cached: bool,
    ) -> Self {
        let mut bus_idx_manager = BusIndexManager::new();
        let bus_inventory = BusInventory::new(&mut bus_idx_manager);
        let power_checker_trace =
            Arc::new(PowerCheckerTraceGenerator::<2, POW_CHECKER_HEIGHT>::default());
        let power_checker_air = Arc::new(PowerCheckerAir::<2, POW_CHECKER_HEIGHT> {
            pow_bus: bus_inventory.power_checker_bus,
            range_bus: bus_inventory.range_checker_bus,
        });

        let transcript =
            TranscriptModule::new(bus_inventory.clone(), child_mvk.inner.params.clone());
        let child_mvk_frame = child_mvk.as_ref().into();
        let proof_shape = ProofShapeModule::new(
            &child_mvk_frame,
            &mut bus_idx_manager,
            bus_inventory.clone(),
            power_checker_trace.clone(),
            continuations_enabled,
        );
        let gkr = GkrModule::new(&child_mvk, &mut bus_idx_manager, bus_inventory.clone());
        let batch_constraint = BatchConstraintModule::new(
            &child_mvk,
            &mut bus_idx_manager,
            bus_inventory.clone(),
            MAX_NUM_PROOFS,
            power_checker_trace.clone(),
            has_cached,
        );
        let stacking = StackingModule::new(&child_mvk, &mut bus_idx_manager, bus_inventory.clone());
        let whir = WhirModule::new(&child_mvk, &mut bus_idx_manager, bus_inventory.clone());

        VerifierSubCircuit {
            bus_inventory,
            bus_idx_manager,
            transcript,
            proof_shape,
            gkr,
            batch_constraint,
            stacking,
            whir,
            power_checker_air,
            power_checker_trace,
        }
    }

    /// Runs preflight for a single proof.
    #[tracing::instrument(name = "execute_preflight", skip_all)]
    pub fn run_preflight<TS>(
        &self,
        mut sponge: TS,
        child_vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
    ) -> Preflight
    where
        TS: FiatShamirTranscript + TranscriptHistory,
    {
        let mut preflight = Preflight::default();

        // NOTE: it is not required that we group preflight into modules
        let preflight_modules = [
            TraceModuleRef::ProofShape(&self.proof_shape),
            TraceModuleRef::Gkr(&self.gkr),
            TraceModuleRef::BatchConstraint(&self.batch_constraint),
            TraceModuleRef::Stacking(&self.stacking),
            TraceModuleRef::Whir(&self.whir),
        ];
        for module in &preflight_modules {
            module.run_preflight(child_vk, proof, &mut preflight, &mut sponge);
        }
        preflight.transcript = sponge.into_log();

        #[cfg(feature = "cuda")]
        let merkle_precomputation = Self::compute_merkle_precomputation_cuda(proof);
        #[cfg(not(feature = "cuda"))]
        let merkle_precomputation = Self::compute_merkle_precomputation(proof);

        preflight.poseidon2_perm_inputs = merkle_precomputation.poseidon2_perm_inputs;
        preflight.poseidon2_compress_inputs = merkle_precomputation.poseidon2_compress_inputs;
        preflight.initial_row_states = merkle_precomputation.initial_row_states;
        preflight.codeword_states = merkle_precomputation.codeword_states;

        preflight
    }

    #[cfg_attr(feature = "cuda", allow(dead_code))]
    #[tracing::instrument(name = "compute_merkle_precomputation", level = "info", skip_all)]
    fn compute_merkle_precomputation(proof: &Proof) -> MerklePrecomputation {
        let initial_chunks: usize = proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .flat_map(|c| c.iter().flat_map(|q| q.iter()))
            .map(|row| row.len().div_ceil(CHUNK))
            .sum();
        let codeword_chunks: usize = proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .map(|r| r.iter().map(|q| q.len()).sum::<usize>())
            .sum();

        // InitialOpenedValuesAir (initial_row_states) does Poseidon2 *permute* lookups per chunk.
        // NonInitialOpenedValuesAir (codeword_states) does Poseidon2 *compress* lookups per value.
        let mut poseidon2_perm_inputs = Vec::with_capacity(initial_chunks);
        let mut poseidon2_compress_inputs = Vec::with_capacity(codeword_chunks);

        let initial_row_states: Vec<Vec<Vec<Vec<[F; WIDTH]>>>> = proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .map(|opened_rows_per_commit| {
                opened_rows_per_commit
                    .iter()
                    .map(|opened_rows_per_query| {
                        opened_rows_per_query
                            .iter()
                            .map(|opened_row| {
                                let (_leaf_hash, pre_states, post_states) =
                                    poseidon2_hash_slice_with_states(opened_row);
                                poseidon2_perm_inputs.extend(pre_states);
                                post_states
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let codeword_states = proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .map(|round_values| {
                round_values
                    .iter()
                    .map(|opened_values_per_query| {
                        opened_values_per_query
                            .iter()
                            .map(|opened_value| {
                                let (_leaf_hash, pre_states, post_states) =
                                    poseidon2_hash_slice_with_states(
                                        opened_value.as_basis_coefficients_slice(),
                                    );
                                // This is not quite a compression, but the AIR will constrain that
                                // the padded pre_state gets
                                // compressed into _leaf_hash via Poseidon2CompressBus.
                                poseidon2_compress_inputs.extend(pre_states);
                                debug_assert_eq!(post_states.len(), 1);
                                post_states.into_iter().next().unwrap()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        MerklePrecomputation {
            poseidon2_perm_inputs,
            poseidon2_compress_inputs,
            initial_row_states,
            codeword_states,
        }
    }

    #[cfg(feature = "cuda")]
    #[tracing::instrument(name = "compute_merkle_precomputation_cuda", level = "info", skip_all)]
    fn compute_merkle_precomputation_cuda(proof: &Proof) -> MerklePrecomputation {
        use openvm_cuda_common::{
            copy::{MemCopyD2H, MemCopyH2D},
            d_buffer::DeviceBuffer,
        };

        use crate::cuda::abi::{VectorDescriptor, merkle_precomputation_hash_vectors};

        let num_chunks = |len: usize| len.div_ceil(CHUNK);

        let mut num_vectors = 0usize;
        let mut total_data_len = 0usize;
        let mut total_chunks = 0usize;

        for row in proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .flat_map(|per_commit| per_commit.iter().flat_map(|per_query| per_query.iter()))
        {
            num_vectors += 1;
            total_data_len += row.len();
            total_chunks += num_chunks(row.len());
        }
        let num_perm_chunks = total_chunks;

        for opened_value in proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .flat_map(|per_round| per_round.iter().flat_map(|per_query| per_query.iter()))
        {
            let len = <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(opened_value).len();
            num_vectors += 1;
            total_data_len += len;
            total_chunks += num_chunks(len);
        }

        let mut flat_data = Vec::with_capacity(total_data_len);
        let mut descriptors = Vec::with_capacity(num_vectors);
        let mut output_offset_chunks = 0usize;

        let mut push_vector = |data: &[F]| {
            let len = data.len();
            let chunks = num_chunks(len);
            descriptors.push(VectorDescriptor {
                data_offset: flat_data.len(),
                len,
                output_offset: output_offset_chunks,
            });
            output_offset_chunks += chunks;
            flat_data.extend_from_slice(data);
        };

        for row in proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .flat_map(|per_commit| per_commit.iter().flat_map(|per_query| per_query.iter()))
        {
            push_vector(row);
        }
        for opened_value in proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .flat_map(|per_round| per_round.iter().flat_map(|per_query| per_query.iter()))
        {
            push_vector(opened_value.as_basis_coefficients_slice());
        }

        debug_assert_eq!(descriptors.len(), num_vectors);
        debug_assert_eq!(flat_data.len(), total_data_len);
        debug_assert_eq!(output_offset_chunks, total_chunks);

        // Upload to GPU and run kernel
        let d_data = flat_data.to_device().expect("failed to upload data");
        let d_descriptors = descriptors
            .to_device()
            .expect("failed to upload descriptors");
        let d_pre_states = DeviceBuffer::<F>::with_capacity(total_chunks * WIDTH);
        let d_post_states = DeviceBuffer::<F>::with_capacity(total_chunks * WIDTH);

        unsafe {
            merkle_precomputation_hash_vectors(
                &d_data,
                &d_descriptors,
                num_vectors,
                &d_pre_states,
                &d_post_states,
            )
            .expect("hash_vectors kernel failed");
        }

        // Download results
        let pre_states_flat = d_pre_states
            .to_host()
            .expect("failed to download pre_states");
        let post_states_flat = d_post_states
            .to_host()
            .expect("failed to download post_states");
        debug_assert_eq!(pre_states_flat.len(), total_chunks * WIDTH);
        debug_assert_eq!(post_states_flat.len(), total_chunks * WIDTH);

        // Split pre_states into poseidon permute and compress inputs
        let (perm_flat, compress_flat) = pre_states_flat.split_at(num_perm_chunks * WIDTH);
        let poseidon2_perm_inputs: Vec<[F; WIDTH]> = perm_flat
            .chunks_exact(WIDTH)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();
        let poseidon2_compress_inputs: Vec<[F; WIDTH]> = compress_flat
            .chunks_exact(WIDTH)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();

        let mut post_iter = post_states_flat.chunks_exact(WIDTH);

        let initial_row_states: Vec<Vec<Vec<Vec<[F; WIDTH]>>>> = proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .map(|per_commit| {
                per_commit
                    .iter()
                    .map(|per_query| {
                        per_query
                            .iter()
                            .map(|row| {
                                (0..num_chunks(row.len()))
                                    .map(|_| post_iter.next().unwrap().try_into().unwrap())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let codeword_states: Vec<Vec<Vec<[F; WIDTH]>>> = proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .map(|per_round| {
                per_round
                    .iter()
                    .map(|per_query| {
                        per_query
                            .iter()
                            .map(|opened_value| {
                                debug_assert_eq!(
                                    num_chunks(
                                        <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(
                                            opened_value
                                        )
                                        .len()
                                    ),
                                    1
                                );
                                post_iter.next().unwrap().try_into().unwrap()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        debug_assert_eq!(post_iter.len(), 0);

        MerklePrecomputation {
            poseidon2_perm_inputs,
            poseidon2_compress_inputs,
            initial_row_states,
            codeword_states,
        }
    }

    /// Utility function to split a slice of required trace heights per-module. Fails
    /// an assert if the slice length doesn't match the number of AIRs.
    #[allow(clippy::type_complexity)]
    fn split_required_heights<'a>(
        &self,
        required_heights: Option<&'a [usize]>,
    ) -> (Vec<Option<&'a [usize]>>, Option<usize>, Option<usize>) {
        let bc_n = self.batch_constraint.num_airs();
        let t_n = self.transcript.num_airs();
        let ps_n = self.proof_shape.num_airs();
        let gkr_n = self.gkr.num_airs();
        let st_n = self.stacking.num_airs();
        let w_n = self.whir.num_airs();
        let module_air_counts = [bc_n, t_n, ps_n, gkr_n, st_n, w_n];

        let Some(heights) = required_heights else {
            return (vec![None; module_air_counts.len()], None, None);
        };

        let total_module_airs: usize = module_air_counts.iter().sum();
        let total = total_module_airs + 2; // PowerChecker + ExpBitsLen
        assert_eq!(heights.len(), total);

        let mut offset = 0usize;
        let mut per_module = Vec::with_capacity(module_air_counts.len());
        for n in module_air_counts {
            per_module.push(Some(&heights[offset..offset + n]));
            offset += n;
        }
        debug_assert_eq!(heights.len() - offset, 2);

        (per_module, Some(heights[offset]), Some(heights[offset + 1]))
    }
}

impl<const MAX_NUM_PROOFS: usize> AggregationSubCircuit for VerifierSubCircuit<MAX_NUM_PROOFS> {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let exp_bits_len_air = ExpBitsLenAir::new(self.bus_inventory.exp_bits_len_bus);

        // WARNING: SymbolicExpressionAir MUST be the first AIR in verifier circuit
        iter::empty()
            .chain(self.batch_constraint.airs())
            .chain(self.transcript.airs())
            .chain(self.proof_shape.airs())
            .chain(self.gkr.airs())
            .chain(self.stacking.airs())
            .chain(self.whir.airs())
            .chain([
                self.power_checker_air.clone() as AirRef<_>,
                Arc::new(exp_bits_len_air) as AirRef<_>,
            ])
            .collect()
    }

    fn bus_inventory(&self) -> &BusInventory {
        &self.bus_inventory
    }

    fn next_bus_idx(&self) -> BusIndex {
        self.bus_idx_manager.bus_idx_max
    }

    fn max_num_proofs(&self) -> usize {
        MAX_NUM_PROOFS
    }
}

impl<const MAX_NUM_PROOFS: usize> VerifierTraceGen<CpuBackendV2>
    for VerifierSubCircuit<MAX_NUM_PROOFS>
{
    fn new(
        child_mvk: Arc<MultiStarkVerifyingKeyV2>,
        continuations_enabled: bool,
        has_cached: bool,
    ) -> Self {
        Self::new_with_options(child_mvk, continuations_enabled, has_cached)
    }

    fn commit_child_vk<E: StarkEngineV2<SC = SC, PB = CpuBackendV2>>(
        &self,
        engine: &E,
        child_vk: &MultiStarkVerifyingKeyV2,
    ) -> CommittedTraceDataV2<CpuBackendV2> {
        self.batch_constraint.commit_child_vk(engine, child_vk)
    }

    #[tracing::instrument(name = "subcircuit_generate_proving_ctxs", skip_all)]
    fn generate_proving_ctxs<TS: FiatShamirTranscript + TranscriptHistory + Default>(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        child_vk_pcs_data: CommittedTraceDataV2<CpuBackendV2>,
        proofs: &[Proof],
        external_poseidon2_compress_inputs: &Vec<[F; WIDTH]>,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContextV2<CpuBackendV2>>> {
        debug_assert!(proofs.len() <= MAX_NUM_PROOFS);
        // Use std::thread::scope for preflight parallelism. With only 3-4 proofs max, this avoids
        // Rayon's thread pool overhead (wake-up, work stealing, synchronization) while still
        // getting parallelism with minimal overhead.
        let span = tracing::Span::current();
        let preflights = std::thread::scope(|s| {
            let handles: Vec<_> = proofs
                .iter()
                .map(|proof| {
                    s.spawn(|| {
                        let _guard = span.enter();
                        let sponge = TS::default();
                        self.run_preflight(sponge, child_vk, proof)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });

        self.power_checker_trace.reset();
        let exp_bits_len_gen = ExpBitsLenTraceGenerator::default();

        let (module_required, power_checker_required, exp_bits_len_required) =
            self.split_required_heights(required_heights);

        // WARNING: SymbolicExpressionAir MUST be the first AIR in verifier circuit
        let modules = vec![
            TraceModuleRef::BatchConstraint(&self.batch_constraint),
            TraceModuleRef::Transcript(&self.transcript),
            TraceModuleRef::ProofShape(&self.proof_shape),
            TraceModuleRef::Gkr(&self.gkr),
            TraceModuleRef::Stacking(&self.stacking),
            TraceModuleRef::Whir(&self.whir),
        ];
        let span = tracing::Span::current();
        let ctxs_by_module = modules
            .into_par_iter()
            .zip(module_required)
            .map(|(module, required_heights)| {
                let _guard = span.enter();
                module.generate_cpu_ctxs(
                    child_vk,
                    proofs,
                    &preflights,
                    &exp_bits_len_gen,
                    external_poseidon2_compress_inputs,
                    required_heights,
                )
            })
            .collect::<Vec<_>>();

        let mut ctxs_by_module: Vec<Vec<AirProvingContextV2<CpuBackendV2>>> =
            ctxs_by_module.into_iter().collect::<Option<Vec<_>>>()?;
        if self.batch_constraint.has_cached {
            ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX]
                .cached_mains = vec![child_vk_pcs_data];
        }
        let mut ctx_per_trace = ctxs_by_module.into_iter().flatten().collect::<Vec<_>>();
        if power_checker_required.is_some_and(|h| h != POW_CHECKER_HEIGHT) {
            return None;
        }
        // Caution: this must be done after GKR and WHIR tracegen
        tracing::trace_span!("wrapper.generate_proving_ctxs", air_module = "Primitives",).in_scope(
            || {
                tracing::trace_span!("wrapper.generate_trace", air = "PowerChecker").in_scope(
                    || {
                        ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
                            ColMajorMatrix::from_row_major(
                                &self.power_checker_trace.generate_trace_row_major(),
                            ),
                        ));
                    },
                );
            },
        );
        let exp_bits_trace_rm = tracing::trace_span!("wrapper.generate_trace", air = "ExpBitsLen")
            .in_scope(|| exp_bits_len_gen.generate_trace_row_major(exp_bits_len_required))?;
        ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
            ColMajorMatrix::from_row_major(&exp_bits_trace_rm),
        ));
        Some(ctx_per_trace)
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_tracegen {
    use std::iter::zip;

    use cuda_backend_v2::{GpuBackendV2, transport_matrix_h2d_col_major};

    use super::*;
    use crate::cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu};

    impl<'a> TraceModuleRef<'a> {
        #[tracing::instrument(
            name = "wrapper.generate_proving_ctxs",
            level = "trace",
            skip_all,
            fields(air_module = %self)
        )]
        fn generate_gpu_ctxs(
            self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            exp_bits_len_gen: &ExpBitsLenTraceGenerator,
            external_poseidon2_compress_inputs: &Vec<[F; WIDTH]>,
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContextV2<GpuBackendV2>>> {
            match self {
                TraceModuleRef::Transcript(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    external_poseidon2_compress_inputs,
                    required_heights,
                ),
                TraceModuleRef::ProofShape(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    &(),
                    required_heights,
                ),
                TraceModuleRef::Gkr(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    exp_bits_len_gen,
                    required_heights,
                ),
                TraceModuleRef::BatchConstraint(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    &(),
                    required_heights,
                ),
                TraceModuleRef::Stacking(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    &(),
                    required_heights,
                ),
                TraceModuleRef::Whir(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    exp_bits_len_gen,
                    required_heights,
                ),
            }
        }
    }

    impl<const MAX_NUM_PROOFS: usize> VerifierTraceGen<GpuBackendV2>
        for VerifierSubCircuit<MAX_NUM_PROOFS>
    {
        fn new(
            child_mvk: Arc<MultiStarkVerifyingKeyV2>,
            continuations_enabled: bool,
            has_cached: bool,
        ) -> Self {
            Self::new_with_options(child_mvk, continuations_enabled, has_cached)
        }

        fn commit_child_vk<E: StarkEngineV2<SC = SC, PB = GpuBackendV2>>(
            &self,
            engine: &E,
            child_vk: &MultiStarkVerifyingKeyV2,
        ) -> CommittedTraceDataV2<GpuBackendV2> {
            self.batch_constraint.commit_child_vk_gpu(engine, child_vk)
        }

        #[tracing::instrument(name = "subcircuit_generate_proving_ctxs", skip_all)]
        fn generate_proving_ctxs<TS: FiatShamirTranscript + TranscriptHistory + Default>(
            &self,
            child_vk: &MultiStarkVerifyingKeyV2,
            child_vk_pcs_data: CommittedTraceDataV2<GpuBackendV2>,
            proofs: &[Proof],
            external_poseidon2_compress_inputs: &Vec<[F; WIDTH]>,
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContextV2<GpuBackendV2>>> {
            debug_assert!(proofs.len() <= MAX_NUM_PROOFS);
            let child_vk_gpu = VerifyingKeyGpu::new(child_vk);
            let proofs_gpu = proofs
                .iter()
                .map(|proof_cpu| ProofGpu::new(child_vk, proof_cpu))
                .collect::<Vec<_>>();
            // Use std::thread::scope for preflight parallelism. With only 3-4 proofs max, this
            // avoids Rayon's thread pool overhead while still getting parallelism.
            let span = tracing::Span::current();
            let preflights_cpu = std::thread::scope(|s| {
                let handles: Vec<_> = proofs
                    .iter()
                    .map(|proof| {
                        s.spawn(|| {
                            let _guard = span.enter();
                            let sponge = TS::default();
                            self.run_preflight(sponge, child_vk, proof)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });

            let exp_bits_len_gen = ExpBitsLenTraceGenerator::default();
            self.power_checker_trace.reset();

            let (module_required, power_checker_required, exp_bits_len_required) =
                self.split_required_heights(required_heights);

            // NOTE: avoid par_iter for now so H2D transfer all happens on same stream to avoid sync
            // issues
            let preflights_gpu = zip(proofs, preflights_cpu)
                .map(|(proof, preflight_cpu)| PreflightGpu::new(child_vk, proof, &preflight_cpu))
                .collect::<Vec<_>>();
            let modules = vec![
                TraceModuleRef::BatchConstraint(&self.batch_constraint),
                TraceModuleRef::Transcript(&self.transcript),
                TraceModuleRef::ProofShape(&self.proof_shape),
                TraceModuleRef::Gkr(&self.gkr),
                TraceModuleRef::Stacking(&self.stacking),
                TraceModuleRef::Whir(&self.whir),
            ];
            // PERF[jpw]: we avoid par_iter so that kernel launches occur on the same stream.
            // This can be parallelized to separate streams for more CUDA stream parallelism, but it
            // will require recording events so streams properly sync for cudaMemcpyAsync and kernel
            // launches
            let mut ctxs_by_module = Vec::with_capacity(modules.len());
            for (module, required_heights) in modules.into_iter().zip(module_required) {
                ctxs_by_module.push(module.generate_gpu_ctxs(
                    &child_vk_gpu,
                    &proofs_gpu,
                    &preflights_gpu,
                    &exp_bits_len_gen,
                    external_poseidon2_compress_inputs,
                    required_heights,
                )?);
            }
            if self.batch_constraint.has_cached {
                ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX]
                    .cached_mains = vec![child_vk_pcs_data];
            }
            let mut ctx_per_trace = ctxs_by_module.into_iter().flatten().collect::<Vec<_>>();
            if power_checker_required.is_some_and(|h| h != POW_CHECKER_HEIGHT) {
                return None;
            }
            // Caution: this must be done after GKR and WHIR tracegen
            tracing::trace_span!("wrapper.generate_proving_ctxs", air_module = "Primitives",)
                .in_scope(|| {
                    tracing::trace_span!("wrapper.generate_trace", air = "PowerChecker").in_scope(
                        || {
                            let pow_bits_trace = ColMajorMatrix::from_row_major(
                                &self.power_checker_trace.generate_trace_row_major(),
                            );
                            ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
                                transport_matrix_h2d_col_major(&pow_bits_trace).unwrap(),
                            ));
                        },
                    );
                });
            let exp_bits_trace = tracing::trace_span!("wrapper.generate_trace", air = "ExpBitsLen")
                .in_scope(|| exp_bits_len_gen.generate_trace_device(exp_bits_len_required))?;
            ctx_per_trace.push(AirProvingContextV2::simple_no_pis(exp_bits_trace));
            Some(ctx_per_trace)
        }
    }
}
