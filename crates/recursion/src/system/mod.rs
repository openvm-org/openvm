//! Traits and types describing the core interfaces of the verifier sub-circuit. The verifier
//! sub-circuit verifies multiple proofs for the same child verifying key. It supports **recursive**
//! verification, where the child verifying key is equal to the verifying key of the verifier
//! circuit itself.
use std::{iter, sync::Arc};

use openvm_stark_backend::{AirRef, interaction::BusIndex};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, DIGEST_SIZE, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::{
        WIDTH,
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
        AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus, CommitmentsBus,
        ConstraintSumcheckRandomnessBus, EqNegBaseRandBus, EqNegResultBus, ExpressionClaimNMaxBus,
        FractionFolderInputBus, GkrModuleBus, HyperdimBus, LiftedHeightsBus, MerkleVerifyBus,
        Poseidon2Bus, PublicValuesBus, SelUniBus, StackingIndicesBus, StackingModuleBus,
        TranscriptBus, WhirModuleBus, WhirOpeningPointBus, XiRandomnessBus,
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
    whir::{FoldRecord, WhirModule},
};

mod dummy;
pub(crate) mod frame;

const BATCH_CONSTRAINT_MOD_IDX: usize = 3;

// TODO[jpw]: make this generic in <SC: StarkGenericConfig>
pub trait AirModule {
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
        ctx: Self::ModuleSpecificCtx,
    ) -> Vec<AirProvingContextV2<PB>>;
}

/// Trait for internal use within this crate.
/// This trait describes the trace generation for a single AIR within an AIR module.
// NOTE: this trait should stay `pub(crate)` and not `pub` since it is for internal implementation
// use only. The associated types should also stay private if possible.
pub(crate) trait ModuleChip<GC: GlobalTraceGenCtx, PB: ProverBackendV2>:
    Send + Sync
{
    type ModuleSpecificCtx;

    /// Returns the common main trace for this AIR. This trait only provides the interface for
    /// generating common main trace. Cached trace needs to be handled separately.
    fn generate_trace(
        &self,
        child_vk: &GC::ChildVerifyingKey,
        proofs: &GC::MultiProof,
        preflights: &GC::PreflightRecords,
        blob: &Self::ModuleSpecificCtx,
    ) -> PB::Matrix;
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
    pub poseidon2_bus: Poseidon2Bus,
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
    // Merkle bus message + actual commitment
    pub merkle_verify_logs: Vec<MerkleVerifyLog>,
    pub poseidon_inputs: Vec<[F; WIDTH]>,
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
    pub alphas: Vec<EF>,
    pub z0s: Vec<EF>,
    pub zj_roots: Vec<Vec<F>>,
    pub zjs: Vec<Vec<F>>,
    pub yjs: Vec<Vec<EF>>,
    pub gammas: Vec<EF>,
    pub pow_samples: Vec<F>,
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

#[derive(Clone, Debug, Default)]
pub struct MerkleVerifyLog {
    pub leaf_hashes: Vec<[F; DIGEST_SIZE]>,
    pub merkle_idx: usize,
    pub depth: usize,
    pub query_idx: usize,
    pub commit_major: usize,
    pub commit_minor: usize,
}

impl BusInventory {
    pub(crate) fn new(b: &mut BusIndexManager) -> Self {
        Self {
            transcript_bus: TranscriptBus::new(b.new_bus_idx()),
            poseidon2_bus: Poseidon2Bus::new(b.new_bus_idx()),
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
        }
    }
}

#[derive(Clone, Copy)]
enum TraceModuleRef<'a> {
    Transcript(&'a TranscriptModule),
    ProofShape(&'a ProofShapeModule),
    Gkr(&'a GkrModule),
    BatchConstraint(&'a BatchConstraintModule),
    Stacking(&'a StackingModule),
    Whir(&'a WhirModule),
}

impl<'a> TraceModuleRef<'a> {
    fn generate_cpu_ctxs(
        self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
        exp_bits_len_gen: &Arc<ExpBitsLenTraceGenerator>,
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        match self {
            TraceModuleRef::Transcript(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, ())
            }
            TraceModuleRef::ProofShape(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, ())
            }
            TraceModuleRef::Gkr(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, exp_bits_len_gen.clone())
            }
            TraceModuleRef::BatchConstraint(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, ())
            }
            TraceModuleRef::Stacking(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, ())
            }
            TraceModuleRef::Whir(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, exp_bits_len_gen.clone())
            }
        }
    }
}

/// The recursive verifier sub-circuit consists of multiple chips, grouped into **modules**.
///
/// This struct is stateful.
pub struct VerifierSubCircuit<const MAX_NUM_PROOFS: usize> {
    pub bus_inventory: BusInventory,

    transcript: TranscriptModule,
    proof_shape: ProofShapeModule,
    gkr: GkrModule,
    batch_constraint: BatchConstraintModule,
    stacking: StackingModule,
    whir: WhirModule,

    power_checker_air: Arc<PowerCheckerAir<2, 32>>,
    power_checker_trace: Arc<PowerCheckerTraceGenerator<2, 32>>,
}

impl<const MAX_NUM_PROOFS: usize> VerifierSubCircuit<MAX_NUM_PROOFS> {
    pub fn new(child_mvk: Arc<MultiStarkVerifyingKeyV2>) -> Self {
        let mut b = BusIndexManager::new();
        let bus_inventory = BusInventory::new(&mut b);
        let power_checker_trace = Arc::new(PowerCheckerTraceGenerator::<2, 32>::default());
        let power_checker_air = Arc::new(PowerCheckerAir::<2, 32> {
            pow_bus: bus_inventory.power_checker_bus,
            range_bus: bus_inventory.range_checker_bus,
        });

        let transcript = TranscriptModule::new(bus_inventory.clone(), child_mvk.inner.params);
        let child_mvk_frame = child_mvk.as_ref().into();
        let proof_shape = ProofShapeModule::new(
            &child_mvk_frame,
            &mut b,
            bus_inventory.clone(),
            power_checker_trace.clone(),
        );
        let gkr = GkrModule::new(&child_mvk, &mut b, bus_inventory.clone());
        let batch_constraint = BatchConstraintModule::new(
            &child_mvk,
            &mut b,
            bus_inventory.clone(),
            MAX_NUM_PROOFS,
            power_checker_trace.clone(),
        );
        let stacking = StackingModule::new(&child_mvk, &mut b, bus_inventory.clone());
        let whir = WhirModule::new(&child_mvk, &mut b, bus_inventory.clone());

        VerifierSubCircuit {
            bus_inventory,
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

    pub fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let exp_bits_len_air = ExpBitsLenAir::new(self.bus_inventory.exp_bits_len_bus);

        iter::empty()
            .chain(self.transcript.airs())
            .chain(self.proof_shape.airs())
            .chain(self.gkr.airs())
            .chain(self.batch_constraint.airs())
            .chain(self.stacking.airs())
            .chain(self.whir.airs())
            .chain([
                self.power_checker_air.clone() as AirRef<_>,
                Arc::new(exp_bits_len_air) as AirRef<_>,
            ])
            .collect()
    }

    /// Runs preflight for a single proof.
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
        self.proof_shape
            .run_preflight(child_vk, proof, &mut preflight, &mut sponge);
        self.gkr.run_preflight(proof, &mut preflight, &mut sponge);
        self.batch_constraint
            .run_preflight(proof, &mut preflight, &mut sponge);
        self.stacking
            .run_preflight(proof, &mut preflight, &mut sponge);
        self.whir.run_preflight(proof, &mut preflight, &mut sponge);

        preflight.transcript = sponge.into_log();
        preflight
    }

    // TODO: consider making trait for commit_child_vk and generate_proving_ctxs generic in
    // ProverBackendV2
    pub fn commit_child_vk(
        &self,
        engine: &BabyBearPoseidon2CpuEngineV2,
        child_vk: &MultiStarkVerifyingKeyV2,
    ) -> CommittedTraceDataV2<CpuBackendV2> {
        let (commitment, data) = self.batch_constraint.commit_child_vk(engine, child_vk);
        let height = 1 << data.layout.sorted_cols[0].2.log_height();
        CommittedTraceDataV2 {
            commitment,
            data: Arc::new(data),
            height,
        }
    }

    /// The generic `TS` allows using different transcript implementations for debugging purposes.
    /// The default type is use is `DuplexSpongeRecorder`.
    #[tracing::instrument(skip_all)]
    pub fn generate_proving_ctxs<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        child_vk_pcs_data: CommittedTraceDataV2<CpuBackendV2>,
        proofs: &[Proof],
    ) -> Vec<AirProvingContextV2<CpuBackendV2>>
    where
        TS: FiatShamirTranscript + TranscriptHistory + Default,
    {
        debug_assert!(proofs.len() <= MAX_NUM_PROOFS);
        let preflights = proofs
            .par_iter()
            .map(|proof| {
                let sponge = TS::default();
                self.run_preflight(sponge, child_vk, proof)
            })
            .collect::<Vec<_>>();

        self.power_checker_trace.reset();
        let exp_bits_len_gen = Arc::new(ExpBitsLenTraceGenerator::default());

        let modules = vec![
            TraceModuleRef::Transcript(&self.transcript),
            TraceModuleRef::ProofShape(&self.proof_shape),
            TraceModuleRef::Gkr(&self.gkr),
            TraceModuleRef::BatchConstraint(&self.batch_constraint),
            TraceModuleRef::Stacking(&self.stacking),
            TraceModuleRef::Whir(&self.whir),
        ];
        #[cfg(debug_assertions)]
        let modules_iter = modules.into_iter();
        #[cfg(not(debug_assertions))]
        let modules_iter = modules.into_par_iter();
        let mut ctxs_by_module = modules_iter
            .map(|module| {
                module.generate_cpu_ctxs(child_vk, proofs, &preflights, &exp_bits_len_gen)
            })
            .collect::<Vec<_>>();
        ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX].cached_mains =
            vec![child_vk_pcs_data];
        let mut ctx_per_trace = ctxs_by_module.into_iter().flatten().collect::<Vec<_>>();
        // Caution: this must be done after GKR and WHIR tracegen
        ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
            ColMajorMatrix::from_row_major(&self.power_checker_trace.generate_trace_row_major()),
        ));
        ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
            ColMajorMatrix::from_row_major(&exp_bits_len_gen.generate_trace_row_major()),
        ));

        ctx_per_trace
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_tracegen {
    use std::iter::zip;

    use cuda_backend_v2::{
        BabyBearPoseidon2GpuEngineV2, GpuBackendV2, transport_matrix_h2d_col_major,
    };

    use super::*;
    use crate::cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu};

    impl<'a> TraceModuleRef<'a> {
        fn generate_gpu_ctxs(
            self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            exp_bits_len_gen: &Arc<ExpBitsLenTraceGenerator>,
        ) -> Vec<AirProvingContextV2<cuda_backend_v2::GpuBackendV2>> {
            match self {
                TraceModuleRef::Transcript(module) => {
                    module.generate_proving_ctxs(child_vk, proofs, preflights, ())
                }
                TraceModuleRef::ProofShape(module) => {
                    module.generate_proving_ctxs(child_vk, proofs, preflights, ())
                }
                TraceModuleRef::Gkr(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    exp_bits_len_gen.clone(),
                ),
                TraceModuleRef::BatchConstraint(module) => {
                    module.generate_proving_ctxs(child_vk, proofs, preflights, ())
                }
                TraceModuleRef::Stacking(module) => {
                    module.generate_proving_ctxs(child_vk, proofs, preflights, ())
                }
                TraceModuleRef::Whir(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    exp_bits_len_gen.clone(),
                ),
            }
        }
    }

    impl<const MAX_NUM_PROOFS: usize> VerifierSubCircuit<MAX_NUM_PROOFS> {
        pub fn commit_child_vk_gpu(
            &self,
            engine: &BabyBearPoseidon2GpuEngineV2,
            child_vk: &MultiStarkVerifyingKeyV2,
        ) -> CommittedTraceDataV2<GpuBackendV2> {
            let (commitment, data) = self.batch_constraint.commit_child_vk_gpu(engine, child_vk);
            let height = 1 << data.layout.sorted_cols[0].2.log_height();
            CommittedTraceDataV2 {
                commitment,
                data: Arc::new(data),
                height,
            }
        }

        pub fn generate_proving_ctxs_gpu<TS>(
            &self,
            child_vk: &MultiStarkVerifyingKeyV2,
            child_vk_pcs_data: CommittedTraceDataV2<GpuBackendV2>,
            proofs: &[Proof],
        ) -> Vec<AirProvingContextV2<GpuBackendV2>>
        where
            TS: FiatShamirTranscript + TranscriptHistory + Default,
        {
            debug_assert!(proofs.len() <= MAX_NUM_PROOFS);
            let child_vk_gpu = VerifyingKeyGpu::new(child_vk);
            let proofs_gpu = proofs
                .iter()
                .map(|proof_cpu| ProofGpu::new(child_vk, proof_cpu))
                .collect::<Vec<_>>();
            // Run CPU preflight for each proof in parallel
            let preflights_cpu = proofs
                .par_iter()
                .map(|proof| {
                    let sponge = TS::default();
                    self.run_preflight(sponge, child_vk, proof)
                })
                .collect::<Vec<_>>();

            let exp_bits_len_gen = Arc::new(ExpBitsLenTraceGenerator::default());
            self.power_checker_trace.reset();

            // NOTE: avoid par_iter for now so H2D transfer all happens on same stream to avoid sync
            // issues
            let preflights_gpu = zip(proofs, preflights_cpu)
                .map(|(proof, preflight_cpu)| PreflightGpu::new(child_vk, proof, &preflight_cpu))
                .collect::<Vec<_>>();
            let modules = vec![
                TraceModuleRef::Transcript(&self.transcript),
                TraceModuleRef::ProofShape(&self.proof_shape),
                TraceModuleRef::Gkr(&self.gkr),
                TraceModuleRef::BatchConstraint(&self.batch_constraint),
                TraceModuleRef::Stacking(&self.stacking),
                TraceModuleRef::Whir(&self.whir),
            ];
            // PERF[jpw]: we avoid par_iter so that kernel launches occur on the same stream.
            // This can be parallelized to separate streams for more CUDA stream parallelism, but it
            // will require recording events so streams properly sync for cudaMemcpyAsync and kernel
            // launches
            let mut ctxs_by_module = modules
                .into_iter()
                .map(|module| {
                    module.generate_gpu_ctxs(
                        &child_vk_gpu,
                        &proofs_gpu,
                        &preflights_gpu,
                        &exp_bits_len_gen,
                    )
                })
                .collect::<Vec<_>>();
            ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX]
                .cached_mains = vec![child_vk_pcs_data];
            let mut ctx_per_trace = ctxs_by_module.into_iter().flatten().collect::<Vec<_>>();
            // TODO: move this to gpu
            // Caution: this must be done after GKR and WHIR tracegen
            let pow_bits_trace = ColMajorMatrix::from_row_major(
                &self.power_checker_trace.generate_trace_row_major(),
            );
            ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
                transport_matrix_h2d_col_major(&pow_bits_trace).unwrap(),
            ));

            let exp_bits_trace =
                ColMajorMatrix::from_row_major(&exp_bits_len_gen.generate_trace_row_major());
            ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
                transport_matrix_h2d_col_major(&exp_bits_trace).unwrap(),
            ));

            ctx_per_trace
        }
    }
}
