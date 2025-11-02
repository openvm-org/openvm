//! Traits and types describing the core interfaces of the verifier sub-circuit. The verifier
//! sub-circuit verifies multiple proofs for the same child verifying key. It supports **recursive**
//! verification, where the child verifying key is equal to the verifying key of the verifier
//! circuit itself.
use std::{iter, sync::Arc};

use openvm_stark_backend::{AirRef, interaction::BusIndex};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, DIGEST_SIZE, Digest, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory, TranscriptLog},
    proof::{Proof, TraceVData},
    prover::{
        AirProvingContextV2, ColMajorMatrix, CpuBackendV2, ProverBackendV2,
        stacked_pcs::StackedPcsData,
    },
};

use crate::{
    batch_constraint::{BatchConstraintModule, LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX},
    bus::{
        AirHeightsBus, AirPartShapeBus, AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus,
        CommitmentsBus, ConstraintSumcheckRandomnessBus, ExpBitsLenBus, GkrModuleBus,
        MerkleVerifyBus, MerkleVerifyBusMessage, Poseidon2Bus, PublicValuesBus, StackingIndicesBus,
        StackingModuleBus, TranscriptBus, WhirModuleBus, WhirOpeningPointBus, XiRandomnessBus,
    },
    gkr::GkrModule,
    primitives::exp_bits_len::ExpBitsLenAir,
    proof_shape::ProofShapeModule,
    stacking::StackingModule,
    transcript::TranscriptModule,
    whir::{FoldRecord, WhirModule},
};

mod dummy;

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
    fn generate_proving_ctxs(
        &self,
        child_vk: &GC::ChildVerifyingKey,
        proofs: &GC::MultiProof,
        preflights: &GC::PreflightRecords,
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
    pub air_part_shape_bus: AirPartShapeBus,
    pub air_heights_bus: AirHeightsBus,
    pub stacking_indices_bus: StackingIndicesBus,
    pub commitments_bus: CommitmentsBus,
    pub public_values_bus: PublicValuesBus,

    // Randomness buses
    pub xi_randomness_bus: XiRandomnessBus,
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,

    // Claims buses
    pub column_claims_bus: ColumnClaimsBus,

    // Exp bits length bus
    pub exp_bits_len_bus: ExpBitsLenBus,
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
    pub merkle_verify_logs: Vec<(MerkleVerifyBusMessage<F>, [F; DIGEST_SIZE])>,
}

#[derive(Clone, Debug, Default)]
pub struct ProofShapePreflight {
    pub sorted_trace_vdata: Vec<(usize, TraceVData)>,
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
    pub tidx_before_univariate: usize,
    pub tidx_before_multilinear: usize,
    pub tidx_before_column_openings: usize,
    pub post_tidx: usize,
    pub xi: Vec<EF>,
    pub sumcheck_rnd: Vec<EF>,
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
    pub query_indices: Vec<u32>,
    pub tidx_per_round: Vec<usize>,
    pub query_tidx_per_round: Vec<usize>,
    pub initial_claim_per_round: Vec<EF>,
    pub post_sumcheck_claims: Vec<EF>,
    pub pre_query_claims: Vec<EF>,
    pub eq_partials: Vec<EF>,
    pub fold_records: Vec<FoldRecord>,
    pub initial_round_coset_vals: Vec<Vec<EF>>,
    pub final_poly_at_u: EF,
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
            air_part_shape_bus: AirPartShapeBus::new(b.new_bus_idx()),
            air_heights_bus: AirHeightsBus::new(b.new_bus_idx()),
            stacking_indices_bus: StackingIndicesBus::new(b.new_bus_idx()),
            commitments_bus: CommitmentsBus::new(b.new_bus_idx()),
            public_values_bus: PublicValuesBus::new(b.new_bus_idx()),

            // Randomness buses
            xi_randomness_bus: XiRandomnessBus::new(b.new_bus_idx()),
            constraint_randomness_bus: ConstraintSumcheckRandomnessBus::new(b.new_bus_idx()),
            whir_opening_point_bus: WhirOpeningPointBus::new(b.new_bus_idx()),

            // Claims buses
            column_claims_bus: ColumnClaimsBus::new(b.new_bus_idx()),

            exp_bits_len_bus: ExpBitsLenBus::new(b.new_bus_idx()),
            // Stacking module internal buses
        }
    }
}

impl BusInventory {
    pub fn air_part_shape_bus(&self) -> AirPartShapeBus {
        self.air_part_shape_bus
    }
}

/// The recursive verifier sub-circuit consists of multiple chips, grouped into **modules**.
///
/// This struct is stateful.
pub struct VerifierSubCircuit<const MAX_NUM_PROOFS: usize> {
    transcript: TranscriptModule,
    proof_shape: ProofShapeModule,
    gkr: GkrModule,
    batch_constraint: BatchConstraintModule,
    stacking: StackingModule,
    whir: WhirModule,

    exp_bits_len_air: Arc<ExpBitsLenAir>,
}

impl<const MAX_NUM_PROOFS: usize> VerifierSubCircuit<MAX_NUM_PROOFS> {
    pub fn new(child_mvk: Arc<MultiStarkVerifyingKeyV2>) -> Self {
        let mut b = BusIndexManager::new();
        let bus_inventory = BusInventory::new(&mut b);
        let exp_bits_len_air = Arc::new(ExpBitsLenAir::new(bus_inventory.exp_bits_len_bus));

        let transcript = TranscriptModule::new(bus_inventory.clone());
        let proof_shape = ProofShapeModule::new(child_mvk.clone(), &mut b, bus_inventory.clone());
        let gkr = GkrModule::new(
            &child_mvk,
            &mut b,
            bus_inventory.clone(),
            exp_bits_len_air.clone(),
        );
        let batch_constraint =
            BatchConstraintModule::new(&child_mvk, &mut b, bus_inventory.clone(), MAX_NUM_PROOFS);
        let stacking = StackingModule::new(&child_mvk, &mut b, bus_inventory.clone());
        let whir = WhirModule::new(
            &child_mvk,
            &mut b,
            bus_inventory.clone(),
            exp_bits_len_air.clone(),
        );

        VerifierSubCircuit {
            transcript,
            proof_shape,
            gkr,
            batch_constraint,
            stacking,
            whir,
            exp_bits_len_air,
        }
    }

    pub fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        iter::empty()
            .chain(self.transcript.airs())
            .chain(self.proof_shape.airs())
            .chain(self.gkr.airs())
            .chain(self.batch_constraint.airs())
            .chain(self.stacking.airs())
            .chain(self.whir.airs())
            .chain([self.exp_bits_len_air.clone() as AirRef<_>])
            .collect()
    }

    /// Runs preflight for a single proof.
    pub fn run_preflight<TS>(&self, mut sponge: TS, proof: &Proof) -> Preflight
    where
        TS: FiatShamirTranscript + TranscriptHistory,
    {
        let mut preflight = Preflight::default();
        // NOTE: it is not required that we group preflight into modules
        self.proof_shape
            .run_preflight(proof, &mut preflight, &mut sponge);
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
    ) -> (Digest, Arc<StackedPcsData<F, Digest>>) {
        let (commit, data) = self.batch_constraint.commit_child_vk(engine, child_vk);
        (commit, Arc::new(data))
    }

    /// The generic `TS` allows using different transcript implementations for debugging purposes.
    /// The default type is use is `DuplexSpongeRecorder`.
    pub fn generate_proving_ctxs<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        child_vk_pcs_data: (Digest, Arc<StackedPcsData<F, Digest>>),
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
                self.run_preflight(sponge, proof)
            })
            .collect::<Vec<_>>();

        const BATCH_CONSTRAINT_MOD_IDX: usize = 3;
        let modules = vec![
            &self.transcript as &dyn TraceGenModule<GlobalCtxCpu, CpuBackendV2>,
            &self.proof_shape as &dyn TraceGenModule<_, _>,
            &self.gkr as &dyn TraceGenModule<_, _>,
            &self.batch_constraint as &dyn TraceGenModule<_, _>,
            &self.stacking as &dyn TraceGenModule<_, _>,
            &self.whir as &dyn TraceGenModule<_, _>,
        ];
        let mut ctxs_by_module = modules
            .into_par_iter()
            .map(|module| module.generate_proving_ctxs(child_vk, proofs, &preflights))
            .collect::<Vec<_>>();
        ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX].cached_mains =
            vec![child_vk_pcs_data];
        let mut ctx_per_trace = ctxs_by_module.into_iter().flatten().collect::<Vec<_>>();
        // Caution: this must be done after GKR and WHIR tracegen
        ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
            ColMajorMatrix::from_row_major(&self.exp_bits_len_air.generate_trace_row_major()),
        ));
        self.clear();

        ctx_per_trace
    }

    // TODO[jpw]: remove this, Circuit should not be stateful. INT-5366
    pub(crate) fn clear(&self) {
        self.exp_bits_len_air.records.lock().unwrap().clear();
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_tracegen {
    use cuda_backend_v2::{
        BabyBearPoseidon2GpuEngineV2, GpuBackendV2, stacked_pcs::StackedPcsDataGpu,
        transport_matrix_h2d_col_major,
    };

    use super::*;
    use crate::cuda::{
        GlobalCtxGpu, preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu,
    };

    impl<const MAX_NUM_PROOFS: usize> VerifierSubCircuit<MAX_NUM_PROOFS> {
        pub fn commit_child_vk_gpu(
            &self,
            engine: &BabyBearPoseidon2GpuEngineV2,
            child_vk: &MultiStarkVerifyingKeyV2,
        ) -> (Digest, Arc<StackedPcsDataGpu<F, Digest>>) {
            let (commit, data) = self.batch_constraint.commit_child_vk_gpu(engine, child_vk);
            (commit, Arc::new(data))
        }

        pub fn generate_proving_ctxs_gpu<TS>(
            &self,
            child_vk: &MultiStarkVerifyingKeyV2,
            child_vk_pcs_data: (Digest, Arc<StackedPcsDataGpu<F, Digest>>),
            proofs: &[Proof],
        ) -> Vec<AirProvingContextV2<GpuBackendV2>>
        where
            TS: FiatShamirTranscript + TranscriptHistory + Default,
        {
            debug_assert!(proofs.len() <= MAX_NUM_PROOFS);
            let child_vk_gpu = VerifyingKeyGpu::new(&child_vk);
            let proofs_gpu = proofs
                .into_iter()
                .map(|proof_cpu| ProofGpu::new(child_vk, &proof_cpu))
                .collect::<Vec<_>>();
            let preflights_gpu = proofs
                .par_iter()
                .map(|proof| {
                    let sponge = TS::default();
                    let preflight_cpu = self.run_preflight(sponge, proof);
                    // NOTE: this uses one stream per thread for H2D transfer
                    PreflightGpu::new(child_vk, proof, &preflight_cpu)
                })
                .collect::<Vec<_>>();
            const BATCH_CONSTRAINT_MOD_IDX: usize = 3;
            let modules = vec![
                &self.transcript as &dyn TraceGenModule<GlobalCtxGpu, GpuBackendV2>,
                &self.proof_shape as &dyn TraceGenModule<_, _>,
                &self.gkr as &dyn TraceGenModule<_, _>,
                &self.batch_constraint as &dyn TraceGenModule<_, _>,
                &self.stacking as &dyn TraceGenModule<_, _>,
                &self.whir as &dyn TraceGenModule<_, _>,
            ];
            let mut ctxs_by_module = modules
                .into_par_iter()
                .map(|module| {
                    module.generate_proving_ctxs(&child_vk_gpu, &proofs_gpu, &preflights_gpu)
                })
                .collect::<Vec<_>>();
            ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX]
                .cached_mains = vec![child_vk_pcs_data];
            let mut ctx_per_trace = ctxs_by_module.into_iter().flatten().collect::<Vec<_>>();
            // TODO: move this to gpu
            // Caution: this must be done after GKR and WHIR tracegen
            let exp_bits_trace =
                ColMajorMatrix::from_row_major(&self.exp_bits_len_air.generate_trace_row_major());
            ctx_per_trace.push(AirProvingContextV2::simple_no_pis(
                transport_matrix_h2d_col_major(&exp_bits_trace).unwrap(),
            ));
            self.clear();

            ctx_per_trace
        }
    }
}
