use std::sync::Arc;

use openvm_stark_backend::{AirRef, interaction::BusIndex, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{
    EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory, TranscriptLog},
    proof::{Proof, TraceVData},
};

use crate::{
    batch_constraint::BatchConstraintModule,
    bus::{
        AirHeightsBus, AirPartShapeBus, AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus,
        CommitmentsBus, ConstraintSumcheckRandomnessBus, ExpBitsLenBus, GkrModuleBus, Poseidon2Bus,
        PublicValuesBus, StackingIndicesBus, StackingModuleBus, TranscriptBus, WhirModuleBus,
        WhirOpeningPointBus, XiRandomnessBus,
    },
    gkr::GkrModule,
    primitives::exp_bits_len::ExpBitsLenAir,
    proof_shape::ProofShapeModule,
    stacking::StackingModule,
    transcript::TranscriptModule,
    whir::{FoldRecord, WhirModule},
};

mod dummy;

pub trait AirModule<TS: FiatShamirTranscript + TranscriptHistory> {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>>;
    fn run_preflight(&self, proof: &Proof, preflight: &mut Preflight, transcript: &mut TS);
    fn generate_proof_inputs(
        &self,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>>;
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

#[derive(Debug, Default)]
pub struct Preflight {
    /// The concatenated sequence of observes/samples. Not available during preflight; populated
    /// after.
    pub transcript: TranscriptLog,
    pub proof_shape: ProofShapePreflight,
    pub gkr: GkrPreflight,
    pub batch_constraint: BatchConstraintPreflight,
    pub stacking: StackingPreflight,
    pub whir: WhirPreflight,
}

#[derive(Debug, Default)]
pub struct ProofShapePreflight {
    pub stacked_common_width: usize,
    pub sorted_trace_vdata: Vec<(usize, TraceVData)>,
    pub n_global: usize,
    pub n_max: usize,
    pub n_logup: usize,
    pub l_skip: usize,
    pub logup_pow_bits: usize,
    pub post_tidx: usize,
    pub pvs_tidx: Vec<usize>,
}

#[derive(Debug, Default)]
pub struct GkrPreflight {
    pub post_tidx: usize,
    pub post_layer_tidx: usize,
    pub xi: Vec<(usize, EF)>,
    /// For each sumcheck round: (layer_idx, sumcheck_round, claim_in, claim_out, eq_in, eq_out)
    pub sumcheck_round_data: Vec<(usize, usize, EF, EF, EF, EF)>,
    /// For each layer (1..num_layers): (new_claim, eq_at_r_prime) - the final sumcheck output
    pub layer_sumcheck_output: Vec<(EF, EF)>,
    /// For each layer (1..num_layers): the claim sent to sumcheck (numer_claim_prev + lambda *
    /// denom_claim_prev)
    pub layer_claim: Vec<EF>,
}

#[derive(Debug, Default)]
pub struct BatchConstraintPreflight {
    pub post_tidx: usize,
    pub sumcheck_rnd: Vec<EF>,
}

#[derive(Debug, Default)]
pub struct StackingPreflight {
    pub intermediate_tidx: [usize; 3],
    pub post_tidx: usize,
    pub univariate_poly_rand_eval: EF,
    pub stacking_batching_challenge: EF,
    pub lambda: EF,
    pub sumcheck_rnd: Vec<EF>,
}

#[derive(Debug, Default)]
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
    fn new(b: &mut BusIndexManager) -> Self {
        Self {
            transcript_bus: TranscriptBus::new(b.new_bus_idx()),
            poseidon2_bus: Poseidon2Bus::new(b.new_bus_idx()),

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

pub struct VerifierCircuit<TS> {
    modules: Vec<Box<dyn AirModule<TS>>>,
    exp_bits_len_air: Arc<ExpBitsLenAir>,
}

impl<TS: FiatShamirTranscript + TranscriptHistory> VerifierCircuit<TS> {
    pub fn new(child_mvk: Arc<MultiStarkVerifyingKeyV2>) -> Self {
        let mut b = BusIndexManager::new();
        let bus_inventory = BusInventory::new(&mut b);
        let exp_bits_len_air = Arc::new(ExpBitsLenAir::new(bus_inventory.exp_bits_len_bus));

        let transcript_module = TranscriptModule::new(child_mvk.clone(), bus_inventory.clone());
        let proof_shape_module =
            ProofShapeModule::new(child_mvk.clone(), &mut b, bus_inventory.clone());
        let gkr_module = GkrModule::new(
            child_mvk.clone(),
            &mut b,
            bus_inventory.clone(),
            exp_bits_len_air.clone(),
        );
        let batch_constraint_module =
            BatchConstraintModule::new(child_mvk.clone(), bus_inventory.clone());
        let stacking_module = StackingModule::new(child_mvk.clone(), &mut b, bus_inventory.clone());
        let whir_module = WhirModule::new(
            child_mvk.clone(),
            &mut b,
            bus_inventory.clone(),
            exp_bits_len_air.clone(),
        );

        let modules: Vec<Box<dyn AirModule<TS>>> = vec![
            Box::new(transcript_module),
            Box::new(proof_shape_module),
            Box::new(gkr_module),
            Box::new(batch_constraint_module),
            Box::new(stacking_module),
            Box::new(whir_module),
        ];
        VerifierCircuit {
            modules,
            exp_bits_len_air,
        }
    }

    pub fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let mut airs = vec![];
        for module in &self.modules {
            airs.extend(module.airs());
        }
        airs.push(self.exp_bits_len_air.clone());
        airs
    }

    pub fn run_preflight(&self, mut sponge: TS, proof: &Proof) -> Preflight {
        let mut preflight = Preflight::default();
        for module in self.modules.iter() {
            module.run_preflight(proof, &mut preflight, &mut sponge);
        }
        preflight.transcript = sponge.into_log();
        preflight
    }

    pub fn generate_proof_inputs(&self, sponge: TS, proof: &Proof) -> Vec<AirProofRawInput<F>> {
        let preflight = self.run_preflight(sponge, proof);

        let mut proof_inputs = vec![];
        for (i, module) in self.modules.iter().enumerate() {
            let module_proof_inputs = module.generate_proof_inputs(proof, &preflight);
            debug_assert_eq!(
                module_proof_inputs.len(),
                module.airs().len(),
                "module {} generated {} proof inputs but {} airs",
                i,
                module_proof_inputs.len(),
                module.airs().len()
            );
            proof_inputs.extend(module_proof_inputs);
        }
        proof_inputs.push(self.exp_bits_len_air.generate_proof_input());
        proof_inputs
    }
}
