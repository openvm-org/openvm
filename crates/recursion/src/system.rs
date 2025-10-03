use openvm_stark_backend::{AirRef, interaction::BusIndex, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldExtensionAlgebra;
use stark_backend_v2::{
    D_EF, DIGEST_SIZE, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::DuplexSponge,
    proof::{Proof, TraceVData},
};

use crate::{
    batch_constraint::BatchConstraintModule,
    bus::{
        AirPartShapeBus, AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus,
        ConstraintSumcheckRandomnessBus, GkrModuleBus, GkrRandomnessBus,
        InitialZerocheckRandomnessBus, PublicValuesBus, StackingClaimsBus, StackingCommitmentsBus,
        StackingModuleBus, StackingSumcheckRandomnessBus, StackingWidthsBus, TranscriptBus,
        WhirModuleBus,
    },
    gkr::GkrModule,
    proof_shape::ProofShapeModule,
    stacking::StackingModule,
    transcript::TranscriptModule,
    whir::WhirModule,
};

pub trait AirModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>>;
    fn run_preflight(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight,
    );
    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
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

pub struct VerifierCircuit {
    modules: Vec<Box<dyn AirModule>>,
}

#[derive(Clone, Debug)]
pub struct BusInventory {
    // Control flow buses
    pub transcript_bus: TranscriptBus,
    pub gkr_module_bus: GkrModuleBus,
    pub bc_module_bus: BatchConstraintModuleBus,
    pub stacking_module_bus: StackingModuleBus,
    pub whir_module_bus: WhirModuleBus,

    // Data buses
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub stacking_widths_bus: StackingWidthsBus,
    pub stacking_commitments_bus: StackingCommitmentsBus,
    pub public_values_bus: PublicValuesBus,

    // Randomness buses
    pub initial_zerocheck_randomness_bus: InitialZerocheckRandomnessBus,
    pub gkr_randomness_bus: GkrRandomnessBus,
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,

    // Claims buses
    pub column_claims_bus: ColumnClaimsBus,
    pub stacking_claims_bus: StackingClaimsBus,
}

#[derive(Debug, Default)]
pub struct Transcript {
    pub(crate) data: Vec<F>,
    pub(crate) is_sample: Vec<bool>,
    pub(crate) sponge: DuplexSponge,
}

impl Transcript {
    pub fn observe(&mut self, value: F) {
        self.data.push(value);
        self.is_sample.push(false);
        self.sponge.observe(value);
    }

    pub fn observe_ext(&mut self, value: EF) {
        self.data.extend_from_slice(&value.as_base_slice());
        self.is_sample.extend_from_slice(&[false; D_EF]);
        self.sponge.observe_ext(value);
    }

    pub fn observe_commit(&mut self, digest: [F; DIGEST_SIZE]) {
        self.data.extend_from_slice(&digest);
        self.is_sample.extend_from_slice(&[false; DIGEST_SIZE]);
        self.sponge.observe_commit(digest);
    }

    pub fn observe_slice(&mut self, slc: &[F]) {
        for x in slc {
            self.observe(*x);
        }
    }

    pub fn sample(&mut self) -> F {
        let sample = self.sponge.sample();
        self.data.push(sample);
        self.is_sample.push(true);
        sample
    }

    pub fn sample_ext(&mut self) -> EF {
        let sample = self.sponge.sample_ext();
        self.data.extend_from_slice(&sample.as_base_slice());
        self.is_sample.extend_from_slice(&[true; D_EF]);
        sample
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Default)]
pub struct Preflight {
    pub transcript: Transcript,
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
    pub n_max: usize,
    pub n_logup: usize,
    pub post_tidx: usize,
}

#[derive(Debug, Default)]
pub struct GkrPreflight {
    pub post_tidx: usize,
    pub input_layer_numerator_claim: EF,
    pub input_layer_denominator_claim: EF,
}

#[derive(Debug, Default)]
pub struct BatchConstraintPreflight {
    pub post_tidx: usize,
}

#[derive(Debug, Default)]
pub struct StackingPreflight {
    pub post_tidx: usize,
}

#[derive(Debug, Default)]
pub struct WhirPreflight {}

impl BusInventory {
    pub fn new() -> Self {
        let mut b = BusIndexManager::new();

        Self {
            // Control flow buses
            transcript_bus: TranscriptBus::new(b.new_bus_idx()),
            gkr_module_bus: GkrModuleBus::new(b.new_bus_idx()),
            bc_module_bus: BatchConstraintModuleBus::new(b.new_bus_idx()),
            stacking_module_bus: StackingModuleBus::new(b.new_bus_idx()),
            whir_module_bus: WhirModuleBus::new(b.new_bus_idx()),

            // Data buses
            air_shape_bus: AirShapeBus::new(b.new_bus_idx()),
            air_part_shape_bus: AirPartShapeBus::new(b.new_bus_idx()),
            stacking_widths_bus: StackingWidthsBus::new(b.new_bus_idx()),
            stacking_commitments_bus: StackingCommitmentsBus::new(b.new_bus_idx()),
            public_values_bus: PublicValuesBus::new(b.new_bus_idx()),

            // Randomness buses
            initial_zerocheck_randomness_bus: InitialZerocheckRandomnessBus::new(b.new_bus_idx()),
            gkr_randomness_bus: GkrRandomnessBus::new(b.new_bus_idx()),
            constraint_randomness_bus: ConstraintSumcheckRandomnessBus::new(b.new_bus_idx()),
            stacking_randomness_bus: StackingSumcheckRandomnessBus::new(b.new_bus_idx()),

            // Claims buses
            column_claims_bus: ColumnClaimsBus::new(b.new_bus_idx()),
            stacking_claims_bus: StackingClaimsBus::new(b.new_bus_idx()),
        }
    }

    pub fn air_part_shape_bus(&self) -> AirPartShapeBus {
        self.air_part_shape_bus
    }
}

impl VerifierCircuit {
    pub fn new() -> Self {
        let bus_inventory = BusInventory::new();

        let transcript_module = TranscriptModule::new(bus_inventory.clone());
        let proof_shape_module = ProofShapeModule::new(bus_inventory.clone());
        let gkr_module = GkrModule::new(bus_inventory.clone());
        let batch_constraint_module = BatchConstraintModule::new(bus_inventory.clone());
        let stacking_module = StackingModule::new(bus_inventory.clone());
        let whir_module = WhirModule::new(bus_inventory.clone());

        let modules: Vec<Box<dyn AirModule>> = vec![
            Box::new(transcript_module),
            Box::new(proof_shape_module),
            Box::new(gkr_module),
            Box::new(batch_constraint_module),
            Box::new(stacking_module),
            Box::new(whir_module),
        ];
        VerifierCircuit { modules }
    }

    pub fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let mut airs = vec![];
        for module in &self.modules {
            airs.extend(module.airs());
        }
        airs
    }

    pub fn run_preflight(&self, vk: &MultiStarkVerifyingKeyV2, proof: &Proof) -> Preflight {
        let mut preflight = Preflight::default();
        for module in self.modules.iter() {
            module.run_preflight(vk, proof, &mut preflight);
        }
        preflight
    }

    pub fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
    ) -> Vec<AirProofRawInput<F>> {
        let preflight = self.run_preflight(vk, proof);

        let mut proof_inputs = vec![];
        for (i, module) in self.modules.iter().enumerate() {
            let module_proof_inputs = module.generate_proof_inputs(vk, proof, &preflight);
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
        proof_inputs
    }
}
