use core::cmp::max;

use openvm_stark_backend::{AirRef, interaction::BusIndex, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

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
    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        public_values_per_air: &[Vec<F>],
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

pub struct Preflight {
    pub transcript: Vec<F>,
    pub gkr_tidx: usize,
    pub batch_constraint_tidx: usize,
    pub stacking_tidx: usize,
    pub whir_tidx: usize,

    pub n_max: usize,
    pub num_present_airs: usize,
    pub stacked_common_width: usize,

    pub gkr_input_layer_numerator_claim: EF,
    pub gkr_input_layer_denominator_claim: EF,
}

impl Preflight {
    pub fn run(vk: &MultiStarkVerifyingKeyV2, proof: &Proof) -> Self {
        let &MultiStarkVerifyingKeyV2 {
            pre_hash: _vk_prehash,
            inner: vk,
        } = &vk;

        let num_optional_airs = vk
            .per_air
            .iter()
            .map(|avk| 1 - avk.is_required as usize)
            .sum::<usize>();
        let num_present_optional_airs = proof
            .is_optional_air_present
            .iter()
            .map(|b| *b as usize)
            .sum::<usize>();

        let num_present_airs = vk.per_air.len() - (num_optional_airs - num_present_optional_airs);

        let mut num_common_cells: u64 = 0;
        let mut is_optional_air_present = proof.is_optional_air_present.iter();
        let mut log_heights = proof.log_heights.iter();
        let mut n_max = 0;
        for avk in &vk.per_air {
            let is_present = if avk.is_required {
                true
            } else {
                *is_optional_air_present.next().unwrap()
            };
            if is_present {
                let log_height = if avk.preprocessed_data.is_some() {
                    avk.preprocessed_data
                        .as_ref()
                        .map(|avk| avk.log_height)
                        .unwrap()
                } else {
                    *log_heights.next().unwrap() as usize
                };
                n_max = max(n_max, log_height as usize - vk.params.l_skip);
                num_common_cells += (1 << log_height) * avk.params.width.common_main as u64;
            }
        }
        let stack_height = 1 << (vk.params.l_skip + vk.params.n_stack);

        Self {
            transcript: vec![F::ZERO; 1000],
            gkr_tidx: 100,
            batch_constraint_tidx: 200,
            stacking_tidx: 300,
            whir_tidx: 400,
            num_present_airs,
            n_max,
            stacked_common_width: ((num_common_cells + stack_height - 1) / stack_height) as usize,
            gkr_input_layer_numerator_claim: EF::from_canonical_usize(123),
            gkr_input_layer_denominator_claim: EF::from_canonical_usize(456),
        }
    }
}

impl BusInventory {
    pub fn new() -> Self {
        let mut b = BusIndexManager::new();

        dbg!(Self {
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
        })
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

    pub fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        public_values_per_air: &[Vec<F>],
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        let mut proof_inputs = vec![];
        for (i, module) in self.modules.iter().enumerate() {
            let module_proof_inputs =
                module.generate_proof_inputs(vk, proof, public_values_per_air, &preflight);
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
