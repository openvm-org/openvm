use std::sync::Arc;

use openvm_stark_backend::{AirRef, interaction::BusIndex, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldExtensionAlgebra;
use stark_backend_v2::{
    D_EF, DIGEST_SIZE, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{Proof, TraceVData},
};

use crate::{
    batch_constraint::BatchConstraintModule,
    bus::{
        AirPartShapeBus, AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus, CommitmentsBus,
        ConstraintSumcheckRandomnessBus, GkrModuleBus, PowerCheckerBus, PublicValuesBus,
        RangeCheckerBus, StackingIndicesBus, StackingModuleBus, StackingSumcheckRandomnessBus,
        TranscriptBus, WhirModuleBus, XiRandomnessBus,
    },
    gkr::GkrModule,
    primitives::{pow::PowerCheckerAir, range::RangeCheckerAir},
    proof_shape::ProofShapeModule,
    stacking::StackingModule,
    transcript::TranscriptModule,
    whir::WhirModule,
};

mod dummy;

pub trait AirModule<TS: FiatShamirTranscript> {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>>;
    fn run_preflight(&self, proof: &Proof, preflight: &mut Preflight<TS>);
    fn generate_proof_inputs(
        &self,
        proof: &Proof,
        preflight: &Preflight<TS>,
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
    pub gkr_module_bus: GkrModuleBus,
    pub bc_module_bus: BatchConstraintModuleBus,
    pub stacking_module_bus: StackingModuleBus,
    pub whir_module_bus: WhirModuleBus,

    // Data buses
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub stacking_indices_bus: StackingIndicesBus,
    pub commitments_bus: CommitmentsBus,
    pub public_values_bus: PublicValuesBus,

    // Randomness buses
    pub xi_randomness_bus: XiRandomnessBus,
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,

    // Claims buses
    pub column_claims_bus: ColumnClaimsBus,

    // Peripheral buses
    pub range_checker_bus: RangeCheckerBus,
    pub power_of_two_bus: PowerCheckerBus,
}

#[derive(Debug, Default)]
pub struct Transcript<TS: FiatShamirTranscript> {
    pub(crate) data: Vec<F>,
    pub(crate) is_sample: Vec<bool>,
    pub(crate) sponge: TS,
}

impl<TS: FiatShamirTranscript> Transcript<TS> {
    pub fn new(sponge: TS) -> Self {
        Self {
            data: vec![],
            is_sample: vec![],
            sponge,
        }
    }
}

impl<TS: FiatShamirTranscript> Transcript<TS> {
    pub fn observe(&mut self, value: F) {
        self.data.push(value);
        self.is_sample.push(false);
        self.sponge.observe(value);
    }

    pub fn observe_ext(&mut self, value: EF) {
        self.data.extend_from_slice(value.as_base_slice());
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
        self.data.extend_from_slice(sample.as_base_slice());
        self.is_sample.extend_from_slice(&[true; D_EF]);
        sample
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Default)]
pub struct Preflight<TS: FiatShamirTranscript> {
    pub transcript: Transcript<TS>,
    pub proof_shape: ProofShapePreflight,
    pub gkr: GkrPreflight,
    pub batch_constraint: BatchConstraintPreflight,
    pub stacking: StackingPreflight,
    pub whir: WhirPreflight,
}

impl<TS: FiatShamirTranscript> Preflight<TS> {
    fn new(sponge: TS) -> Self {
        Self {
            transcript: Transcript::new(sponge),
            proof_shape: Default::default(),
            gkr: Default::default(),
            batch_constraint: Default::default(),
            stacking: Default::default(),
            whir: Default::default(),
        }
    }
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
pub struct WhirPreflight {}

impl BusInventory {
    fn new(b: &mut BusIndexManager) -> Self {
        Self {
            transcript_bus: TranscriptBus::new(b.new_bus_idx()),

            // Control flow buses
            gkr_module_bus: GkrModuleBus::new(b.new_bus_idx()),
            bc_module_bus: BatchConstraintModuleBus::new(b.new_bus_idx()),
            stacking_module_bus: StackingModuleBus::new(b.new_bus_idx()),
            whir_module_bus: WhirModuleBus::new(b.new_bus_idx()),

            // Data buses
            air_shape_bus: AirShapeBus::new(b.new_bus_idx()),
            air_part_shape_bus: AirPartShapeBus::new(b.new_bus_idx()),
            stacking_indices_bus: StackingIndicesBus::new(b.new_bus_idx()),
            commitments_bus: CommitmentsBus::new(b.new_bus_idx()),
            public_values_bus: PublicValuesBus::new(b.new_bus_idx()),

            // Randomness buses
            xi_randomness_bus: XiRandomnessBus::new(b.new_bus_idx()),
            constraint_randomness_bus: ConstraintSumcheckRandomnessBus::new(b.new_bus_idx()),
            stacking_randomness_bus: StackingSumcheckRandomnessBus::new(b.new_bus_idx()),

            // Claims buses
            column_claims_bus: ColumnClaimsBus::new(b.new_bus_idx()),

            // Peripheral buses
            range_checker_bus: RangeCheckerBus::new(b.new_bus_idx()),
            power_of_two_bus: PowerCheckerBus::new(b.new_bus_idx()),
            // Stacking module internal buses
        }
    }
}

impl BusInventory {
    pub fn air_part_shape_bus(&self) -> AirPartShapeBus {
        self.air_part_shape_bus
    }
}

pub struct VerifierCircuit<TS: FiatShamirTranscript> {
    modules: Vec<Box<dyn AirModule<TS>>>,
    range_checker: Arc<RangeCheckerAir<8>>,
    pow_2_checker: Arc<PowerCheckerAir<2, 32>>,
}

impl<TS: FiatShamirTranscript> VerifierCircuit<TS> {
    pub fn new(child_mvk: Arc<MultiStarkVerifyingKeyV2>) -> Self {
        let mut b = BusIndexManager::new();
        let bus_inventory = BusInventory::new(&mut b);

        let range_checker = Arc::new(RangeCheckerAir::<8>::new(bus_inventory.range_checker_bus));
        let pow_2_checker = Arc::new(PowerCheckerAir::<2, 32>::new(
            bus_inventory.power_of_two_bus,
            bus_inventory.range_checker_bus,
        ));

        let transcript_module = TranscriptModule::new(child_mvk.clone(), bus_inventory.clone());
        let proof_shape_module = ProofShapeModule::new(child_mvk.clone(), bus_inventory.clone());
        let gkr_module = GkrModule::new(child_mvk.clone(), &mut b, bus_inventory.clone());
        let batch_constraint_module =
            BatchConstraintModule::new(child_mvk.clone(), bus_inventory.clone());
        let stacking_module = StackingModule::new(child_mvk.clone(), &mut b, bus_inventory.clone());
        let whir_module = WhirModule::new(child_mvk.clone(), bus_inventory.clone());

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
            range_checker,
            pow_2_checker,
        }
    }

    pub fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let mut airs = vec![];
        for module in &self.modules {
            airs.extend(module.airs());
        }
        airs.push(self.range_checker.clone());
        airs.push(self.pow_2_checker.clone());
        airs
    }

    pub fn run_preflight(&self, sponge: TS, proof: &Proof) -> Preflight<TS> {
        let mut preflight = Preflight::<TS>::new(sponge);
        for module in self.modules.iter() {
            module.run_preflight(proof, &mut preflight);
        }
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
        proof_inputs.push(self.range_checker.generate_proof_input());
        proof_inputs.push(self.pow_2_checker.generate_proof_input());
        proof_inputs
    }
}
