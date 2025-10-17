use std::sync::Arc;

use itertools::izip;
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{Proof, StackingProof},
};

use crate::{
    stacking::{
        claims::{StackingClaimsAir, StackingClaimsTraceGenerator},
        opening::{OpeningClaimsAir, OpeningClaimsTraceGenerator},
        sumcheck::{SumcheckRoundsAir, SumcheckRoundsTraceGenerator},
        univariate::{UnivariateRoundAir, UnivariateRoundTraceGenerator},
    },
    system::{AirModule, BusInventory, Preflight, StackingPreflight},
};

pub mod claims;
pub mod opening;
pub mod sumcheck;
pub mod univariate;

pub struct StackingModule {
    bus_inventory: BusInventory,
}

impl StackingModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        StackingModule { bus_inventory }
    }
}

impl<TS: FiatShamirTranscript> AirModule<TS> for StackingModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let opening_air = OpeningClaimsAir {
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            column_claims_bus: self.bus_inventory.column_claims_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.bus_inventory.stacking_tidx_bus,
            claim_coefficients_bus: self.bus_inventory.claim_coefficients_bus,
            sumcheck_claims_bus: self.bus_inventory.sumcheck_claims_bus,
            eq_kernel_lookup_bus: self.bus_inventory.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.bus_inventory.eq_bits_lookup_bus,
        };
        let univariate_round_air = UnivariateRoundAir {
            constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            stacking_randomness_bus: self.bus_inventory.stacking_randomness_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.bus_inventory.stacking_tidx_bus,
            sumcheck_claims_bus: self.bus_inventory.sumcheck_claims_bus,
            eq_kernel_lookup_bus: self.bus_inventory.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.bus_inventory.eq_bits_lookup_bus,
        };
        let sumcheck_rounds_air = SumcheckRoundsAir {
            constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            stacking_randomness_bus: self.bus_inventory.stacking_randomness_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.bus_inventory.stacking_tidx_bus,
            sumcheck_claims_bus: self.bus_inventory.sumcheck_claims_bus,
            eq_kernel_lookup_bus: self.bus_inventory.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.bus_inventory.eq_bits_lookup_bus,
        };
        let stacking_claims_air = StackingClaimsAir {
            stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
            whir_module_bus: self.bus_inventory.whir_module_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.bus_inventory.stacking_tidx_bus,
            claim_coefficients_bus: self.bus_inventory.claim_coefficients_bus,
            sumcheck_claims_bus: self.bus_inventory.sumcheck_claims_bus,
        };
        vec![
            Arc::new(opening_air),
            Arc::new(univariate_round_air),
            Arc::new(sumcheck_rounds_air),
            Arc::new(stacking_claims_air),
        ]
    }

    fn run_preflight(
        &self,
        _vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight<TS>,
    ) {
        let mut sumcheck_rnd = vec![];
        let mut intermediate_tidx = [0; 3];

        let ts = &mut preflight.transcript;
        let StackingProof {
            univariate_round_coeffs,
            sumcheck_round_polys,
            stacking_openings,
        } = &proof.stacking_proof;

        let lambda = ts.sample_ext();
        intermediate_tidx[0] = ts.len();

        for coef in univariate_round_coeffs {
            ts.observe_ext(*coef);
        }
        let u0 = ts.sample_ext();
        let univariate_poly_rand_eval = izip!(univariate_round_coeffs, u0.powers())
            .map(|(&coef, pow)| coef * pow)
            .sum();
        sumcheck_rnd.push(u0);
        intermediate_tidx[1] = ts.len();

        for poly in sumcheck_round_polys {
            for eval in poly {
                ts.observe_ext(*eval);
            }
            let ui = ts.sample_ext();
            sumcheck_rnd.push(ui);
        }
        intermediate_tidx[2] = ts.len();

        for matrix_openings in stacking_openings {
            for col_opening in matrix_openings {
                ts.observe_ext(*col_opening);
            }
        }

        let stacking_batching_challenge = ts.sample_ext();

        preflight.stacking = StackingPreflight {
            intermediate_tidx,
            post_tidx: ts.len(),
            univariate_poly_rand_eval,
            stacking_batching_challenge,
            lambda,
            sumcheck_rnd,
        };
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight<TS>,
    ) -> Vec<AirProofRawInput<F>> {
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(OpeningClaimsTraceGenerator::generate_trace(
                    vk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(UnivariateRoundTraceGenerator::generate_trace(
                    vk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(SumcheckRoundsTraceGenerator::generate_trace(
                    vk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(StackingClaimsTraceGenerator::generate_trace(
                    vk, proof, preflight,
                ))),
                public_values: vec![],
            },
        ]
    }
}
