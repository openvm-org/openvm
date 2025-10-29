use std::sync::Arc;

use itertools::izip;
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::{Proof, StackingProof},
};

use crate::{
    stacking::{
        bus::*,
        claims::{StackingClaimsAir, StackingClaimsTraceGenerator},
        eq_base::{EqBaseAir, EqBaseTraceGenerator},
        eq_bits::{EqBitsAir, EqBitsTraceGenerator},
        opening::{OpeningClaimsAir, OpeningClaimsTraceGenerator},
        sumcheck::{SumcheckRoundsAir, SumcheckRoundsTraceGenerator},
        univariate::{UnivariateRoundAir, UnivariateRoundTraceGenerator},
    },
    system::{AirModule, BusIndexManager, BusInventory, Preflight, StackingPreflight},
};

mod bus;
pub mod claims;
pub mod eq_base;
pub mod eq_bits;
pub mod opening;
pub mod sumcheck;
pub mod univariate;
mod utils;

pub struct StackingModule {
    mvk: Arc<MultiStarkVerifyingKeyV2>,
    bus_inventory: BusInventory,

    // Internal buses
    stacking_tidx_bus: StackingModuleTidxBus,
    claim_coefficients_bus: ClaimCoefficientsBus,
    sumcheck_claims_bus: SumcheckClaimsBus,
    eq_rand_values_bus: EqRandValuesLookupBus,
    eq_base_bus: EqBaseBus,
    eq_bits_internal_bus: EqBitsInternalBus,
    eq_kernel_lookup_bus: EqKernelLookupBus,
    eq_bits_lookup_bus: EqBitsLookupBus,
}

impl StackingModule {
    pub fn new(
        mvk: Arc<MultiStarkVerifyingKeyV2>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Self {
        Self {
            mvk,
            bus_inventory,
            stacking_tidx_bus: StackingModuleTidxBus::new(b.new_bus_idx()),
            claim_coefficients_bus: ClaimCoefficientsBus::new(b.new_bus_idx()),
            sumcheck_claims_bus: SumcheckClaimsBus::new(b.new_bus_idx()),
            eq_rand_values_bus: EqRandValuesLookupBus::new(b.new_bus_idx()),
            eq_base_bus: EqBaseBus::new(b.new_bus_idx()),
            eq_bits_internal_bus: EqBitsInternalBus::new(b.new_bus_idx()),
            eq_kernel_lookup_bus: EqKernelLookupBus::new(b.new_bus_idx()),
            eq_bits_lookup_bus: EqBitsLookupBus::new(b.new_bus_idx()),
        }
    }
}

impl<TS: FiatShamirTranscript + TranscriptHistory> AirModule<TS> for StackingModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let opening_air = OpeningClaimsAir {
            air_heights_bus: self.bus_inventory.air_heights_bus,
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            column_claims_bus: self.bus_inventory.column_claims_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            claim_coefficients_bus: self.claim_coefficients_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            l_skip: self.mvk.inner.params.l_skip,
            n_stack: self.mvk.inner.params.n_stack,
        };
        let univariate_round_air = UnivariateRoundAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            l_skip: self.mvk.inner.params.l_skip,
        };
        let sumcheck_rounds_air = SumcheckRoundsAir {
            constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            eq_base_bus: self.eq_base_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            l_skip: self.mvk.inner.params.l_skip,
        };
        let stacking_claims_air = StackingClaimsAir {
            stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
            whir_module_bus: self.bus_inventory.whir_module_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            claim_coefficients_bus: self.claim_coefficients_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
        };
        let eq_base_air = EqBaseAir {
            constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            eq_base_bus: self.eq_base_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            l_skip: self.mvk.inner.params.l_skip,
        };
        let eq_bits_air = EqBitsAir {
            eq_bits_internal_bus: self.eq_bits_internal_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            n_stack: self.mvk.inner.params.n_stack,
            l_skip: self.mvk.inner.params.l_skip,
        };
        vec![
            Arc::new(opening_air),
            Arc::new(univariate_round_air),
            Arc::new(sumcheck_rounds_air),
            Arc::new(stacking_claims_air),
            Arc::new(eq_base_air),
            Arc::new(eq_bits_air),
        ]
    }

    fn run_preflight(&self, proof: &Proof, preflight: &mut Preflight, ts: &mut TS) {
        let mut sumcheck_rnd = vec![];
        let mut intermediate_tidx = [0; 3];

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
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> Vec<AirProofRawInput<F>> {
        // TODO: support multiple proofs
        debug_assert_eq!(proofs.len(), 1);
        debug_assert_eq!(preflights.len(), 1);
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(OpeningClaimsTraceGenerator::generate_trace(
                    &self.mvk,
                    &proofs[0],
                    &preflights[0],
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(UnivariateRoundTraceGenerator::generate_trace(
                    &self.mvk,
                    &proofs[0],
                    &preflights[0],
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(SumcheckRoundsTraceGenerator::generate_trace(
                    &self.mvk,
                    &proofs[0],
                    &preflights[0],
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(StackingClaimsTraceGenerator::generate_trace(
                    &self.mvk,
                    &proofs[0],
                    &preflights[0],
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(EqBaseTraceGenerator::generate_trace(
                    &self.mvk,
                    &proofs[0],
                    &preflights[0],
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(EqBitsTraceGenerator::generate_trace(
                    &self.mvk,
                    &preflights[0],
                ))),
                public_values: vec![],
            },
        ]
    }
}
