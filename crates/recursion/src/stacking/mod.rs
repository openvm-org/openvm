use std::sync::Arc;

use itertools::izip;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::{Proof, StackingProof},
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
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
    system::{
        AirModule, BusIndexManager, BusInventory, GlobalCtxCpu, Preflight, StackingPreflight,
        TraceGenModule,
    },
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

    l_skip: usize,
    n_stack: usize,
    stacking_index_mult: usize,
}

impl StackingModule {
    pub fn new(
        child_vk: &MultiStarkVerifyingKeyV2,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Self {
        Self {
            bus_inventory,
            stacking_tidx_bus: StackingModuleTidxBus::new(b.new_bus_idx()),
            claim_coefficients_bus: ClaimCoefficientsBus::new(b.new_bus_idx()),
            sumcheck_claims_bus: SumcheckClaimsBus::new(b.new_bus_idx()),
            eq_rand_values_bus: EqRandValuesLookupBus::new(b.new_bus_idx()),
            eq_base_bus: EqBaseBus::new(b.new_bus_idx()),
            eq_bits_internal_bus: EqBitsInternalBus::new(b.new_bus_idx()),
            eq_kernel_lookup_bus: EqKernelLookupBus::new(b.new_bus_idx()),
            eq_bits_lookup_bus: EqBitsLookupBus::new(b.new_bus_idx()),
            l_skip: child_vk.inner.params.l_skip,
            n_stack: child_vk.inner.params.n_stack,
            stacking_index_mult: child_vk.inner.params.num_whir_queries
                << child_vk.inner.params.k_whir,
        }
    }

    pub fn run_preflight<TS>(&self, proof: &Proof, preflight: &mut Preflight, ts: &mut TS)
    where
        TS: FiatShamirTranscript + TranscriptHistory,
    {
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
}

impl AirModule for StackingModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let opening_air = OpeningClaimsAir {
            lifted_heights_bus: self.bus_inventory.lifted_heights_bus,
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            column_claims_bus: self.bus_inventory.column_claims_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            claim_coefficients_bus: self.claim_coefficients_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            l_skip: self.l_skip,
            n_stack: self.n_stack,
        };
        let univariate_round_air = UnivariateRoundAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            l_skip: self.l_skip,
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
            l_skip: self.l_skip,
        };
        let stacking_claims_air = StackingClaimsAir {
            stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
            whir_module_bus: self.bus_inventory.whir_module_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            claim_coefficients_bus: self.claim_coefficients_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            stacking_index_mult: self.stacking_index_mult,
        };
        let eq_base_air = EqBaseAir {
            constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            eq_base_bus: self.eq_base_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            eq_neg_base_rand_bus: self.bus_inventory.eq_neg_base_rand_bus,
            eq_neg_result_bus: self.bus_inventory.eq_neg_result_bus,
            l_skip: self.l_skip,
        };
        let eq_bits_air = EqBitsAir {
            eq_bits_internal_bus: self.eq_bits_internal_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            n_stack: self.n_stack,
            l_skip: self.l_skip,
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
}

impl TraceGenModule<GlobalCtxCpu, CpuBackendV2> for StackingModule {
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        // TODO: parallelize
        let traces = [
            OpeningClaimsTraceGenerator::generate_trace(child_vk, proofs, preflights),
            UnivariateRoundTraceGenerator::generate_trace(child_vk, proofs, preflights),
            SumcheckRoundsTraceGenerator::generate_trace(child_vk, proofs, preflights),
            StackingClaimsTraceGenerator::generate_trace(child_vk, proofs, preflights),
            EqBaseTraceGenerator::generate_trace(child_vk, proofs, preflights),
            EqBitsTraceGenerator::generate_trace(child_vk, proofs, preflights),
        ];
        traces
            .into_iter()
            .map(|trace| AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace)))
            .collect()
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use cuda_backend_v2::{GpuBackendV2, transport_matrix_h2d_col_major};
    use itertools::Itertools;

    use super::*;
    use crate::cuda::{
        GlobalCtxGpu, preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu,
    };

    impl TraceGenModule<GlobalCtxGpu, GpuBackendV2> for StackingModule {
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
        ) -> Vec<AirProvingContextV2<GpuBackendV2>> {
            // default hybrid implementation:
            let ctxs_cpu = TraceGenModule::<GlobalCtxCpu, CpuBackendV2>::generate_proving_ctxs(
                self,
                &child_vk.cpu,
                &proofs.iter().map(|proof| proof.cpu.clone()).collect_vec(),
                &preflights
                    .iter()
                    .map(|preflight| preflight.cpu.clone())
                    .collect_vec(),
            );
            ctxs_cpu
                .into_iter()
                .map(|ctx| {
                    assert!(ctx.cached_mains.is_empty());
                    AirProvingContextV2::simple_no_pis(
                        transport_matrix_h2d_col_major(&ctx.common_main).unwrap(),
                    )
                })
                .collect()
        }
    }
}
