use core::{cmp::Reverse, iter::zip};
use std::sync::Arc;

use itertools::{izip, multizip};
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, dummy_airs::fib_air::trace,
};
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    proof::{BatchConstraintProof, Proof},
};

use crate::{
    batch_constraint::{
        columns::DummyPerColumnAir, selector::BatchConstraintSelectorAir,
        sumcheck::BatchConstraintSumcheckAir,
    },
    system::{AirModule, BatchConstraintPreflight, BusInventory, Preflight},
};

mod columns;
mod selector;
mod sumcheck;

pub struct BatchConstraintModule {
    bus_inventory: BusInventory,
}

impl BatchConstraintModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        BatchConstraintModule { bus_inventory }
    }
}

impl AirModule for BatchConstraintModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        // TODO: GkrRandomnessBus
        // TODO: ColumnClaimsBus
        let sumcheck_air = BatchConstraintSumcheckAir {
            bc_module_bus: self.bus_inventory.bc_module_bus,
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            initial_zc_randomness_bus: self.bus_inventory.initial_zerocheck_randomness_bus,
            batch_constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
        };
        let selector_air = BatchConstraintSelectorAir {
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_part_shape_bus: self.bus_inventory.air_part_shape_bus,
        };
        let per_column_air = DummyPerColumnAir {
            column_claims_bus: self.bus_inventory.column_claims_bus,
        };
        vec![
            Arc::new(sumcheck_air) as AirRef<_>,
            Arc::new(selector_air) as AirRef<_>,
            Arc::new(per_column_air) as AirRef<_>,
        ]
    }

    fn run_preflight(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight,
    ) {
        let ts = &mut preflight.transcript;
        let BatchConstraintProof {
            numerator_term_per_air,
            denominator_term_per_air,
            univariate_round_coeffs,
            sumcheck_round_polys,
            column_openings,
        } = &proof.batch_constraint_proof;

        // Constraint batching
        let _lambda = ts.sample_ext();

        for (sum_claim_p, sum_claim_q) in zip(numerator_term_per_air, denominator_term_per_air) {
            ts.observe_ext(*sum_claim_p);
            ts.observe_ext(*sum_claim_q);
        }
        let _mu = ts.sample_ext();

        // univariate round
        for coef in univariate_round_coeffs {
            ts.observe_ext(*coef);
        }
        let _ = ts.sample_ext();

        for poly in sumcheck_round_polys {
            for eval in poly {
                ts.observe_ext(*eval)
            }
            let _ = ts.sample_ext();
        }

        let _r0 = ts.sample_ext();

        for polys in sumcheck_round_polys {
            for eval in polys {
                ts.observe_ext(*eval);
            }
            let _ri = ts.sample_ext();
        }

        // Common main
        for (sort_idx, (air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let width = &vk.inner.per_air[*air_id].params.width;

            for col_idx in 0..width.common_main {
                let (col_opening, rot_opening) = column_openings[sort_idx][0][col_idx];
                ts.observe_ext(col_opening);
                ts.observe_ext(rot_opening);
            }
        }

        for (sort_idx, (air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let width = &vk.inner.per_air[*air_id].params.width;
            let widths = width.preprocessed.iter().chain(width.cached_mains.iter());

            for (i, w) in widths.enumerate() {
                for col_idx in 0..*w {
                    let (col_opening, rot_opening) = column_openings[sort_idx][i + 1][col_idx];
                    ts.observe_ext(col_opening);
                    ts.observe_ext(rot_opening);
                }
            }
        }

        preflight.batch_constraint = BatchConstraintPreflight {
            post_tidx: ts.len(),
        }
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_trace(proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(selector::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(columns::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
        ]
    }
}
