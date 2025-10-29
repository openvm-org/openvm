use core::iter::zip;
use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::{BatchConstraintProof, Proof},
};

use crate::{
    batch_constraint::{
        bus::{
            Eq3bBus, EqMleBus, EqSharpUniBus, EqZeroNBus, ExpressionClaimBus,
            InteractionsFoldingBus, SumcheckClaimBus, SymbolicExpressionBus,
        },
        eq_airs::{Eq3bAir, EqMleAir, EqNsAir, EqSharpUniAir, EqSharpUniReceiverAir},
        expr_eval::{
            ColumnClaimAir, ExpressionClaimAir, InteractionsFoldingAir, SymbolicExpressionAir,
        },
        fractions_folder::FractionFolderAir,
        sumcheck::{MultilinearSumcheckAir, UnivariateSumcheckAir},
    },
    bus::{
        AirPartShapeBus, AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus,
        ConstraintSumcheckRandomnessBus, StackingModuleBus, TranscriptBus, XiRandomnessBus,
    },
    system::{AirModule, BatchConstraintPreflight, BusIndexManager, BusInventory, Preflight},
};

pub mod bus;
pub mod eq_airs;
pub mod expr_eval;
pub mod fractions_folder;
pub mod sumcheck;

pub struct BatchConstraintModule {
    mvk: Arc<MultiStarkVerifyingKeyV2>,

    transcript_bus: TranscriptBus,
    constraint_sumcheck_randomness_bus: ConstraintSumcheckRandomnessBus,
    xi_randomness_bus: XiRandomnessBus,
    gkr_claim_bus: BatchConstraintModuleBus,
    stacking_module_bus: StackingModuleBus,
    column_opening_bus: ColumnClaimsBus,
    air_shape_bus: AirShapeBus,
    air_part_shape_bus: AirPartShapeBus,

    sumcheck_bus: SumcheckClaimBus,

    zero_n_bus: EqZeroNBus,
    eq_mle_bus: EqMleBus,
    eq_sharp_uni_bus: EqSharpUniBus,
    eq_3b_bus: Eq3bBus,

    symbolic_expression_bus: SymbolicExpressionBus,
    expression_claim_bus: ExpressionClaimBus,
    interactions_folding_bus: InteractionsFoldingBus,
}

impl BatchConstraintModule {
    pub fn new(
        mvk: Arc<MultiStarkVerifyingKeyV2>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Self {
        BatchConstraintModule {
            mvk,
            transcript_bus: bus_inventory.transcript_bus,
            constraint_sumcheck_randomness_bus: bus_inventory.constraint_randomness_bus,
            xi_randomness_bus: bus_inventory.xi_randomness_bus,
            gkr_claim_bus: bus_inventory.bc_module_bus,
            stacking_module_bus: bus_inventory.stacking_module_bus,
            column_opening_bus: bus_inventory.column_claims_bus,
            air_shape_bus: bus_inventory.air_shape_bus,
            air_part_shape_bus: bus_inventory.air_part_shape_bus,
            sumcheck_bus: SumcheckClaimBus::new(b.new_bus_idx()),
            zero_n_bus: EqZeroNBus::new(b.new_bus_idx()),
            eq_mle_bus: EqMleBus::new(b.new_bus_idx()),
            eq_sharp_uni_bus: EqSharpUniBus::new(b.new_bus_idx()),
            eq_3b_bus: Eq3bBus::new(b.new_bus_idx()),
            symbolic_expression_bus: SymbolicExpressionBus::new(b.new_bus_idx()),
            expression_claim_bus: ExpressionClaimBus::new(b.new_bus_idx()),
            interactions_folding_bus: InteractionsFoldingBus::new(b.new_bus_idx()),
        }
    }
}

impl<TS: FiatShamirTranscript + TranscriptHistory> AirModule<TS> for BatchConstraintModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let l_skip = self.mvk.inner.params.l_skip;

        let fraction_folder_air = FractionFolderAir {
            transcript_bus: self.transcript_bus,
            sumcheck_bus: self.sumcheck_bus,
            gkr_claim_bus: self.gkr_claim_bus,
        };
        let sumcheck_uni_air = UnivariateSumcheckAir {
            univariate_deg: (self.mvk.inner.max_constraint_degree + 1) * ((1 << l_skip) - 1) + 1,
            domain_size: 1 << l_skip,
            claim_bus: self.sumcheck_bus,
            transcript_bus: self.transcript_bus,
            randomness_bus: self.constraint_sumcheck_randomness_bus,
        };
        let sumcheck_lin_air = MultilinearSumcheckAir {
            claim_bus: self.sumcheck_bus,
            transcript_bus: self.transcript_bus,
            randomness_bus: self.constraint_sumcheck_randomness_bus,
        };
        let eq_ns_air = EqNsAir {
            zero_n_bus: self.zero_n_bus,
            r_bus: self.constraint_sumcheck_randomness_bus,
        };
        let eq_mle_air = EqMleAir {
            xi_bus: self.xi_randomness_bus,
            transcript_bus: self.transcript_bus,
            eq_mle_bus: self.eq_mle_bus,
            l_skip,
        };
        let eq_sharp_uni_air = EqSharpUniAir {
            xi_bus: self.xi_randomness_bus,
            eq_bus: self.eq_sharp_uni_bus,
            l_skip,
        };
        let eq_sharp_uni_receiver_air = EqSharpUniReceiverAir {
            r_bus: self.constraint_sumcheck_randomness_bus,
            eq_bus: self.eq_sharp_uni_bus,
        };
        let eq_3b_air = Eq3bAir {
            air_shape_bus: self.air_shape_bus,
            eq_mle_bus: self.eq_mle_bus,
            eq_3b_bus: self.eq_3b_bus,
        };
        let symbolic_expression_air = SymbolicExpressionAir {
            expr_bus: self.symbolic_expression_bus,
            claim_bus: self.expression_claim_bus,
            stacking_module_bus: self.stacking_module_bus,
            column_claims_bus: self.column_opening_bus,
            interactions_folding_bus: self.interactions_folding_bus,
            cnt_proofs: 1, // TODO!
        };
        let column_claim_air = ColumnClaimAir {
            transcript_bus: self.transcript_bus,
            column_claims_bus: self.column_opening_bus,
            air_shape_bus: self.air_shape_bus,
            air_part_shape_bus: self.air_part_shape_bus,
        };
        let expression_claim_air = ExpressionClaimAir {
            claim_bus: self.expression_claim_bus,
            transcript_bus: self.transcript_bus,
        };
        let interactions_folding_air = InteractionsFoldingAir {
            transcript_bus: self.transcript_bus,
            air_shape_bus: self.air_shape_bus,
            interaction_bus: self.interactions_folding_bus,
        };
        vec![
            Arc::new(fraction_folder_air) as AirRef<_>,
            Arc::new(sumcheck_uni_air) as AirRef<_>,
            Arc::new(sumcheck_lin_air) as AirRef<_>,
            Arc::new(eq_ns_air) as AirRef<_>,
            Arc::new(eq_mle_air) as AirRef<_>,
            Arc::new(eq_sharp_uni_air) as AirRef<_>,
            Arc::new(eq_sharp_uni_receiver_air) as AirRef<_>,
            Arc::new(eq_3b_air) as AirRef<_>,
            Arc::new(symbolic_expression_air) as AirRef<_>,
            Arc::new(column_claim_air) as AirRef<_>,
            Arc::new(expression_claim_air) as AirRef<_>,
            Arc::new(interactions_folding_air) as AirRef<_>,
        ]
    }

    fn run_preflight(&self, proof: &Proof, preflight: &mut Preflight, ts: &mut TS) {
        let BatchConstraintProof {
            numerator_term_per_air,
            denominator_term_per_air,
            univariate_round_coeffs,
            sumcheck_round_polys,
            column_openings,
        } = &proof.batch_constraint_proof;

        let mut sumcheck_rnd = vec![];

        let mut xi = preflight.gkr.xi.iter().map(|(_, x)| *x).collect_vec();
        let l_skip = preflight.proof_shape.l_skip;
        let n_global = preflight.proof_shape.n_global();
        for _ in xi.len()..(l_skip + n_global) {
            xi.push(ts.sample_ext());
        }

        // Constraint batching
        let _lambda = ts.sample_ext();

        for (sum_claim_p, sum_claim_q) in zip(numerator_term_per_air, denominator_term_per_air) {
            ts.observe_ext(*sum_claim_p);
            ts.observe_ext(*sum_claim_q);
        }
        let _mu = ts.sample_ext();

        let tidx_before_univariate = ts.len();

        // univariate round
        for coef in univariate_round_coeffs {
            ts.observe_ext(*coef);
        }
        let r0 = ts.sample_ext();
        sumcheck_rnd.push(r0);

        let tidx_before_multilinear = ts.len();

        for polys in sumcheck_round_polys {
            for eval in polys {
                ts.observe_ext(*eval);
            }
            let ri = ts.sample_ext();
            sumcheck_rnd.push(ri);
        }

        let tidx_before_column_openings = ts.len();

        // Common main
        for (sort_idx, (air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let width = &self.mvk.inner.per_air[*air_id].params.width;

            for col_idx in 0..width.common_main {
                let (col_opening, rot_opening) = column_openings[sort_idx][0][col_idx];
                ts.observe_ext(col_opening);
                ts.observe_ext(rot_opening);
            }
        }

        for (sort_idx, (air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let width = &self.mvk.inner.per_air[*air_id].params.width;
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
            tidx_before_univariate,
            tidx_before_multilinear,
            tidx_before_column_openings,
            post_tidx: ts.len(),
            xi,
            sumcheck_rnd,
        }
    }

    fn generate_proof_inputs(
        &self,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(fractions_folder::generate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_univariate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_multilinear_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(eq_airs::generate_eq_ns_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(eq_airs::generate_eq_mle_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(eq_airs::generate_eq_sharp_uni_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(eq_airs::generate_eq_sharp_uni_receiver_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(eq_airs::generate_eq_3b_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(expr_eval::generate_symbolic_expression_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(expr_eval::generate_column_claim_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(expr_eval::generate_expression_claim_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(expr_eval::generate_interactions_folding_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
        ]
    }
}
