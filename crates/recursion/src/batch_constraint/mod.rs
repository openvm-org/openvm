use core::iter::zip;
use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{
    AirRef,
    air_builders::symbolic::{SymbolicExpressionNode, symbolic_variable::Entry},
    keygen::types::TraceWidth,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, Digest, EF, F, StarkEngineV2,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::eval_eq_uni_at_one,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::{BatchConstraintProof, Proof},
    prover::{
        AirProvingContextV2, ColMajorMatrix, CpuBackendV2, TraceCommitterV2,
        stacked_pcs::StackedPcsData,
    },
};

use crate::{
    batch_constraint::{
        bus::{
            BatchConstraintConductorBus, ConstraintsFoldingBus, Eq3bBus, EqMleBus, EqSharpUniBus,
            EqZeroNBus, ExpressionClaimBus, InteractionsFoldingBus, SumcheckClaimBus,
            SymbolicExpressionBus,
        },
        eq_airs::{Eq3bAir, EqMleAir, EqNsAir, EqSharpUniAir, EqSharpUniReceiverAir, EqUniAir},
        expr_eval::{
            ColumnClaimAir, ConstraintsFoldingAir, InteractionsFoldingAir, SymbolicExpressionAir,
        },
        expression_claim::ExpressionClaimAir,
        fractions_folder::FractionsFolderAir,
        sumcheck::{MultilinearSumcheckAir, UnivariateSumcheckAir},
    },
    bus::{
        AirPartShapeBus, AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus,
        ConstraintSumcheckRandomnessBus, HyperdimBus, PublicValuesBus, StackingModuleBus,
        TranscriptBus, XiRandomnessBus,
    },
    system::{
        AirModule, BatchConstraintPreflight, BusIndexManager, BusInventory, GlobalCtxCpu,
        Preflight, TraceGenModule,
    },
};

pub mod bus;
pub mod eq_airs;
pub mod expr_eval;
pub mod expression_claim;
pub mod fractions_folder;
pub mod sumcheck;

/// AIR index within the BatchConstraintModule
pub(crate) const LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX: usize = 9;

pub struct BatchConstraintModule {
    transcript_bus: TranscriptBus,
    constraint_sumcheck_randomness_bus: ConstraintSumcheckRandomnessBus,
    xi_randomness_bus: XiRandomnessBus,
    gkr_claim_bus: BatchConstraintModuleBus,
    stacking_module_bus: StackingModuleBus,
    column_opening_bus: ColumnClaimsBus,
    air_shape_bus: AirShapeBus,
    air_part_shape_bus: AirPartShapeBus,
    hyperdim_bus: HyperdimBus,
    public_values_bus: PublicValuesBus,

    batch_constraint_conductor_bus: BatchConstraintConductorBus,
    sumcheck_bus: SumcheckClaimBus,

    zero_n_bus: EqZeroNBus,
    eq_mle_bus: EqMleBus,
    eq_sharp_uni_bus: EqSharpUniBus,
    eq_3b_bus: Eq3bBus,

    symbolic_expression_bus: SymbolicExpressionBus,
    expression_claim_bus: ExpressionClaimBus,
    interactions_folding_bus: InteractionsFoldingBus,
    constraints_folding_bus: ConstraintsFoldingBus,

    l_skip: usize,
    max_constraint_degree: usize,
    widths: Vec<TraceWidth>,

    max_num_proofs: usize,
}

impl BatchConstraintModule {
    pub fn new(
        child_vk: &MultiStarkVerifyingKeyV2,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        max_num_proofs: usize,
    ) -> Self {
        let l_skip = child_vk.inner.params.l_skip;
        let max_constraint_degree = child_vk.inner.max_constraint_degree;
        let widths = child_vk
            .inner
            .per_air
            .iter()
            .map(|vk| vk.params.width.clone())
            .collect_vec();
        BatchConstraintModule {
            transcript_bus: bus_inventory.transcript_bus,
            constraint_sumcheck_randomness_bus: bus_inventory.constraint_randomness_bus,
            xi_randomness_bus: bus_inventory.xi_randomness_bus,
            gkr_claim_bus: bus_inventory.bc_module_bus,
            stacking_module_bus: bus_inventory.stacking_module_bus,
            column_opening_bus: bus_inventory.column_claims_bus,
            air_shape_bus: bus_inventory.air_shape_bus,
            air_part_shape_bus: bus_inventory.air_part_shape_bus,
            hyperdim_bus: bus_inventory.hyperdim_bus,
            public_values_bus: bus_inventory.public_values_bus,
            batch_constraint_conductor_bus: BatchConstraintConductorBus::new(b.new_bus_idx()),
            sumcheck_bus: SumcheckClaimBus::new(b.new_bus_idx()),
            zero_n_bus: EqZeroNBus::new(b.new_bus_idx()),
            eq_mle_bus: EqMleBus::new(b.new_bus_idx()),
            eq_sharp_uni_bus: EqSharpUniBus::new(b.new_bus_idx()),
            eq_3b_bus: Eq3bBus::new(b.new_bus_idx()),
            symbolic_expression_bus: SymbolicExpressionBus::new(b.new_bus_idx()),
            expression_claim_bus: ExpressionClaimBus::new(b.new_bus_idx()),
            interactions_folding_bus: InteractionsFoldingBus::new(b.new_bus_idx()),
            constraints_folding_bus: ConstraintsFoldingBus::new(b.new_bus_idx()),
            l_skip,
            max_constraint_degree,
            widths,
            max_num_proofs,
        }
    }

    pub fn run_preflight<TS>(&self, proof: &Proof, preflight: &mut Preflight, ts: &mut TS)
    where
        TS: FiatShamirTranscript + TranscriptHistory,
    {
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
        let lambda_tidx = ts.len();
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
            let width = &self.widths[*air_id];

            for col_idx in 0..width.common_main {
                let (col_opening, rot_opening) = column_openings[sort_idx][0][col_idx];
                ts.observe_ext(col_opening);
                ts.observe_ext(rot_opening);
            }
        }

        for (sort_idx, (air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let width = &self.widths[*air_id];
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
            lambda_tidx,
            tidx_before_univariate,
            tidx_before_multilinear,
            tidx_before_column_openings,
            post_tidx: ts.len(),
            xi,
            sumcheck_rnd,
        }
    }
}

impl AirModule for BatchConstraintModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let l_skip = self.l_skip;

        let fraction_folder_air = FractionsFolderAir {
            transcript_bus: self.transcript_bus,
            sumcheck_bus: self.sumcheck_bus,
            gkr_claim_bus: self.gkr_claim_bus,
        };
        let sumcheck_uni_air = UnivariateSumcheckAir {
            l_skip,
            univariate_deg: (self.max_constraint_degree + 1) * ((1 << l_skip) - 1),
            claim_bus: self.sumcheck_bus,
            transcript_bus: self.transcript_bus,
            randomness_bus: self.constraint_sumcheck_randomness_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
        };
        let sumcheck_lin_air = MultilinearSumcheckAir {
            max_constraint_degree: self.max_constraint_degree,
            claim_bus: self.sumcheck_bus,
            transcript_bus: self.transcript_bus,
            randomness_bus: self.constraint_sumcheck_randomness_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
        };
        let eq_ns_air = EqNsAir {
            zero_n_bus: self.zero_n_bus,
            xi_bus: self.xi_randomness_bus,
            r_xi_bus: self.batch_constraint_conductor_bus,
            l_skip,
        };
        let eq_mle_air = EqMleAir {
            transcript_bus: self.transcript_bus,
            eq_mle_bus: self.eq_mle_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
            l_skip,
        };
        let eq_sharp_uni_air = EqSharpUniAir {
            xi_bus: self.xi_randomness_bus,
            eq_bus: self.eq_sharp_uni_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
            l_skip,
            canonical_inverse_generator: F::two_adic_generator(l_skip).inverse(),
        };
        let eq_sharp_uni_receiver_air = EqSharpUniReceiverAir {
            r_bus: self.batch_constraint_conductor_bus,
            eq_bus: self.eq_sharp_uni_bus,
            zero_n_bus: self.zero_n_bus,
            l_skip,
        };
        let eq_uni_air = EqUniAir {
            r_xi_bus: self.batch_constraint_conductor_bus,
            zero_n_bus: self.zero_n_bus,
            l_skip,
        };
        let eq_3b_air = Eq3bAir {
            eq_mle_bus: self.eq_mle_bus,
            eq_3b_bus: self.eq_3b_bus,
            l_skip,
        };
        let symbolic_expression_air = SymbolicExpressionAir {
            expr_bus: self.symbolic_expression_bus,
            claim_bus: self.expression_claim_bus,
            column_claims_bus: self.column_opening_bus,
            interactions_folding_bus: self.interactions_folding_bus,
            constraints_folding_bus: self.constraints_folding_bus,
            hyperdim_bus: self.hyperdim_bus,
            public_values_bus: self.public_values_bus,
            cnt_proofs: self.max_num_proofs,
        };
        let column_claim_air = ColumnClaimAir {
            transcript_bus: self.transcript_bus,
            column_claims_bus: self.column_opening_bus,
            air_shape_bus: self.air_shape_bus,
            air_part_shape_bus: self.air_part_shape_bus,
            hyperdim_bus: self.hyperdim_bus,
            stacking_module_bus: self.stacking_module_bus,
        };
        let expression_claim_air = ExpressionClaimAir {
            claim_bus: self.expression_claim_bus,
            transcript_bus: self.transcript_bus,
        };
        let interactions_folding_air = InteractionsFoldingAir {
            transcript_bus: self.transcript_bus,
            air_shape_bus: self.air_shape_bus,
            interaction_bus: self.interactions_folding_bus,
            expression_claim_bus: self.expression_claim_bus,
        };
        let constraints_folding_air = ConstraintsFoldingAir {
            transcript_bus: self.transcript_bus,
            constraint_bus: self.constraints_folding_bus,
            expression_claim_bus: self.expression_claim_bus,
        };
        vec![
            Arc::new(fraction_folder_air) as AirRef<_>,
            Arc::new(sumcheck_uni_air) as AirRef<_>,
            Arc::new(sumcheck_lin_air) as AirRef<_>,
            Arc::new(eq_ns_air) as AirRef<_>,
            Arc::new(eq_mle_air) as AirRef<_>,
            Arc::new(eq_sharp_uni_air) as AirRef<_>,
            Arc::new(eq_sharp_uni_receiver_air) as AirRef<_>,
            Arc::new(eq_uni_air) as AirRef<_>,
            Arc::new(eq_3b_air) as AirRef<_>,
            Arc::new(symbolic_expression_air) as AirRef<_>,
            Arc::new(column_claim_air) as AirRef<_>,
            Arc::new(expression_claim_air) as AirRef<_>,
            Arc::new(interactions_folding_air) as AirRef<_>,
            Arc::new(constraints_folding_air) as AirRef<_>,
        ]
    }
}

pub(in crate::batch_constraint) struct BatchConstraintBlobCpu {
    // Per proof, per air (vkey order), the evaluations. For optional AIRs without traces, the
    // innermost vec is empty.
    // TODO: Make this flatter.
    pub expr_evals: Vec<Vec<Vec<EF>>>,
}

impl BatchConstraintModule {
    fn generate_blob(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> BatchConstraintBlobCpu {
        let child_vk = &child_vk.inner;
        let params = child_vk.params;

        let mut expr_evals_per_proof = vec![];
        for (proof, preflight) in zip(proofs, preflights) {
            let rs = &preflight.batch_constraint.sumcheck_rnd;

            let (&rs_0, rs_rest) = rs.split_first().unwrap();
            let mut is_first_row_by_log_height = vec![];
            let mut is_last_row_by_log_height = vec![];

            for log_height in 0..=params.l_skip {
                let omega = F::two_adic_generator(log_height);
                is_first_row_by_log_height.push(eval_eq_uni_at_one(log_height, rs_0));
                is_last_row_by_log_height.push(eval_eq_uni_at_one(log_height, rs_0 * omega));
            }
            for (i, &r) in rs_rest.iter().enumerate() {
                is_first_row_by_log_height
                    .push(is_first_row_by_log_height[params.l_skip + i] * (EF::ONE - r));
                is_last_row_by_log_height.push(is_last_row_by_log_height[params.l_skip + i] * r);
            }

            let mut expr_evals_per_air = vec![];
            for (air_idx, vk) in child_vk.per_air.iter().enumerate() {
                if proof.trace_vdata[air_idx].is_none() {
                    continue;
                }

                let openings = &proof.batch_constraint_proof.column_openings;
                let (sorted_idx, ref vdata) = preflight.proof_shape.sorted_trace_vdata[air_idx];

                let constraints = &vk.symbolic_constraints.constraints;
                let mut expr_evals = vec![EF::ZERO; constraints.nodes.len()];

                for (node_idx, node) in constraints.nodes.iter().enumerate() {
                    match node {
                        SymbolicExpressionNode::Variable(var) => match var.entry {
                            Entry::Preprocessed { offset } => {
                                expr_evals[node_idx] = match offset {
                                    0 => openings[sorted_idx][1][var.index].0,
                                    1 => openings[sorted_idx][1][var.index].1,
                                    _ => unreachable!(),
                                };
                            }
                            Entry::Main { part_index, offset } => {
                                let part = if part_index == 0 {
                                    0
                                } else {
                                    part_index + vk.preprocessed_data.is_some() as usize
                                };
                                expr_evals[node_idx] = match offset {
                                    0 => openings[sorted_idx][part][var.index].0,
                                    1 => openings[sorted_idx][part][var.index].1,
                                    _ => unreachable!(),
                                };
                            }
                            Entry::Permutation { .. } => unreachable!(),
                            Entry::Public => {
                                expr_evals[node_idx] =
                                    EF::from_base(proof.public_values[air_idx][var.index]);
                            }
                            Entry::Challenge => unreachable!(),
                            Entry::Exposed => unreachable!(),
                        },
                        SymbolicExpressionNode::IsFirstRow => {
                            expr_evals[node_idx] = is_first_row_by_log_height[vdata.log_height];
                        }
                        SymbolicExpressionNode::IsLastRow => {
                            expr_evals[node_idx] = is_last_row_by_log_height[vdata.log_height];
                        }
                        SymbolicExpressionNode::IsTransition => {
                            expr_evals[node_idx] =
                                EF::ONE - is_last_row_by_log_height[vdata.log_height];
                        }
                        SymbolicExpressionNode::Constant(val) => {
                            expr_evals[node_idx] = EF::from_base(*val);
                        }
                        SymbolicExpressionNode::Add {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            debug_assert!(*left_idx < node_idx);
                            debug_assert!(*right_idx < node_idx);
                            expr_evals[node_idx] = expr_evals[*left_idx] + expr_evals[*right_idx];
                        }
                        SymbolicExpressionNode::Sub {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            debug_assert!(*left_idx < node_idx);
                            debug_assert!(*right_idx < node_idx);
                            expr_evals[node_idx] = expr_evals[*left_idx] - expr_evals[*right_idx];
                        }
                        SymbolicExpressionNode::Neg {
                            idx,
                            degree_multiple: _,
                        } => {
                            debug_assert!(*idx < node_idx);
                            expr_evals[node_idx] = -expr_evals[*idx];
                        }
                        SymbolicExpressionNode::Mul {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            debug_assert!(*left_idx < node_idx);
                            debug_assert!(*right_idx < node_idx);
                            expr_evals[node_idx] = expr_evals[*left_idx] * expr_evals[*right_idx];
                        }
                    };
                }
                expr_evals_per_air.push(expr_evals);
            }
            expr_evals_per_proof.push(expr_evals_per_air);
        }
        BatchConstraintBlobCpu {
            expr_evals: expr_evals_per_proof,
        }
    }
}

impl TraceGenModule<GlobalCtxCpu, CpuBackendV2> for BatchConstraintModule {
    /// **Note**: This generates all common main traces but leaves the cached trace for
    /// `SymbolicExpressionAir` unset. The cached trace must be loaded **after** calling this
    /// function.
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        let blob = self.generate_blob(child_vk, proofs, preflights);

        let common = expr_eval::generate_symbolic_expr_common_trace(
            child_vk,
            proofs,
            preflights,
            self.max_num_proofs,
            &blob,
        );
        let transpose =
            |trace| AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace));
        // NOTE: this leaves cached = vec![]. The cached trace must be set **after**.
        let symbolic_expr_ctx = transpose(common);
        let (uni_trace, uni_receiver_trace) =
            eq_airs::generate_eq_sharp_uni_traces(child_vk, proofs, preflights);
        let mle_blob = eq_airs::generate_eq_mle_blob(child_vk, preflights);
        vec![
            transpose(fractions_folder::generate_trace(
                child_vk, proofs, preflights,
            )),
            transpose(sumcheck::univariate::generate_trace(
                child_vk, proofs, preflights,
            )),
            transpose(sumcheck::multilinear::generate_trace(
                child_vk, proofs, preflights,
            )),
            transpose(eq_airs::generate_eq_ns_trace(child_vk, proofs, preflights)),
            transpose(eq_airs::generate_eq_mle_trace(
                child_vk, &mle_blob, preflights,
            )),
            transpose(uni_trace),
            transpose(uni_receiver_trace),
            transpose(eq_airs::generate_eq_uni_trace(child_vk, proofs, preflights)),
            transpose(eq_airs::generate_eq_3b_trace(
                child_vk, &mle_blob, preflights,
            )),
            symbolic_expr_ctx,
            transpose(expr_eval::generate_column_claim_trace(
                child_vk, proofs, preflights,
            )),
            transpose(expression_claim::generate_trace(
                child_vk, proofs, preflights,
            )),
            transpose(expr_eval::generate_interactions_folding_trace(
                child_vk, &blob, preflights,
            )),
            transpose(expr_eval::generate_constraints_folding_trace(
                child_vk, &blob, preflights,
            )),
        ]
    }
}

impl BatchConstraintModule {
    /// Generates and then commits to the cache trace for `SymbolicExpressionAir`. Returns the
    /// committed PCS data.
    pub fn commit_child_vk(
        &self,
        engine: &BabyBearPoseidon2CpuEngineV2,
        child_vk: &MultiStarkVerifyingKeyV2,
    ) -> (Digest, StackedPcsData<F, Digest>) {
        let cached_trace = expr_eval::generate_symbolic_expr_cached_trace(child_vk);
        engine
            .device()
            .commit(&[&ColMajorMatrix::from_row_major(&cached_trace)])
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_tracegen {
    use cuda_backend_v2::{
        BabyBearPoseidon2GpuEngineV2, GpuBackendV2, GpuDeviceV2, stacked_pcs::StackedPcsDataGpu,
        transport_matrix_h2d_col_major,
    };
    use itertools::Itertools;
    use p3_matrix::Matrix;

    use super::*;
    use crate::cuda::{
        GlobalCtxGpu, preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu,
    };

    impl TraceGenModule<GlobalCtxGpu, GpuBackendV2> for BatchConstraintModule {
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

    impl BatchConstraintModule {
        /// Generates and then commits to the cache trace for `SymbolicExpressionAir`. Returns the
        /// committed PCS data.
        pub fn commit_child_vk_gpu(
            &self,
            engine: &BabyBearPoseidon2GpuEngineV2,
            child_vk: &MultiStarkVerifyingKeyV2,
        ) -> (Digest, StackedPcsDataGpu<F, Digest>) {
            let cached_trace = expr_eval::generate_symbolic_expr_cached_trace(child_vk);
            let cached_trace = ColMajorMatrix::from_row_major(&cached_trace);
            engine
                .device()
                .commit(&[&transport_matrix_h2d_col_major(&cached_trace).unwrap()])
        }
    }
}
