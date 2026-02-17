use core::iter::zip;
use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{
    air_builders::symbolic::{symbolic_variable::Entry, SymbolicExpressionNode},
    keygen::types::MultiStarkVerifyingKey,
    poly_common::{eval_eq_sharp_uni, eval_eq_uni, eval_eq_uni_at_one},
    proof::{column_openings_by_rot, BatchConstraintProof, Proof},
    prover::{AirProvingContext, ColMajorMatrix, CommittedTraceData, CpuBackend, TraceCommitter},
    AirRef, FiatShamirTranscript, StarkEngine, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, EF, F};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use strum::{EnumCount, EnumDiscriminants};

use crate::{
    batch_constraint::{
        bus::{
            BatchConstraintConductorBus, ConstraintsFoldingBus, Eq3bBus, EqNOuterBus,
            EqNegInternalBus, EqSharpUniBus, EqZeroNBus, ExpressionClaimBus,
            InteractionsFoldingBus, SumcheckClaimBus, SymbolicExpressionBus,
            UnivariateSumcheckInputBus,
        },
        eq_airs::{
            generate_eq_sharp_uni_blob, Eq3bAir, Eq3bBlob, EqNegAir, EqNegTraceGenerator, EqNsAir,
            EqSharpUniAir, EqSharpUniBlob, EqSharpUniReceiverAir, EqUniAir,
        },
        expr_eval::{
            ConstraintsFoldingAir, ConstraintsFoldingBlob, InteractionsFoldingAir,
            InteractionsFoldingBlob, SymbolicExpressionAir,
        },
        expression_claim::{
            generate_expression_claim_blob, ExpressionClaimAir, ExpressionClaimBlob,
            ExpressionClaimTraceGenerator,
        },
        fractions_folder::{FractionsFolderAir, FractionsFolderTraceGenerator},
        sumcheck::{
            multilinear::MultilinearSumcheckTraceGenerator,
            univariate::UnivariateSumcheckTraceGenerator, MultilinearSumcheckAir,
            UnivariateSumcheckAir,
        },
    },
    bus::{
        AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus, ConstraintSumcheckRandomnessBus,
        DagCommitBus, EqNegBaseRandBus, EqNegResultBus, ExpressionClaimNMaxBus,
        FractionFolderInputBus, HyperdimBus, PublicValuesBus, SelHypercubeBus, SelUniBus,
        StackingModuleBus, TranscriptBus, XiRandomnessBus,
    },
    primitives::{bus::PowerCheckerBus, pow::PowerCheckerTraceGenerator},
    system::{
        AirModule, BatchConstraintPreflight, BusIndexManager, BusInventory, GlobalCtxCpu,
        Preflight, TraceGenModule,
    },
    tracegen::{ModuleChip, RowMajorChip, StandardTracegenCtx},
    utils::MultiVecWithBounds,
};

pub mod bus;
pub mod eq_airs;
pub mod expr_eval;
pub mod expression_claim;
pub mod fractions_folder;
pub mod sumcheck;

#[cfg(feature = "cuda")]
mod cuda_abi;
#[cfg(feature = "cuda")]
mod cuda_utils;

/// AIR index within the BatchConstraintModule
pub(crate) const LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX: usize = 0;

pub struct BatchConstraintModule {
    transcript_bus: TranscriptBus,
    constraint_sumcheck_randomness_bus: ConstraintSumcheckRandomnessBus,
    xi_randomness_bus: XiRandomnessBus,
    gkr_claim_bus: BatchConstraintModuleBus,
    fraction_folder_input_bus: FractionFolderInputBus,
    univariate_sumcheck_input_bus: UnivariateSumcheckInputBus,
    stacking_module_bus: StackingModuleBus,
    column_opening_bus: ColumnClaimsBus,
    air_shape_bus: AirShapeBus,
    hyperdim_bus: HyperdimBus,
    public_values_bus: PublicValuesBus,
    sel_uni_bus: SelUniBus,
    eq_n_outer_bus: EqNOuterBus,

    batch_constraint_conductor_bus: BatchConstraintConductorBus,
    sumcheck_bus: SumcheckClaimBus,
    expression_claim_n_max_bus: ExpressionClaimNMaxBus,

    zero_n_bus: EqZeroNBus,
    eq_sharp_uni_bus: EqSharpUniBus,
    eq_3b_bus: Eq3bBus,
    sel_hypercube_bus: SelHypercubeBus,
    eq_neg_result_bus: EqNegResultBus,
    eq_neg_base_rand_bus: EqNegBaseRandBus,
    eq_neg_internal_bus: EqNegInternalBus,

    symbolic_expression_bus: SymbolicExpressionBus,
    expression_claim_bus: ExpressionClaimBus,
    interactions_folding_bus: InteractionsFoldingBus,
    constraints_folding_bus: ConstraintsFoldingBus,
    power_checker_bus: PowerCheckerBus,
    pow_checker: Arc<PowerCheckerTraceGenerator<2, 32>>,

    l_skip: usize,
    max_constraint_degree: usize,

    max_num_proofs: usize,
    pub(crate) has_cached: bool,
    dag_commit_bus: DagCommitBus,
}

impl BatchConstraintModule {
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        max_num_proofs: usize,
        pow_checker: Arc<PowerCheckerTraceGenerator<2, 32>>,
        has_cached: bool,
    ) -> Self {
        let l_skip = child_vk.inner.params.l_skip;
        let max_constraint_degree = child_vk.max_constraint_degree();
        BatchConstraintModule {
            transcript_bus: bus_inventory.transcript_bus,
            constraint_sumcheck_randomness_bus: bus_inventory.constraint_randomness_bus,
            xi_randomness_bus: bus_inventory.xi_randomness_bus,
            gkr_claim_bus: bus_inventory.bc_module_bus,
            fraction_folder_input_bus: bus_inventory.fraction_folder_input_bus,
            stacking_module_bus: bus_inventory.stacking_module_bus,
            column_opening_bus: bus_inventory.column_claims_bus,
            air_shape_bus: bus_inventory.air_shape_bus,
            hyperdim_bus: bus_inventory.hyperdim_bus,
            public_values_bus: bus_inventory.public_values_bus,
            sel_uni_bus: bus_inventory.sel_uni_bus,
            eq_neg_base_rand_bus: bus_inventory.eq_neg_base_rand_bus,
            eq_neg_result_bus: bus_inventory.eq_neg_result_bus,
            expression_claim_n_max_bus: bus_inventory.expression_claim_n_max_bus,
            batch_constraint_conductor_bus: BatchConstraintConductorBus::new(b.new_bus_idx()),
            univariate_sumcheck_input_bus: UnivariateSumcheckInputBus::new(b.new_bus_idx()),
            sumcheck_bus: SumcheckClaimBus::new(b.new_bus_idx()),

            zero_n_bus: EqZeroNBus::new(b.new_bus_idx()),
            eq_sharp_uni_bus: EqSharpUniBus::new(b.new_bus_idx()),
            eq_3b_bus: Eq3bBus::new(b.new_bus_idx()),
            eq_neg_internal_bus: EqNegInternalBus::new(b.new_bus_idx()),
            sel_hypercube_bus: SelHypercubeBus::new(b.new_bus_idx()),
            eq_n_outer_bus: EqNOuterBus::new(b.new_bus_idx()),
            // sel_uni bus is shared via inventory
            symbolic_expression_bus: SymbolicExpressionBus::new(b.new_bus_idx()),
            expression_claim_bus: ExpressionClaimBus::new(b.new_bus_idx()),
            interactions_folding_bus: InteractionsFoldingBus::new(b.new_bus_idx()),
            constraints_folding_bus: ConstraintsFoldingBus::new(b.new_bus_idx()),
            power_checker_bus: bus_inventory.power_checker_bus,
            pow_checker,
            l_skip,
            max_constraint_degree,
            max_num_proofs,
            has_cached,
            dag_commit_bus: bus_inventory.dag_commit_bus,
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    pub fn run_preflight<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory,
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
            let need_rot = child_vk.inner.per_air[*air_id].params.need_rot;
            for (col_opening, rot_opening) in
                column_openings_by_rot(&column_openings[sort_idx][0], need_rot)
            {
                ts.observe_ext(col_opening);
                ts.observe_ext(rot_opening);
            }
        }

        for (sort_idx, (air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let need_rot = child_vk.inner.per_air[*air_id].params.need_rot;
            for part in column_openings[sort_idx].iter().skip(1) {
                for (col_opening, rot_opening) in column_openings_by_rot(part, need_rot) {
                    ts.observe_ext(col_opening);
                    ts.observe_ext(rot_opening);
                }
            }
        }

        let omega_skip_pows = F::two_adic_generator(l_skip)
            .powers()
            .take(1 << l_skip)
            .collect_vec();

        let mut eq_ns = Vec::with_capacity(preflight.proof_shape.n_max + 1);
        let mut eq_sharp_ns = Vec::with_capacity(preflight.proof_shape.n_max + 1);
        let mut eq = eval_eq_uni(l_skip, xi[0], sumcheck_rnd[0]);
        let mut eq_sharp = eval_eq_sharp_uni(&omega_skip_pows, &xi[..l_skip], sumcheck_rnd[0]);
        eq_ns.push(eq);
        eq_sharp_ns.push(eq_sharp);
        for i in 0..preflight.proof_shape.n_max {
            let mult = EF::ONE - xi[l_skip + i] - sumcheck_rnd[1 + i]
                + (xi[l_skip + i] * sumcheck_rnd[1 + i]).double();
            eq *= mult;
            eq_sharp *= mult;
            eq_ns.push(eq);
            eq_sharp_ns.push(eq_sharp);
        }

        let mut r_rev_prod = sumcheck_rnd[preflight.proof_shape.n_max];
        let mut eq_ns_frontloaded = Vec::with_capacity(preflight.proof_shape.n_max + 1);
        let mut eq_sharp_ns_frontloaded = Vec::with_capacity(preflight.proof_shape.n_max + 1);
        // Product with r_i's to account for \hat{f} vs \tilde{f} for different n's in front-loaded
        // batch sumcheck.
        for i in (0..preflight.proof_shape.n_max).rev() {
            eq_ns_frontloaded.push(eq_ns[i] * r_rev_prod);
            eq_sharp_ns_frontloaded.push(eq_sharp_ns[i] * r_rev_prod);
            r_rev_prod *= sumcheck_rnd[i];
        }
        eq_ns_frontloaded.reverse();
        eq_sharp_ns_frontloaded.reverse();
        eq_ns_frontloaded.push(eq_ns[preflight.proof_shape.n_max]);
        eq_sharp_ns_frontloaded.push(eq_sharp_ns[preflight.proof_shape.n_max]);

        preflight.batch_constraint = BatchConstraintPreflight {
            lambda_tidx,
            tidx_before_univariate,
            tidx_before_multilinear,
            tidx_before_column_openings,
            post_tidx: ts.len(),
            xi,
            sumcheck_rnd,
            eq_ns,
            eq_sharp_ns,
            eq_ns_frontloaded,
            eq_sharp_ns_frontloaded,
        }
    }
}

impl AirModule for BatchConstraintModule {
    fn num_airs(&self) -> usize {
        BatchConstraintModuleChipDiscriminants::COUNT
    }

    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let l_skip = self.l_skip;

        let symbolic_expression_air = SymbolicExpressionAir {
            expr_bus: self.symbolic_expression_bus,
            claim_bus: self.expression_claim_bus,
            air_shape_bus: self.air_shape_bus,
            column_claims_bus: self.column_opening_bus,
            interactions_folding_bus: self.interactions_folding_bus,
            constraints_folding_bus: self.constraints_folding_bus,
            hyperdim_bus: self.hyperdim_bus,
            public_values_bus: self.public_values_bus,
            sel_hypercube_bus: self.sel_hypercube_bus,
            sel_uni_bus: self.sel_uni_bus,
            eq_neg_internal_bus: self.eq_neg_internal_bus,
            dag_commit_bus: self.dag_commit_bus,
            cnt_proofs: self.max_num_proofs,
            has_cached: self.has_cached,
        };
        let fraction_folder_air = FractionsFolderAir {
            transcript_bus: self.transcript_bus,
            univariate_sumcheck_input_bus: self.univariate_sumcheck_input_bus,
            fraction_folder_input_bus: self.fraction_folder_input_bus,
            sumcheck_bus: self.sumcheck_bus,
            mu_bus: self.batch_constraint_conductor_bus,
            gkr_claim_bus: self.gkr_claim_bus,
        };
        let sumcheck_uni_air = UnivariateSumcheckAir {
            l_skip,
            univariate_deg: (self.max_constraint_degree + 1) * ((1 << l_skip) - 1),
            univariate_sumcheck_input_bus: self.univariate_sumcheck_input_bus,
            stacking_module_bus: self.stacking_module_bus,
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
            stacking_module_bus: self.stacking_module_bus,
        };
        let eq_ns_air = EqNsAir {
            zero_n_bus: self.zero_n_bus,
            xi_bus: self.xi_randomness_bus,
            r_xi_bus: self.batch_constraint_conductor_bus,
            sel_hypercube_bus: self.sel_hypercube_bus,
            eq_n_outer_bus: self.eq_n_outer_bus,
            l_skip,
        };
        let eq_3b_air = Eq3bAir {
            eq_3b_bus: self.eq_3b_bus,
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
        let eq_neg_air = EqNegAir {
            result_bus: self.eq_neg_result_bus,
            base_rand_bus: self.eq_neg_base_rand_bus,
            internal_bus: self.eq_neg_internal_bus,
            sel_uni_bus: self.sel_uni_bus,
            l_skip: self.l_skip,
        };
        let expression_claim_air = ExpressionClaimAir {
            expression_claim_n_max_bus: self.expression_claim_n_max_bus,
            expr_claim_bus: self.expression_claim_bus,
            mu_bus: self.batch_constraint_conductor_bus,
            sumcheck_claim_bus: self.sumcheck_bus,
            eq_n_outer_bus: self.eq_n_outer_bus,
            pow_checker_bus: self.power_checker_bus,
            hyperdim_bus: self.hyperdim_bus,
        };
        let interactions_folding_air = InteractionsFoldingAir {
            transcript_bus: self.transcript_bus,
            air_shape_bus: self.air_shape_bus,
            interaction_bus: self.interactions_folding_bus,
            expression_claim_bus: self.expression_claim_bus,
            eq_3b_bus: self.eq_3b_bus,
        };
        let constraints_folding_air = ConstraintsFoldingAir {
            transcript_bus: self.transcript_bus,
            constraint_bus: self.constraints_folding_bus,
            expression_claim_bus: self.expression_claim_bus,
            eq_n_outer_bus: self.eq_n_outer_bus,
        };
        // WARNING: SymbolicExpressionAir MUST be the first AIR in verifier circuit
        vec![
            Arc::new(symbolic_expression_air) as AirRef<_>,
            Arc::new(fraction_folder_air) as AirRef<_>,
            Arc::new(sumcheck_uni_air) as AirRef<_>,
            Arc::new(sumcheck_lin_air) as AirRef<_>,
            Arc::new(eq_ns_air) as AirRef<_>,
            Arc::new(eq_3b_air) as AirRef<_>,
            Arc::new(eq_sharp_uni_air) as AirRef<_>,
            Arc::new(eq_sharp_uni_receiver_air) as AirRef<_>,
            Arc::new(eq_uni_air) as AirRef<_>,
            Arc::new(expression_claim_air) as AirRef<_>,
            Arc::new(interactions_folding_air) as AirRef<_>,
            Arc::new(constraints_folding_air) as AirRef<_>,
            Arc::new(eq_neg_air) as AirRef<_>,
        ]
    }
}

pub(crate) struct BatchConstraintBlob {
    // Per proof, per air (vkey order), the evaluations. For optional AIRs without traces, the
    // innermost vec is empty.
    pub expr_evals: MultiVecWithBounds<EF, 2>,
    // Per proof, per log height.
    pub selector_counts: MultiVecWithBounds<SelectorCount, 1>,

    pub eq_3b_blob: Eq3bBlob,
    pub eq_sharp_uni_blob: EqSharpUniBlob,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SelectorCount {
    pub first: usize,
    pub last: usize,
    pub transition: usize,
}

impl BatchConstraintBlob {
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[&Proof<BabyBearPoseidon2Config>],
        preflights: &[&Preflight],
    ) -> Self {
        let child_vk = &child_vk.inner;
        let params = &child_vk.params;

        let mut expr_evals_per_proof = MultiVecWithBounds::new();
        let mut eq_r_one_counts_per_proof = MultiVecWithBounds::new();
        for (proof, preflight) in zip(proofs, preflights) {
            let rs = &preflight.batch_constraint.sumcheck_rnd;

            let (&rs_0, rs_rest) = rs.split_first().unwrap();
            let mut is_first_row_by_log_height = vec![];
            let mut is_last_row_by_log_height = vec![];
            let n_max = preflight.proof_shape.n_max;
            let mut selector_counts =
                vec![SelectorCount::default(); child_vk.params.l_skip + n_max + 1];

            let omega = F::two_adic_generator(params.l_skip);
            for log_height in 0..=params.l_skip {
                is_first_row_by_log_height.push(eval_eq_uni_at_one(
                    log_height,
                    rs_0.exp_power_of_2(params.l_skip - log_height),
                ));
                is_last_row_by_log_height.push(eval_eq_uni_at_one(
                    log_height,
                    (rs_0 * omega).exp_power_of_2(params.l_skip - log_height),
                ));
            }
            for (i, &r) in rs_rest.iter().enumerate() {
                is_first_row_by_log_height
                    .push(is_first_row_by_log_height[params.l_skip + i] * (EF::ONE - r));
                is_last_row_by_log_height.push(is_last_row_by_log_height[params.l_skip + i] * r);
            }

            let mut expr_evals_per_air = vec![];
            for (air_idx, vk) in child_vk.per_air.iter().enumerate() {
                if proof.trace_vdata[air_idx].is_none() {
                    expr_evals_per_air.push(vec![]);
                    continue;
                }

                let need_rot = child_vk.per_air[air_idx].params.need_rot;
                let openings_per_col = if need_rot { 2 } else { 1 };
                let openings = &proof.batch_constraint_proof.column_openings;
                let (sorted_idx, vdata) = preflight
                    .proof_shape
                    .sorted_trace_vdata
                    .iter()
                    .enumerate()
                    .find_map(|(sorted_idx, (idx, vdata))| {
                        if air_idx == *idx {
                            Some((sorted_idx, vdata))
                        } else {
                            None
                        }
                    })
                    .unwrap();

                let constraints = &vk.symbolic_constraints.constraints;
                let mut expr_evals =
                    vec![EF::ZERO; constraints.nodes.len() + vk.unused_variables.len()];
                let log_height = proof.trace_vdata[air_idx].as_ref().unwrap().log_height;

                for (node_idx, node) in constraints.nodes.iter().enumerate() {
                    match node {
                        SymbolicExpressionNode::Variable(var) => match var.entry {
                            Entry::Preprocessed { offset } => {
                                debug_assert!(offset < openings_per_col);
                                expr_evals[node_idx] =
                                    openings[sorted_idx][1][var.index * openings_per_col + offset];
                            }
                            Entry::Main { part_index, offset } => {
                                let part = vk.dag_main_part_index_to_commit_index(part_index);
                                debug_assert!(offset < openings_per_col);
                                expr_evals[node_idx] = openings[sorted_idx][part]
                                    [var.index * openings_per_col + offset];
                            }
                            Entry::Permutation { .. } => unreachable!(),
                            Entry::Public => {
                                expr_evals[node_idx] =
                                    EF::from(proof.public_values[air_idx][var.index]);
                            }
                            Entry::Challenge => unreachable!(),
                            Entry::Exposed => unreachable!(),
                        },
                        SymbolicExpressionNode::IsFirstRow => {
                            expr_evals[node_idx] = is_first_row_by_log_height[vdata.log_height];
                            selector_counts[log_height].first += 1;
                        }
                        SymbolicExpressionNode::IsLastRow => {
                            expr_evals[node_idx] = is_last_row_by_log_height[vdata.log_height];
                            selector_counts[log_height].last += 1;
                        }
                        SymbolicExpressionNode::IsTransition => {
                            expr_evals[node_idx] =
                                EF::ONE - is_last_row_by_log_height[vdata.log_height];
                            selector_counts[log_height].transition += 1;
                        }
                        SymbolicExpressionNode::Constant(val) => {
                            expr_evals[node_idx] = EF::from(*val);
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
                let mut node_idx = constraints.nodes.len();
                for unused_var in &vk.unused_variables {
                    match unused_var.entry {
                        Entry::Preprocessed { offset } => {
                            debug_assert!(offset < openings_per_col);
                            expr_evals[node_idx] = openings[sorted_idx][1]
                                [unused_var.index * openings_per_col + offset];
                        }
                        Entry::Main { part_index, offset } => {
                            let part = vk.dag_main_part_index_to_commit_index(part_index);
                            debug_assert!(offset < openings_per_col);
                            expr_evals[node_idx] = openings[sorted_idx][part]
                                [unused_var.index * openings_per_col + offset];
                        }
                        Entry::Permutation { .. }
                        | Entry::Public
                        | Entry::Challenge
                        | Entry::Exposed => {
                            unreachable!()
                        }
                    }
                    node_idx += 1;
                }
                expr_evals_per_air.push(expr_evals);
            }
            for v in expr_evals_per_air {
                expr_evals_per_proof.extend(v);
                expr_evals_per_proof.close_level(1);
            }
            expr_evals_per_proof.close_level(0);
            eq_r_one_counts_per_proof.extend(selector_counts);
            eq_r_one_counts_per_proof.close_level(0);
        }
        let eq_3b_blob = eq_airs::generate_eq_3b_blob(child_vk, preflights);
        let eq_sharp_uni_blob = generate_eq_sharp_uni_blob(child_vk, preflights);
        Self {
            expr_evals: expr_evals_per_proof,
            selector_counts: eq_r_one_counts_per_proof,
            eq_3b_blob,
            eq_sharp_uni_blob,
        }
    }
}

pub(crate) struct BatchConstraintBlobCpu {
    pub common_blob: BatchConstraintBlob,
    pub cf_blob: Option<ConstraintsFoldingBlob>,
    pub if_blob: Option<InteractionsFoldingBlob>,
    pub expr_claim_blob: ExpressionClaimBlob,
}

impl BatchConstraintBlobCpu {
    #[tracing::instrument(name = "generate_blob", skip_all)]
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
    ) -> Self {
        let proofs = proofs.iter().collect_vec();
        let preflights = preflights.iter().collect_vec();
        let common_blob = BatchConstraintBlob::new(child_vk, &proofs, &preflights);
        let cf_blob =
            ConstraintsFoldingBlob::new(&child_vk.inner, &common_blob.expr_evals, &preflights);
        let if_blob = InteractionsFoldingBlob::new(
            &child_vk.inner,
            &common_blob.expr_evals,
            &common_blob.eq_3b_blob,
            &preflights,
        );
        let expr_claim_blob =
            generate_expression_claim_blob(&cf_blob.folded_claims, &if_blob.folded_claims);
        Self {
            common_blob,
            cf_blob: Some(cf_blob),
            if_blob: Some(if_blob),
            expr_claim_blob,
        }
    }
}

impl TraceGenModule<GlobalCtxCpu, CpuBackend<BabyBearPoseidon2Config>> for BatchConstraintModule {
    type ModuleSpecificCtx = ();

    /// **Note**: This generates all common main traces but leaves the cached trace for
    /// `SymbolicExpressionAir` unset. The cached trace must be loaded **after** calling this
    /// function.
    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        _ctx: &(),
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>> {
        let blob = BatchConstraintBlobCpu::new(child_vk, proofs, preflights);
        let ctx = (
            StandardTracegenCtx {
                vk: child_vk,
                proofs: &proofs.iter().collect_vec(),
                preflights: &preflights.iter().collect_vec(),
            },
            blob,
        );

        let chips = [
            BatchConstraintModuleChip::SymbolicExpression {
                max_num_proofs: self.max_num_proofs,
                has_cached: self.has_cached,
            },
            BatchConstraintModuleChip::FractionsFolder,
            BatchConstraintModuleChip::SumcheckUni,
            BatchConstraintModuleChip::SumcheckLin,
            BatchConstraintModuleChip::EqNs,
            BatchConstraintModuleChip::Eq3b,
            BatchConstraintModuleChip::EqSharpUni,
            BatchConstraintModuleChip::EqSharpUniReceiver,
            BatchConstraintModuleChip::EqUni,
            BatchConstraintModuleChip::ExpressionClaim {
                pow_checker: self.pow_checker.clone(),
            },
            BatchConstraintModuleChip::InteractionsFolding,
            BatchConstraintModuleChip::ConstraintsFolding,
            BatchConstraintModuleChip::EqNeg,
        ];

        let span = tracing::Span::current();
        chips
            .par_iter()
            .map(|chip| {
                let _guard = span.enter();
                chip.generate_proving_ctx(
                    &ctx,
                    required_heights.map(|heights| heights[chip.index()]),
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }
}

impl BatchConstraintModule {
    /// Generates and then commits to the cache trace for `SymbolicExpressionAir`. Returns the
    /// committed PCS data.
    pub fn commit_child_vk<E>(
        &self,
        engine: &E,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CommittedTraceData<CpuBackend<BabyBearPoseidon2Config>>
    where
        E: StarkEngine<SC = BabyBearPoseidon2Config, PB = CpuBackend<BabyBearPoseidon2Config>>,
    {
        let cached_trace_rm = expr_eval::generate_symbolic_expr_cached_trace(child_vk);
        let cached_trace = ColMajorMatrix::from_row_major(&cached_trace_rm);
        let (commitment, data) = engine.device().commit(&[&cached_trace]);
        CommittedTraceData {
            commitment,
            data: Arc::new(data),
            trace: cached_trace,
        }
    }
}

// NOTE: ordering of enum must match AIR ordering
#[derive(strum_macros::Display, EnumDiscriminants)]
#[strum_discriminants(derive(strum_macros::EnumCount))]
#[strum_discriminants(repr(usize))]
enum BatchConstraintModuleChip {
    SymbolicExpression {
        max_num_proofs: usize,
        has_cached: bool,
    },
    FractionsFolder,
    SumcheckUni,
    SumcheckLin,
    EqNs,
    Eq3b,
    EqSharpUni,
    EqSharpUniReceiver,
    EqUni,
    ExpressionClaim {
        pow_checker: Arc<PowerCheckerTraceGenerator<2, 32>>,
    },
    InteractionsFolding,
    ConstraintsFolding,
    EqNeg,
}

impl BatchConstraintModuleChip {
    fn index(&self) -> usize {
        BatchConstraintModuleChipDiscriminants::from(self) as usize
    }
}

impl RowMajorChip<F> for BatchConstraintModuleChip {
    type Ctx<'a> = (StandardTracegenCtx<'a>, BatchConstraintBlobCpu);

    #[tracing::instrument(
        name = "wrapper.generate_trace",
        level = "trace",
        skip_all,
        fields(air = %self)
    )]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        use BatchConstraintModuleChip::*;
        let child_vk = ctx.0.vk;
        let proofs = ctx.0.proofs;
        let preflights = ctx.0.preflights;
        let blob = &ctx.1;
        match self {
            FractionsFolder => {
                FractionsFolderTraceGenerator.generate_trace(&ctx.0, required_height)
            }
            SumcheckUni => UnivariateSumcheckTraceGenerator.generate_trace(&ctx.0, required_height),
            SumcheckLin => {
                MultilinearSumcheckTraceGenerator.generate_trace(&ctx.0, required_height)
            }
            EqNs => eq_airs::EqNsTraceGenerator.generate_trace(
                &(child_vk, preflights, &blob.common_blob.selector_counts),
                required_height,
            ),
            Eq3b => eq_airs::Eq3bTraceGenerator.generate_trace(
                &(child_vk, &blob.common_blob.eq_3b_blob, preflights),
                required_height,
            ),
            EqSharpUni => eq_airs::EqSharpUniTraceGenerator.generate_trace(
                &(child_vk, &blob.common_blob.eq_sharp_uni_blob, preflights),
                required_height,
            ),
            EqSharpUniReceiver => eq_airs::EqSharpUniReceiverTraceGenerator.generate_trace(
                &(child_vk, &blob.common_blob.eq_sharp_uni_blob, preflights),
                required_height,
            ),
            EqUni => eq_airs::EqUniTraceGenerator.generate_trace(&ctx.0, required_height),
            SymbolicExpression {
                max_num_proofs,
                has_cached,
            } => expr_eval::SymbolicExpressionTraceGenerator {
                max_num_proofs: *max_num_proofs,
                has_cached: *has_cached,
            }
            .generate_trace(
                &expr_eval::SymbolicExpressionCtx {
                    vk: child_vk,
                    proofs,
                    preflights,
                    expr_evals: &blob.common_blob.expr_evals,
                },
                required_height,
            ),
            ExpressionClaim { pow_checker } => ExpressionClaimTraceGenerator.generate_trace(
                &expression_claim::ExpressionClaimCtx {
                    blob: &blob.expr_claim_blob,
                    proofs,
                    preflights,
                    pow_checker: pow_checker.as_ref(),
                },
                required_height,
            ),
            InteractionsFolding => expr_eval::InteractionsFoldingTraceGenerator
                .generate_trace(&(child_vk, blob, preflights), required_height),
            ConstraintsFolding => expr_eval::ConstraintsFoldingTraceGenerator.generate_trace(
                &(blob.cf_blob.as_ref().unwrap(), preflights),
                required_height,
            ),
            EqNeg => EqNegTraceGenerator.generate_trace(
                &(child_vk, preflights, &blob.common_blob.selector_counts),
                required_height,
            ),
        }
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_tracegen {
    use openvm_cuda_backend::{data_transporter::transport_matrix_h2d_col_major, GpuBackend};

    use super::*;
    use crate::{
        batch_constraint::expr_eval::{
            constraints_folding::cuda::ConstraintsFoldingBlobGpu,
            interactions_folding::cuda::InteractionsFoldingBlobGpu,
        },
        cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu, GlobalCtxGpu},
        tracegen::cuda::{generate_gpu_proving_ctx, StandardTracegenGpuCtx},
    };

    impl ModuleChip<GpuBackend> for BatchConstraintModuleChip {
        type Ctx<'a> = (StandardTracegenGpuCtx<'a>, &'a BatchConstraintBlobGpu);

        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            use BatchConstraintModuleChip::*;
            let child_vk = ctx.0.vk;
            let proofs = ctx.0.proofs;
            let preflights = ctx.0.preflights;
            let blob = ctx.1;
            match self {
                Eq3b => eq_airs::Eq3bTraceGenerator.generate_proving_ctx(
                    &(&child_vk.cpu, &blob.common_blob.eq_3b_blob, preflights),
                    required_height,
                ),
                SymbolicExpression {
                    max_num_proofs,
                    has_cached,
                } => expr_eval::SymbolicExpressionTraceGenerator {
                    max_num_proofs: *max_num_proofs,
                    has_cached: *has_cached,
                }
                .generate_proving_ctx(
                    &expr_eval::symbolic_expression::cuda::SymbolicExpressionGpuCtx {
                        vk: &child_vk.cpu,
                        proofs,
                        preflights,
                        expr_evals: &blob.common_blob.expr_evals,
                    },
                    required_height,
                ),
                InteractionsFolding => expr_eval::InteractionsFoldingTraceGenerator
                    .generate_proving_ctx(&(child_vk, preflights, &blob.if_blob), required_height),
                ConstraintsFolding => expr_eval::ConstraintsFoldingTraceGenerator
                    .generate_proving_ctx(&(child_vk, preflights, &blob.cf_blob), required_height),
                _ => unreachable!(),
            }
        }
    }

    pub(in crate::batch_constraint) struct BatchConstraintBlobGpu {
        pub common_blob: BatchConstraintBlob,
        pub cf_blob: ConstraintsFoldingBlobGpu,
        pub if_blob: InteractionsFoldingBlobGpu,
        pub expr_claim_blob: ExpressionClaimBlob,
    }

    impl BatchConstraintBlobGpu {
        #[tracing::instrument(name = "generate_blob", skip_all)]
        pub fn new(
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
        ) -> Self {
            let cpu_proofs = proofs.iter().map(|p| &p.cpu).collect_vec();
            let cpu_preflights = preflights.iter().map(|p| &p.cpu).collect_vec();
            let common_blob = BatchConstraintBlob::new(&child_vk.cpu, &cpu_proofs, &cpu_preflights);
            let cf_blob =
                ConstraintsFoldingBlobGpu::new(child_vk, &common_blob.expr_evals, preflights);
            let if_blob = InteractionsFoldingBlobGpu::new(
                child_vk,
                &common_blob.expr_evals,
                &common_blob.eq_3b_blob,
                preflights,
            );
            let expr_claim_blob =
                generate_expression_claim_blob(&cf_blob.folded_claims, &if_blob.folded_claims);
            Self {
                common_blob,
                cf_blob,
                if_blob,
                expr_claim_blob,
            }
        }
    }

    impl TraceGenModule<GlobalCtxGpu, GpuBackend> for BatchConstraintModule {
        type ModuleSpecificCtx = ();

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            _module_ctx: &(),
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContext<GpuBackend>>> {
            let blob = BatchConstraintBlobGpu::new(child_vk, proofs, preflights);
            let ctx = (
                StandardTracegenGpuCtx {
                    vk: child_vk,
                    proofs,
                    preflights,
                },
                &blob,
            );

            // Chips with cuda kernels for tracegen
            let gpu_chips = [
                BatchConstraintModuleChip::SymbolicExpression {
                    max_num_proofs: self.max_num_proofs,
                    has_cached: self.has_cached,
                },
                BatchConstraintModuleChip::Eq3b,
                BatchConstraintModuleChip::ConstraintsFolding,
                BatchConstraintModuleChip::InteractionsFolding,
            ];
            // Chips that will use fallback cpu tracegen
            let cpu_chips = [
                BatchConstraintModuleChip::FractionsFolder,
                BatchConstraintModuleChip::SumcheckUni,
                BatchConstraintModuleChip::SumcheckLin,
                BatchConstraintModuleChip::EqNs,
                BatchConstraintModuleChip::EqSharpUni,
                BatchConstraintModuleChip::EqSharpUniReceiver,
                BatchConstraintModuleChip::EqUni,
                BatchConstraintModuleChip::ExpressionClaim {
                    pow_checker: self.pow_checker.clone(),
                },
                BatchConstraintModuleChip::EqNeg,
            ];
            let span = tracing::Span::current();
            // NOTE: do NOT use par_iter since that will lead to kernels on cuda streams != default
            // stream, whereas previous H2D transfer was on default stream.
            let indexed_gpu_traces = gpu_chips
                .iter()
                .map(|chip| {
                    // This span is not very useful because the kernel does not synchronize on host:
                    let _guard = span.enter();
                    (
                        chip.index(),
                        chip.generate_proving_ctx(
                            &ctx,
                            required_heights.map(|heights| heights[chip.index()]),
                        ),
                    )
                })
                .collect_vec();

            let blob = BatchConstraintBlobCpu {
                common_blob: blob.common_blob,
                cf_blob: None,
                if_blob: None,
                expr_claim_blob: blob.expr_claim_blob,
            };
            let cpu_proofs = proofs.iter().map(|p| &p.cpu).collect_vec();
            let cpu_preflights = preflights.iter().map(|p| &p.cpu).collect_vec();
            let cpu_ctx = (
                StandardTracegenCtx {
                    vk: &child_vk.cpu,
                    proofs: &cpu_proofs,
                    preflights: &cpu_preflights,
                },
                blob,
            );

            // We parallelize the CPU trace generation
            let indexed_cpu_traces = cpu_chips
                .par_iter()
                .map(|chip| {
                    // span within par_iter to handle parallelism
                    let _guard = span.enter();
                    (
                        chip.index(),
                        generate_gpu_proving_ctx(
                            chip,
                            &cpu_ctx,
                            required_heights.map(|heights| heights[chip.index()]),
                        ),
                    )
                })
                .collect::<Vec<_>>();

            indexed_gpu_traces
                .into_iter()
                .chain(indexed_cpu_traces)
                .sorted_by(|a, b| a.0.cmp(&b.0))
                .map(|(_index, ctx)| ctx)
                .collect()
        }
    }

    impl BatchConstraintModule {
        /// Generates and then commits to the cache trace for `SymbolicExpressionAir`. Returns the
        /// committed PCS data.
        pub fn commit_child_vk_gpu<E>(
            &self,
            engine: &E,
            child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        ) -> CommittedTraceData<GpuBackend>
        where
            E: StarkEngine<SC = BabyBearPoseidon2Config, PB = GpuBackend>,
        {
            let cached_trace = expr_eval::generate_symbolic_expr_cached_trace(child_vk);
            // TODO: gpu transpose
            let cached_trace = ColMajorMatrix::from_row_major(&cached_trace);
            let d_cached_trace = transport_matrix_h2d_col_major(&cached_trace).unwrap();
            let (commitment, data) = engine.device().commit(&[&d_cached_trace]);
            CommittedTraceData {
                commitment,
                trace: d_cached_trace,
                data: Arc::new(data),
            }
        }
    }
}
