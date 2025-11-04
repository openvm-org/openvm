use core::iter::zip;
use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{AirRef, keygen::types::TraceWidth};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::{Field, TwoAdicField};
use p3_matrix::Matrix;
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, Digest, F, StarkEngineV2,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::{BatchConstraintProof, Proof},
    prover::{
        AirProvingContextV2, ColMajorMatrix, CpuBackendV2, CpuDeviceV2, TraceCommitterV2,
        stacked_pcs::StackedPcsData,
    },
};

use crate::{
    batch_constraint::{
        bus::{
            BatchConstraintConductorBus, Eq3bBus, EqMleBus, EqSharpUniBus, EqZeroNBus,
            ExpressionClaimBus, InteractionsFoldingBus, SumcheckClaimBus, SymbolicExpressionBus,
        },
        eq_airs::{Eq3bAir, EqMleAir, EqNsAir, EqSharpUniAir, EqSharpUniReceiverAir, EqUniAir},
        expr_eval::{
            ColumnClaimAir, ExpressionClaimAir, InteractionsFoldingAir, SymbolicExpressionAir,
        },
        fractions_folder::FractionsFolderAir,
        sumcheck::{MultilinearSumcheckAir, UnivariateSumcheckAir},
    },
    bus::{
        AirPartShapeBus, AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus,
        ConstraintSumcheckRandomnessBus, StackingModuleBus, TranscriptBus, XiRandomnessBus,
    },
    system::{
        AirModule, BatchConstraintPreflight, BusIndexManager, BusInventory, GlobalCtxCpu,
        Preflight, TraceGenModule,
    },
};

pub mod bus;
pub mod eq_airs;
pub mod expr_eval;
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

    batch_constraint_conductor_bus: BatchConstraintConductorBus,
    sumcheck_bus: SumcheckClaimBus,

    zero_n_bus: EqZeroNBus,
    eq_mle_bus: EqMleBus,
    eq_sharp_uni_bus: EqSharpUniBus,
    eq_3b_bus: Eq3bBus,

    symbolic_expression_bus: SymbolicExpressionBus,
    expression_claim_bus: ExpressionClaimBus,
    interactions_folding_bus: InteractionsFoldingBus,

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
            batch_constraint_conductor_bus: BatchConstraintConductorBus::new(b.new_bus_idx()),
            sumcheck_bus: SumcheckClaimBus::new(b.new_bus_idx()),
            zero_n_bus: EqZeroNBus::new(b.new_bus_idx()),
            eq_mle_bus: EqMleBus::new(b.new_bus_idx()),
            eq_sharp_uni_bus: EqSharpUniBus::new(b.new_bus_idx()),
            eq_3b_bus: Eq3bBus::new(b.new_bus_idx()),
            symbolic_expression_bus: SymbolicExpressionBus::new(b.new_bus_idx()),
            expression_claim_bus: ExpressionClaimBus::new(b.new_bus_idx()),
            interactions_folding_bus: InteractionsFoldingBus::new(b.new_bus_idx()),
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
            univariate_deg: (self.max_constraint_degree + 1) * ((1 << l_skip) - 1) + 1,
            domain_size: 1 << l_skip,
            claim_bus: self.sumcheck_bus,
            transcript_bus: self.transcript_bus,
            randomness_bus: self.constraint_sumcheck_randomness_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
        };
        let sumcheck_lin_air = MultilinearSumcheckAir {
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
            stacking_module_bus: self.stacking_module_bus,
            column_claims_bus: self.column_opening_bus,
            interactions_folding_bus: self.interactions_folding_bus,
            cnt_proofs: self.max_num_proofs,
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
            Arc::new(eq_uni_air) as AirRef<_>,
            Arc::new(eq_3b_air) as AirRef<_>,
            Arc::new(symbolic_expression_air) as AirRef<_>,
            Arc::new(column_claim_air) as AirRef<_>,
            Arc::new(expression_claim_air) as AirRef<_>,
            Arc::new(interactions_folding_air) as AirRef<_>,
        ]
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
        let common = expr_eval::generate_symbolic_expr_common_trace(
            child_vk,
            proofs,
            preflights,
            self.max_num_proofs,
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
            transpose(expr_eval::generate_expression_claim_trace(
                child_vk, proofs, preflights,
            )),
            transpose(expr_eval::generate_interactions_folding_trace(
                child_vk, proofs, preflights,
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
        assert_eq!(
            cached_trace.height(),
            1,
            "fix me once cached trace is implemented"
        );
        let mut fake_params = engine.device().config();
        fake_params.l_skip = 0;
        let fake_device = CpuDeviceV2::new(fake_params);
        fake_device.commit(&[&ColMajorMatrix::from_row_major(&cached_trace)])
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_tracegen {
    use cuda_backend_v2::{
        BabyBearPoseidon2GpuEngineV2, GpuBackendV2, GpuDeviceV2, stacked_pcs::StackedPcsDataGpu,
        transport_matrix_h2d_col_major,
    };
    use itertools::Itertools;

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
            // TODO: gpu tracegen
            let cached_trace = expr_eval::generate_symbolic_expr_cached_trace(child_vk);
            assert_eq!(
                cached_trace.height(),
                1,
                "fix me once cached trace is implemented"
            );
            let mut fake_params = engine.device().config();
            fake_params.l_skip = 0;
            let fake_device = GpuDeviceV2::new(fake_params);
            let cached_trace = ColMajorMatrix::from_row_major(&cached_trace);
            fake_device.commit(&[&transport_matrix_h2d_col_major(&cached_trace).unwrap()])
        }
    }
}
