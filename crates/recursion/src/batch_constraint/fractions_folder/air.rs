use std::borrow::Borrow;

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, extension::BinomiallyExtendable};
use p3_matrix::Matrix;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{SumcheckClaimBus, SumcheckClaimMessage},
    bus::{
        BatchConstraintModuleBus, BatchConstraintModuleMessage, TranscriptBus, TranscriptBusMessage,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{assert_eq_array, ext_field_add, ext_field_multiply},
};

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct FractionsFolderCols<T> {
    pub is_valid: T,
    pub is_first: T,

    pub proof_idx: T,
    pub air_idx: T,

    pub tidx_alpha_beta: T,
    pub gkr_post_tidx: T, // TODO: this number of tids is really annoying
    pub tidx: T,
    pub n_global: T,

    pub sum_claim_p: [T; D_EF],
    pub sum_claim_q: [T; D_EF],
    pub cur_p_sum: [T; D_EF],
    pub cur_q_sum: [T; D_EF],
    pub mu: [T; D_EF],
    pub cur_hash: [T; D_EF],
}

pub struct FractionsFolderAir {
    pub transcript_bus: TranscriptBus,
    pub sumcheck_bus: SumcheckClaimBus,
    pub gkr_claim_bus: BatchConstraintModuleBus,
}

impl<F> BaseAirWithPublicValues<F> for FractionsFolderAir {}
impl<F> PartitionedBaseAir<F> for FractionsFolderAir {}

impl<F> BaseAir<F> for FractionsFolderAir {
    fn width(&self) -> usize {
        FractionsFolderCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FractionsFolderAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &FractionsFolderCols<AB::Var> = (*local).borrow();
        let next: &FractionsFolderCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Loop Constraints
        ///////////////////////////////////////////////////////////////////////

        type LoopSubAir = NestedForLoopSubAir<1, 0>;

        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        let is_last_air = LoopSubAir::local_is_last(next.is_valid, next.is_first);
        let is_transition = AB::Expr::ONE - is_last_air.clone();

        // TODO(ayush): do i need to constrain that air_idx starts at a particular value?
        // builder
        //     .when(local.is_first)
        //     .assert_zero(local.air_idx);

        // Air index decrements by 1
        builder
            .when(local.is_valid * is_transition.clone())
            .assert_eq(next.air_idx, local.air_idx - AB::Expr::ONE);

        builder.when(is_last_air.clone()).assert_zero(local.air_idx);

        ///////////////////////////////////////////////////////////////////////
        // Transition Constraints
        ///////////////////////////////////////////////////////////////////////

        let is_transition = AB::Expr::ONE - is_last_air.clone();

        assert_array_eq(&mut builder.when(is_transition.clone()), local.mu, next.mu);

        builder.when(is_transition.clone()).assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_canonical_usize(2 * D_EF),
        );

        ///////////////////////////////////////////////////////////////////////
        // Running Sum Constraints
        ///////////////////////////////////////////////////////////////////////

        // Initialize running sum
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.cur_p_sum,
            local.sum_claim_p,
        );

        // Add air sum to running sum
        assert_array_eq(
            &mut builder.when(is_transition.clone()),
            next.cur_p_sum,
            ext_field_add(local.cur_p_sum, next.sum_claim_p),
        );

        ///////////////////////////////////////////////////////////////////////
        // Polynomial Hash Evaluation (a la Horner's Method)
        ///////////////////////////////////////////////////////////////////////

        // Initialize hash
        // h = p + mu * q
        assert_eq_array(
            &mut builder.when(local.is_first),
            local.cur_hash,
            ext_field_add(
                local.sum_claim_p,
                ext_field_multiply(local.mu, local.sum_claim_q),
            ),
        );

        // Update hash
        // h' = p + mu * (q + mu * h)
        assert_eq_array(
            &mut builder.when(is_transition.clone()),
            next.cur_hash,
            ext_field_add(
                next.sum_claim_p,
                ext_field_multiply(
                    next.mu,
                    ext_field_add(
                        next.sum_claim_q,
                        ext_field_multiply(next.mu, local.cur_hash),
                    ),
                ),
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx,
                    value: local.sum_claim_p[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(D_EF + i) + local.tidx,
                    value: local.sum_claim_q[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(2 * D_EF + i) + local.tidx,
                    value: local.mu[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_valid * is_last_air.clone(),
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx_alpha_beta,
                    value: local.cur_q_sum[i] - local.sum_claim_q[i],
                    is_sample: AB::Expr::ONE,
                },
                local.is_first,
            );
        }
        self.sumcheck_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: AB::Expr::ZERO,
                // TODO(ayush): add back
                // value: local.cur_hash.map(|x| x.into()),
                value: [AB::Expr::ZERO; D_EF],
            },
            local.is_first,
        );

        self.gkr_claim_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintModuleMessage {
                tidx_alpha_beta: local.tidx_alpha_beta.into(),
                tidx: local.gkr_post_tidx.into(),
                n_global: local.n_global.into(),
                gkr_input_layer_claim: [
                    local.cur_p_sum.map(|x| x.into()),
                    local.cur_q_sum.map(|x| x.into()),
                ],
            },
            local.is_valid * is_last_air.clone(),
        );
    }
}
