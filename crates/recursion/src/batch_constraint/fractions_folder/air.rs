use std::borrow::Borrow;

use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, SumcheckClaimBus, SumcheckClaimMessage,
        UnivariateSumcheckInputBus, UnivariateSumcheckInputMessage,
    },
    bus::{
        BatchConstraintModuleBus, BatchConstraintModuleMessage, FractionFolderInputBus,
        FractionFolderInputMessage, TranscriptBus,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{ext_field_add, ext_field_multiply},
};

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct FractionsFolderCols<T> {
    pub is_valid: T,
    pub proof_idx: T,
    pub is_first: T,

    pub air_idx: T,

    pub tidx: T,

    pub sum_claim_p: [T; D_EF],
    pub sum_claim_q: [T; D_EF],
    pub cur_p_sum: [T; D_EF],
    pub cur_q_sum: [T; D_EF],
    pub mu: [T; D_EF],
    pub cur_hash: [T; D_EF],
}

pub struct FractionsFolderAir {
    pub transcript_bus: TranscriptBus,
    pub fraction_folder_input_bus: FractionFolderInputBus,
    pub univariate_sumcheck_input_bus: UnivariateSumcheckInputBus,
    pub sumcheck_bus: SumcheckClaimBus,
    pub mu_bus: BatchConstraintConductorBus,
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
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &FractionsFolderCols<AB::Var> = (*local).borrow();
        let next: &FractionsFolderCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Loop Constraints
        ///////////////////////////////////////////////////////////////////////

        type LoopSubAir = NestedForLoopSubAir<1>;

        LoopSubAir {}.eval(
            builder,
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
        );

        let is_transition = next.is_valid - next.is_first;
        let is_last = local.is_valid - is_transition.clone();

        // Air index decrements by 1
        builder
            .when(is_transition.clone())
            .assert_eq(next.air_idx, local.air_idx - AB::Expr::ONE);
        // Air index ends at 0
        builder.when(is_last.clone()).assert_zero(local.air_idx);

        ///////////////////////////////////////////////////////////////////////
        // Transition Constraints
        ///////////////////////////////////////////////////////////////////////

        assert_array_eq(&mut builder.when(is_transition.clone()), local.mu, next.mu);

        builder
            .when(is_transition.clone())
            .assert_eq(next.tidx, local.tidx - AB::Expr::from_usize(2 * D_EF));

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
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.cur_hash,
            ext_field_add(
                local.sum_claim_p,
                ext_field_multiply(local.mu, local.sum_claim_q),
            ),
        );

        // Update hash
        // h' = p + mu * (q + mu * h)
        assert_array_eq(
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

        // Air index starts at num_present_airs - 1
        self.fraction_folder_input_bus.receive(
            builder,
            local.proof_idx,
            FractionFolderInputMessage {
                num_present_airs: local.air_idx + AB::Expr::ONE,
            },
            local.is_first,
        );

        // Sample mu
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_usize(2 * D_EF),
            local.mu,
            local.is_first,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_usize(D_EF),
            local.sum_claim_q,
            local.is_valid,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.sum_claim_p,
            local.is_valid,
        );

        self.sumcheck_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: AB::Expr::ZERO,
                value: local.cur_hash.map(Into::into),
            },
            is_last.clone(),
        );

        // Receive initial tidx and input layer claim from gkr module
        self.gkr_claim_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintModuleMessage {
                // Skip lambda
                tidx: local.tidx - AB::Expr::from_usize(D_EF),
                gkr_input_layer_claim: [
                    local.cur_p_sum.map(Into::into),
                    local.cur_q_sum.map(Into::into),
                ],
            },
            is_last,
        );
        // Send final tidx value to univariate sumcheck
        self.univariate_sumcheck_input_bus.send(
            builder,
            local.proof_idx,
            UnivariateSumcheckInputMessage {
                // Skip mu
                tidx: local.tidx + AB::Expr::from_usize(3 * D_EF),
            },
            local.is_first,
        );

        self.mu_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Mu.to_field(),
                idx: AB::Expr::ZERO,
                value: local.mu.map(Into::into),
            },
            local.is_first,
        );
    }
}
