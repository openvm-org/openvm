use std::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{and, assert_array_eq, not},
    SubAir,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;

use crate::{
    bus::{TranscriptBus, TranscriptBusMessage},
    stacking::bus::{
        EqKernelLookupBus, EqRandValuesLookupBus, EqRandValuesLookupMessage, StackingModuleTidxBus,
        StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{assert_one_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UnivariateRoundCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Sampled transcript values
    pub tidx: F,
    pub u_0: [F; D_EF],
    pub u_0_pow: [F; D_EF],

    // Coefficients of univariate round (s_0) polynomial
    pub coeff: [F; D_EF],

    // Columns to compute s_0(z) sum over all z in D
    pub coeff_idx: F,
    pub coeff_is_d: F,
    pub s_0_sum_over_d: [F; D_EF],

    // Evaluation of s_0 polynomial at u_0
    pub poly_rand_eval: [F; D_EF],
}

pub struct UnivariateRoundAir {
    // External buses
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
    pub eq_rand_values_bus: EqRandValuesLookupBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,

    // Other fields
    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for UnivariateRoundAir {}
impl PartitionedBaseAir<F> for UnivariateRoundAir {}

impl<F> BaseAir<F> for UnivariateRoundAir {
    fn width(&self) -> usize {
        UnivariateRoundCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UnivariateRoundAir
where
    AB::F: PrimeField32,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &UnivariateRoundCols<AB::Var> = (*local).borrow();
        let next: &UnivariateRoundCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1> {}.eval(
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

        builder.when(local.is_valid).assert_eq(
            local.is_last,
            NestedForLoopSubAir::<1>::local_is_last(local.is_valid, next.is_valid, next.is_first),
        );

        builder.assert_bool(local.is_last);
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_zero((local.proof_idx + AB::F::ONE - next.proof_idx) * next.proof_idx);
        builder
            .when(and(not(local.is_valid), local.is_last))
            .assert_zero(next.proof_idx);

        /*
         * Constrain that the sum of s_0(z) for z in D via interaction equals the RLC of column
         * claims from OpeningClaimsAir. We use the properties of D to do this efficiently -
         * since D is a multiplicative subgroup, it turns out this sum is |D| * (a_0 + a_{|D|}).
         */
        let d_card = 1usize << self.l_skip;

        builder.when(local.is_first).assert_zero(local.coeff_idx);
        builder
            .when(and(local.is_last, local.is_valid))
            .assert_eq(local.coeff_idx, AB::F::from_usize(2 * (d_card - 1)));
        builder
            .when(and(not(local.is_last), local.is_valid))
            .assert_one(next.coeff_idx - local.coeff_idx);

        builder.assert_bool(local.coeff_is_d);
        builder.when(local.coeff_is_d).assert_one(local.is_valid);
        builder
            .when(local.coeff_is_d)
            .assert_eq(local.coeff_idx, AB::F::from_usize(d_card));

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_multiply_scalar(local.coeff, AB::F::from_usize(d_card)),
            local.s_0_sum_over_d,
        );

        assert_array_eq(
            &mut builder.when(next.coeff_is_d),
            ext_field_add(
                local.s_0_sum_over_d,
                ext_field_multiply_scalar(next.coeff, AB::F::from_usize(d_card)),
            ),
            next.s_0_sum_over_d,
        );

        self.sumcheck_claims_bus.receive(
            builder,
            next.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ZERO,
                value: next.s_0_sum_over_d.map(Into::into),
            },
            next.coeff_is_d,
        );

        /*
         * Compute evaluation of polynomial s_0(u_0) and send it to SumcheckRoundsAir, where
         * it'll be used to constrain the correctness of s_1(0).
         */
        assert_one_ext(&mut builder.when(local.is_first), local.u_0_pow);
        assert_array_eq(&mut builder.when(not(local.is_last)), local.u_0, next.u_0);

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.u_0, local.u_0_pow),
            next.u_0_pow,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_multiply(local.coeff, local.u_0_pow),
            local.poly_rand_eval,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_add(
                local.poly_rand_eval,
                ext_field_multiply(next.coeff, next.u_0_pow),
            ),
            next.poly_rand_eval,
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ONE,
                value: local.poly_rand_eval.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Because we sample u_0 from the transcript here, we send u_0 to other AIRs that
         * need to use it.
         */
        self.eq_rand_values_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            EqRandValuesLookupMessage {
                idx: AB::Expr::ZERO,
                u: local.u_0.map(Into::into),
            },
            and(local.is_last, local.is_valid) * AB::F::TWO,
        );

        /*
         * Constrain transcript operations and send the final tidx to SumcheckRoundsAir.
         */
        let mut when_same_proof = builder.when(and(local.is_valid, not(local.is_last)));
        when_same_proof.assert_one(next.is_valid);
        when_same_proof.assert_eq(local.tidx + AB::Expr::from_usize(D_EF), next.tidx);

        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ZERO,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i) + local.tidx,
                    value: local.coeff[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i + D_EF) + local.tidx,
                    value: local.u_0[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                and(local.is_last, local.is_valid),
            );
        }

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ONE,
                tidx: AB::Expr::from_usize(2 * D_EF) + local.tidx,
            },
            and(local.is_last, local.is_valid),
        );
    }
}
