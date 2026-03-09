use std::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{and, assert_array_eq, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, TranscriptBus,
        TranscriptBusMessage, WhirOpeningPointBus, WhirOpeningPointMessage,
    },
    stacking::bus::{
        EqBaseBus, EqBaseMessage, EqKernelLookupBus, EqKernelLookupMessage, EqRandValuesLookupBus,
        EqRandValuesLookupMessage, StackingModuleTidxBus, StackingModuleTidxMessage,
        SumcheckClaimsBus, SumcheckClaimsMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{
        assert_zeros, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_one_minus, ext_field_subtract,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SumcheckRoundsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Sumcheck round this row represents
    pub round: F,

    // Starting tidx for this sumcheck round
    pub tidx: F,

    // Evaluations of polynomial s_round
    pub s_eval_at_0: [F; D_EF],
    pub s_eval_at_1: [F; D_EF],
    pub s_eval_at_2: [F; D_EF],
    pub s_eval_at_u: [F; D_EF],

    // Values of sampled u and r for this round
    pub u_round: [F; D_EF],
    pub r_round: [F; D_EF],
    pub has_r: F,
    pub u_mult: F,

    // Values of eq(u_0, r_0), eq(u_0, r_0 * omega), and eq(u_0, 1) * eq(r_0 * omega, 1)
    pub eq_prism_base: [F; D_EF],
    pub eq_cube_base: [F; D_EF],
    pub rot_cube_base: [F; D_EF],

    // Value of eq_cube
    pub eq_cube: [F; D_EF],

    // Intermediate values to compute rot_cube recursively
    pub r_not_u_prod: [F; D_EF],
    pub rot_cube_minus_prod: [F; D_EF],

    // Multiplicity of eq_round(u, r) lookup
    pub eq_rot_mult: F,
}

pub struct SumcheckRoundsAir {
    // External buses
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
    pub eq_base_bus: EqBaseBus,
    pub eq_rand_values_bus: EqRandValuesLookupBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,

    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for SumcheckRoundsAir {}
impl PartitionedBaseAir<F> for SumcheckRoundsAir {}

impl<F> BaseAir<F> for SumcheckRoundsAir {
    fn width(&self) -> usize {
        SumcheckRoundsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for SumcheckRoundsAir
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

        let local: &SumcheckRoundsCols<AB::Var> = (*local).borrow();
        let next: &SumcheckRoundsCols<AB::Var> = (*next).borrow();

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
         * Constrain that round increments correctly.
         */
        builder.when(local.is_first).assert_one(local.round);
        builder
            .when(and(not(local.is_last), local.is_valid))
            .assert_eq(local.round + AB::Expr::ONE, next.round);

        /*
         * Constrain that s_round(u_round) is the quadratic interpolation using values
         * s_round(0), s_round(1), and s_round(2). Additionally, constrain that we have
         * s_round(u_round) = s_{round + 1}(0) + s_{round + 1}(1), and send the value of
         * s_{n_stack}(u_{n_stack}) to StackingClaimsAir.
         */
        self.sumcheck_claims_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ONE,
                value: ext_field_add(local.s_eval_at_0, local.s_eval_at_1),
            },
            local.is_first,
        );

        let s1 = ext_field_subtract(local.s_eval_at_1, local.s_eval_at_0);
        let s2 = ext_field_subtract(local.s_eval_at_2, local.s_eval_at_1);
        let p = ext_field_multiply_scalar::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(s2, s1.clone()),
            AB::F::TWO.inverse(),
        );
        let q = ext_field_subtract::<AB::Expr>(s1, p.clone());

        assert_array_eq(
            builder,
            ext_field_add(
                ext_field_multiply(
                    ext_field_add::<AB::Expr>(ext_field_multiply(p, local.u_round), q),
                    local.u_round,
                ),
                local.s_eval_at_0,
            ),
            local.s_eval_at_u,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.s_eval_at_u,
            ext_field_add(next.s_eval_at_0, next.s_eval_at_1),
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::TWO,
                value: local.s_eval_at_u.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Constrain the correctness of eq_cube and rot_cube at each round and provide the
         * lookups for eq_round(u, r) and k_rot_round(u, r). Computing rot_cube recursively
         * requires us to store the prefix product of r_round * (1 - u_round), which we
         * denote r_not_u_prod, and rot_cube - r_not_u_prod.
         */
        self.eq_base_bus.receive(
            builder,
            local.proof_idx,
            EqBaseMessage {
                eq_u_r: local.eq_prism_base,
                eq_u_r_omega: local.eq_cube_base,
                eq_u_r_prod: local.rot_cube_base,
            },
            local.is_first,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.eq_prism_base,
            next.eq_prism_base,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.eq_cube_base,
            next.eq_cube_base,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.rot_cube_base,
            next.rot_cube_base,
        );

        let local_u_not_r = ext_field_multiply(local.u_round, ext_field_one_minus(local.r_round));
        let local_r_not_u = ext_field_multiply(local.r_round, ext_field_one_minus(local.u_round));
        let next_u_not_r = ext_field_multiply(next.u_round, ext_field_one_minus(next.r_round));
        let next_r_not_u = ext_field_multiply(next.r_round, ext_field_one_minus(next.u_round));

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_one_minus::<AB::Expr>(ext_field_add::<AB::Expr>(
                local_u_not_r.clone(),
                local_r_not_u.clone(),
            )),
            local.eq_cube,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(
                local.eq_cube,
                ext_field_one_minus::<AB::Expr>(ext_field_add::<AB::Expr>(
                    next_u_not_r.clone(),
                    next_r_not_u.clone(),
                )),
            ),
            next.eq_cube,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.r_not_u_prod,
            local_r_not_u,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.r_not_u_prod, next_r_not_u.clone()),
            next.r_not_u_prod,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.rot_cube_minus_prod,
            local_u_not_r,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_add::<AB::Expr>(
                ext_field_multiply(
                    local.rot_cube_minus_prod,
                    ext_field_one_minus::<AB::Expr>(ext_field_add::<AB::Expr>(
                        next_u_not_r.clone(),
                        next_r_not_u,
                    )),
                ),
                ext_field_multiply(next_u_not_r, local.r_not_u_prod),
            ),
            next.rot_cube_minus_prod,
        );

        self.eq_kernel_lookup_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            EqKernelLookupMessage {
                n: local.round.into(),
                eq_in: ext_field_multiply(local.eq_prism_base, local.eq_cube),
                k_rot_in: ext_field_add(
                    ext_field_multiply(local.eq_cube_base, local.eq_cube),
                    ext_field_multiply(
                        local.rot_cube_base,
                        ext_field_subtract(
                            ext_field_add(local.r_not_u_prod, local.rot_cube_minus_prod),
                            local.eq_cube,
                        ),
                    ),
                ),
            },
            local.is_valid * local.eq_rot_mult,
        );

        builder.assert_bool(local.has_r);
        builder
            .when(and(local.is_valid, not(local.is_last)))
            .assert_bool(local.has_r - next.has_r);
        builder
            .when(not(local.has_r))
            .assert_zero(local.eq_rot_mult);

        assert_zeros(&mut builder.when(not(local.has_r)), local.r_round);

        /*
         * Because we sample u_round and r_round from the transcript here, we send
         * them to other AIRs that need to use it.
         */
        self.constraint_randomness_bus.receive(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: local.round,
                challenge: local.r_round,
            },
            and(local.is_valid, local.has_r),
        );

        self.whir_opening_point_bus.send(
            builder,
            local.proof_idx,
            WhirOpeningPointMessage {
                idx: local.round + AB::Expr::from_usize(self.l_skip - 1),
                value: local.u_round.map(Into::into),
            },
            local.is_valid,
        );

        self.eq_rand_values_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            EqRandValuesLookupMessage {
                idx: local.round,
                u: local.u_round,
            },
            local.u_mult,
        );
        builder.when(not(local.is_valid)).assert_zero(local.u_mult);

        /*
         * Constrain transcript operations and send the final tidx to StackingClaimsAir.
         */
        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ONE,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        builder
            .when(not(local.is_last) * local.is_valid)
            .assert_eq(local.tidx + AB::F::from_usize(3 * D_EF), next.tidx);

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i) + local.tidx,
                    value: local.s_eval_at_1[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i + D_EF) + local.tidx,
                    value: local.s_eval_at_2[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i + 2 * D_EF) + local.tidx,
                    value: local.u_round[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_valid,
            );
        }

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::TWO,
                tidx: AB::Expr::from_usize(3 * D_EF) + local.tidx,
            },
            and(local.is_last, local.is_valid),
        );
    }
}
