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
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        AirShapeBus, AirShapeBusMessage, AirShapeProperty, ColumnClaimsBus, ColumnClaimsMessage,
        LiftedHeightsBus, LiftedHeightsBusMessage, StackingModuleBus, StackingModuleMessage,
        TranscriptBus, TranscriptBusMessage,
    },
    stacking::bus::{
        ClaimCoefficientsBus, ClaimCoefficientsMessage, EqBitsLookupBus, EqBitsLookupMessage,
        EqKernelLookupBus, EqKernelLookupMessage, StackingModuleTidxBus, StackingModuleTidxMessage,
        SumcheckClaimsBus, SumcheckClaimsMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{assert_one_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct OpeningClaimsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Received from batch constraints module
    pub sort_idx: F,
    pub part_idx: F,
    pub col_idx: F,
    pub col_claim: [F; D_EF],
    pub rot_claim: [F; D_EF],
    pub need_rot: F,

    // Used to constrain the order of received messages
    pub is_main: F,
    pub is_transition_main: F,

    // From proof shape (n, n_lift + l_skip, 2^{n_lift + l_skip}, 2^{- (n_lift + l_skip)})
    pub hypercube_dim: F,
    pub log_lifted_height: F,
    pub lifted_height: F,
    pub lifted_height_inv: F,

    // Sampled transcript values
    pub tidx: F,
    pub lambda: [F; D_EF],
    pub lambda_pow: [F; D_EF],

    // Location in stacked matrices
    pub commit_idx: F,
    pub stacked_col_idx: F,
    pub row_idx: F,
    pub is_last_for_claim: F,

    // Lookups to compute claim coefficient
    pub eq_in: [F; D_EF],
    pub k_rot_in: [F; D_EF],
    pub eq_bits: [F; D_EF],

    // This is either `k_rot_in` or zero, depending on `need_rot`
    pub k_rot_in_when_needed: [F; D_EF],

    // Intermediate values to compute claim coefficient
    pub lambda_pow_eq_bits: [F; D_EF],

    // Stacking claim coefficient to be sent
    pub stacking_claim_coefficient: [F; D_EF],

    // RLC of column claims * coefficient using lambda
    pub s_0: [F; D_EF],
}

pub struct OpeningClaimsAir {
    // External buses
    pub lifted_heights_bus: LiftedHeightsBus,
    pub stacking_module_bus: StackingModuleBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub transcript_bus: TranscriptBus,
    pub air_shape_bus: AirShapeBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub claim_coefficients_bus: ClaimCoefficientsBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,
    pub eq_bits_lookup_bus: EqBitsLookupBus,

    // Other fields
    pub n_stack: usize,
    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for OpeningClaimsAir {}
impl PartitionedBaseAir<F> for OpeningClaimsAir {}

impl<F> BaseAir<F> for OpeningClaimsAir {
    fn width(&self) -> usize {
        OpeningClaimsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for OpeningClaimsAir
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

        let local: &OpeningClaimsCols<AB::Var> = (*local).borrow();
        let next: &OpeningClaimsCols<AB::Var> = (*next).borrow();

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

        builder.assert_bool(local.is_last);
        builder.assert_bool(local.need_rot);
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_zero((local.proof_idx + AB::F::ONE - next.proof_idx) * next.proof_idx);
        builder
            .when(and(not(local.is_valid), local.is_last))
            .assert_zero(next.proof_idx);

        /*
         * Constrain the sortedness of each ColumnClaimsMessage. Main claims (i.e part_idx = 0)
         * should be first and sorted by sort_idx and then col_idx. The remaining claims should
         * be sorted by sort_idx, then part_idx, and finally col_idx. Note that each proof must
         * have at least one main claim.
         */
        builder.assert_bool(local.is_main);
        builder.when(local.is_main).assert_one(local.is_valid);
        builder.when(local.is_main).assert_zero(local.part_idx);
        builder.when(local.is_main).assert_zero(local.commit_idx);

        builder.when(local.is_first).assert_one(local.is_main);
        builder.when(local.is_first).assert_zero(local.sort_idx);
        builder.when(local.is_first).assert_zero(local.col_idx);

        builder.assert_bool(local.is_transition_main);
        builder
            .when(local.is_transition_main)
            .assert_eq(local.is_main, next.is_main);
        builder
            .when(local.is_transition_main)
            .assert_one(and(local.is_valid, next.is_valid));
        builder
            .when(and::<AB::Expr>(
                and(local.is_main, next.is_main),
                not(local.is_last),
            ))
            .assert_one(local.is_transition_main);
        builder
            .when(and(
                and::<AB::Expr>(not(local.is_main), not(next.is_main)),
                next.is_valid,
            ))
            .assert_one(local.is_transition_main);
        builder
            .when(local.is_transition_main)
            .assert_zero(local.is_last);
        builder
            .when(and(not(local.is_main), next.is_main))
            .assert_one(local.is_last);

        let mut when_both_main = builder.when(and(local.is_main, local.is_transition_main));
        when_both_main.assert_bool(next.sort_idx - local.sort_idx);
        when_both_main
            .when_ne(local.sort_idx, next.sort_idx)
            .assert_zero(next.col_idx);
        when_both_main
            .when_ne(local.sort_idx + AB::F::ONE, next.sort_idx)
            .assert_one(next.col_idx - local.col_idx);

        let mut when_last_main = builder.when(and(local.is_main, not(local.is_transition_main)));
        when_last_main.assert_zero((next.part_idx - AB::F::ONE) * not(local.is_last));
        when_last_main.assert_zero(next.col_idx * not(local.is_last));

        /*
         * Note that we utilize the LiftedHeightsBus interaction to constrain the (sort_idx,
         * part_idx) sorting for non-main commits. Each non-zero commit_idx is constrained to
         * its exact sort_idx and part_idx, so we only need to constrain that (a) commit_idx
         * increases by 0/1 within a proof and (b) col_idx increases by 1 within a commit and
         * is reset between commits.
         */
        builder
            .when(and(local.is_valid, not(local.is_last)))
            .assert_bool(next.commit_idx - local.commit_idx);
        builder
            .when(and(local.is_transition_main, not(local.is_main)))
            .when_ne(local.commit_idx + AB::F::ONE, next.commit_idx)
            .assert_one(next.col_idx - local.col_idx);
        builder
            .when(and(local.is_transition_main, not(local.is_main)))
            .when_ne(local.commit_idx, next.commit_idx)
            .assert_zero(next.col_idx);

        /*
         * Compute col_claim[0] + lambda * rot_claim + ... (i.e. RLC of column/rotation claims)
         * and sent it to UnivariateRoundAir, which will constrain that the RLC is equal to the
         * sum of the proof's s_0 polynomial evaluated at each z in D.
         */
        self.column_claims_bus.send(
            builder,
            local.proof_idx,
            ColumnClaimsMessage {
                sort_idx: local.sort_idx.into(),
                part_idx: local.part_idx.into(),
                col_idx: local.col_idx.into(),
                claim: local.col_claim.map(Into::into),
                is_rot: AB::Expr::ZERO,
            },
            local.is_valid,
        );

        for i in 0..D_EF {
            builder
                .when(and(local.is_valid, not(local.need_rot)))
                .assert_zero(local.rot_claim[i]);
        }
        self.column_claims_bus.send(
            builder,
            local.proof_idx,
            ColumnClaimsMessage {
                sort_idx: local.sort_idx.into(),
                part_idx: local.part_idx.into(),
                col_idx: local.col_idx.into(),
                claim: local.rot_claim.map(Into::into),
                is_rot: AB::Expr::ONE,
            },
            local.is_valid * local.need_rot,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.lambda,
            next.lambda,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(
                ext_field_multiply(local.lambda, local.lambda),
                local.lambda_pow,
            ),
            next.lambda_pow,
        );

        assert_one_ext(&mut builder.when(local.is_first), local.lambda_pow);

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_add::<AB::Expr>(
                local.col_claim,
                ext_field_multiply(local.lambda, local.rot_claim),
            ),
            local.s_0,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_add(
                local.s_0,
                ext_field_add::<AB::Expr>(
                    ext_field_multiply(next.lambda_pow, next.col_claim),
                    ext_field_multiply(
                        ext_field_multiply(next.lambda_pow, next.lambda),
                        next.rot_claim,
                    ),
                ),
            ),
            next.s_0,
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ZERO,
                value: local.s_0.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Compute coefficients for stacking commits and send them to StackingClaimsAir. We do
         * this by mapping each (sort_idx, part_idx, col_idx) to their (commit_idx, col_idx,
         * row_idx) tuple. We can compute this tuple by recording the start and end point of
         * each trace slice in the commit matrix. Note that we assume that column claims are
         * properly ordered to achieve this.
         */
        builder.when(local.is_first).assert_zero(local.commit_idx);
        builder
            .when(local.is_first)
            .assert_zero(local.stacked_col_idx);
        builder.when(local.is_first).assert_zero(local.row_idx);

        builder
            .when(not(local.is_last))
            .assert_bool(next.commit_idx - local.commit_idx);
        builder
            .when(next.commit_idx - local.commit_idx)
            .assert_zero(next.stacked_col_idx);

        builder.assert_bool(local.is_last_for_claim);
        builder
            .when(next.commit_idx - local.commit_idx)
            .assert_one(local.is_last_for_claim);
        builder
            .when(next.stacked_col_idx - local.stacked_col_idx)
            .assert_one(local.is_last_for_claim);
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_one(local.is_last_for_claim);

        builder
            .when(local.is_last_for_claim)
            .assert_zero(next.row_idx);
        builder
            .when(not::<AB::Expr>(local.is_last_for_claim))
            .assert_eq(local.row_idx + local.lifted_height, next.row_idx);

        builder
            .when(and::<AB::Expr>(
                and(local.is_last_for_claim, not(local.is_last)),
                not::<AB::Expr>(next.commit_idx - local.commit_idx),
            ))
            .assert_eq(
                local.row_idx + local.lifted_height,
                AB::F::from_usize(1 << (self.n_stack + self.l_skip)),
            );

        assert_array_eq(
            builder,
            ext_field_multiply(local.lambda_pow, local.eq_bits),
            local.lambda_pow_eq_bits,
        );

        assert_array_eq(
            builder,
            local.k_rot_in_when_needed,
            ext_field_multiply_scalar(local.k_rot_in, local.need_rot),
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_multiply(
                local.lambda_pow_eq_bits,
                ext_field_add::<AB::Expr>(
                    local.eq_in,
                    ext_field_multiply(local.lambda, local.k_rot_in_when_needed),
                ),
            ),
            local.stacking_claim_coefficient,
        );

        assert_array_eq(
            &mut builder.when(local.is_last_for_claim),
            ext_field_multiply(
                next.lambda_pow_eq_bits,
                ext_field_add::<AB::Expr>(
                    next.eq_in,
                    ext_field_multiply(next.lambda, next.k_rot_in_when_needed),
                ),
            ),
            next.stacking_claim_coefficient,
        );

        assert_array_eq(
            &mut builder.when(not::<AB::Expr>(local.is_last_for_claim)),
            ext_field_add(
                local.stacking_claim_coefficient,
                ext_field_multiply(
                    next.lambda_pow_eq_bits,
                    ext_field_add::<AB::Expr>(
                        next.eq_in,
                        ext_field_multiply(next.lambda, next.k_rot_in_when_needed),
                    ),
                ),
            ),
            next.stacking_claim_coefficient,
        );

        self.claim_coefficients_bus.send(
            builder,
            local.proof_idx,
            ClaimCoefficientsMessage {
                commit_idx: local.commit_idx,
                stacked_col_idx: local.stacked_col_idx,
                coefficient: local.stacking_claim_coefficient,
            },
            and(local.is_valid, local.is_last_for_claim),
        );

        /*
         * Constrain correctness of lookup values via interactions. Heights are received
         * from ProofShapeAir, while eq(u, r), k_rot(u, r), and eq_>(u, b) values are
         * computed and provided via lookup by other stacking module AIRs.
         */
        self.lifted_heights_bus.lookup_key(
            builder,
            local.proof_idx,
            LiftedHeightsBusMessage {
                sort_idx: local.sort_idx,
                part_idx: local.part_idx,
                commit_idx: local.commit_idx,
                hypercube_dim: local.hypercube_dim,
                lifted_height: local.lifted_height,
                log_lifted_height: local.log_lifted_height,
            },
            local.is_valid,
        );

        self.eq_kernel_lookup_bus.lookup_key(
            builder,
            local.proof_idx,
            EqKernelLookupMessage {
                n: local.hypercube_dim,
                eq_in: local.eq_in,
                k_rot_in: local.k_rot_in,
            },
            local.is_valid,
        );

        builder
            .when(local.is_valid)
            .assert_one(local.lifted_height * local.lifted_height_inv);

        self.eq_bits_lookup_bus.lookup_key(
            builder,
            local.proof_idx,
            EqBitsLookupMessage {
                b_value: local.row_idx
                    * local.lifted_height_inv
                    * AB::F::from_usize(1 << self.l_skip),
                num_bits: AB::Expr::from_usize(self.n_stack + self.l_skip)
                    - local.log_lifted_height,
                eval: local.eq_bits.map(Into::into),
            },
            local.is_valid,
        );

        self.air_shape_bus.lookup_key(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sort_idx.into(),
                property_idx: AirShapeProperty::NeedRot.to_field(),
                value: local.need_rot.into(),
            },
            local.is_valid,
        );

        /*
         * Constrain transcript operations and send the final tidx to UnivariateRoundAir.
         */
        self.stacking_module_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleMessage {
                tidx: local.tidx.into(),
            },
            and(local.is_first, local.is_valid),
        );

        builder
            .when(and(local.is_valid, not(local.is_last)))
            .assert_eq(local.tidx + AB::Expr::from_usize(2 * D_EF), next.tidx);

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_usize(i),
                    value: local.col_claim[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_usize(D_EF + i),
                    value: local.rot_claim[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(2 * D_EF + i) + local.tidx,
                    value: local.lambda[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                and(local.is_last, local.is_valid),
            );
        }

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ZERO,
                tidx: local.tidx + AB::Expr::from_usize(3 * D_EF),
            },
            and(local.is_last, local.is_valid),
        );
    }
}
