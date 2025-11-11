use std::borrow::Borrow;

use openvm_circuit_primitives::{
    SubAir,
    is_equal::{IsEqSubAir, IsEqualAuxCols, IsEqualIo},
    utils::{assert_array_eq, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, TwoAdicField, extension::BinomiallyExtendable};
use p3_matrix::Matrix;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, SumcheckClaimBus, SumcheckClaimMessage,
        UnivariateSumcheckInputBus, UnivariateSumcheckInputMessage,
    },
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, StackingModuleBus,
        StackingModuleMessage, TranscriptBus,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
};

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct UnivariateSumcheckCols<T> {
    pub is_valid: T,
    pub proof_idx: T,
    pub is_first: T,

    pub coeff_idx: T,

    // perf(ayush): can these be preprocessed cols?
    /// Powers of generator of order 2^{l_skip} to get periodic selector columns
    pub omega_skip_power: T,
    pub is_omega_skip_power_equal_to_one: T,
    pub is_omega_skip_power_equal_to_one_aux: IsEqualAuxCols<T>,

    pub coeff: [T; D_EF],
    pub sum_at_roots: [T; D_EF],

    pub r: [T; D_EF],
    pub value_at_r: [T; D_EF],

    pub tidx: T,
}

pub struct UnivariateSumcheckAir {
    /// The univariate domain size is `2^{l_skip}`
    pub l_skip: usize,
    /// The degree of the univariate polynomial
    pub univariate_deg: usize,

    pub univariate_sumcheck_input_bus: UnivariateSumcheckInputBus,
    pub claim_bus: SumcheckClaimBus,
    pub stacking_module_bus: StackingModuleBus,
    pub transcript_bus: TranscriptBus,
    pub randomness_bus: ConstraintSumcheckRandomnessBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,
}

impl<F> BaseAirWithPublicValues<F> for UnivariateSumcheckAir {}
impl<F> PartitionedBaseAir<F> for UnivariateSumcheckAir {}

impl<F> BaseAir<F> for UnivariateSumcheckAir {
    fn width(&self) -> usize {
        UnivariateSumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UnivariateSumcheckAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &UnivariateSumcheckCols<AB::Var> = (*local).borrow();
        let next: &UnivariateSumcheckCols<AB::Var> = (*next).borrow();

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
                NestedForLoopAuxCols::default(),
            ),
        );

        let is_last = LoopSubAir::local_is_last(next.is_valid, next.is_first);
        let is_transition = AB::Expr::ONE - is_last.clone();

        // Coeff index starts at univariate degree
        builder.when(local.is_first).assert_eq(
            local.coeff_idx,
            AB::Expr::from_canonical_usize(self.univariate_deg),
        );
        // Coeff index decrements by 1
        builder
            .when(is_transition.clone())
            .assert_eq(next.coeff_idx, local.coeff_idx - AB::Expr::ONE);
        // Coeff index ends at zero
        builder.when(is_last.clone()).assert_zero(local.coeff_idx);

        ///////////////////////////////////////////////////////////////////////
        // Powers of omega constraints
        ///////////////////////////////////////////////////////////////////////

        let omega_skip = AB::Expr::from_f(<AB::Expr as FieldAlgebra>::F::two_adic_generator(
            self.l_skip,
        ));

        // Omega power ends at 1
        builder
            .when(local.is_valid * is_last.clone())
            .assert_one(local.omega_skip_power);
        // Powers of omega are calculated properly
        builder
            .when(is_transition.clone())
            .assert_eq(local.omega_skip_power, next.omega_skip_power * omega_skip);

        IsEqSubAir.eval(
            builder,
            (
                IsEqualIo::new(
                    local.omega_skip_power.into(),
                    AB::Expr::ONE,
                    local.is_omega_skip_power_equal_to_one.into(),
                    local.is_valid.into(),
                ),
                local.is_omega_skip_power_equal_to_one_aux.inv,
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Sum over Roots Constraints
        ///////////////////////////////////////////////////////////////////////

        let domain_size = AB::Expr::from_canonical_usize(1 << self.l_skip);

        // Initialize sum over roots
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.sum_at_roots,
            ext_field_multiply_scalar(local.coeff, domain_size.clone()),
        );
        // Add c * 2^{l_skip} at every 2^{l_skip} coefficient
        assert_array_eq(
            &mut builder
                .when(is_transition.clone())
                .when(next.is_omega_skip_power_equal_to_one),
            next.sum_at_roots,
            ext_field_add(
                local.sum_at_roots,
                ext_field_multiply_scalar(next.coeff, domain_size.clone()),
            ),
        );
        // Keep the sum over roots unchanged for other values
        assert_array_eq(
            &mut builder
                .when(is_transition.clone())
                .when(not(next.is_omega_skip_power_equal_to_one)),
            next.sum_at_roots,
            local.sum_at_roots,
        );

        ///////////////////////////////////////////////////////////////////////
        // Horner evaluation at r
        ///////////////////////////////////////////////////////////////////////

        assert_array_eq(&mut builder.when(is_transition.clone()), next.r, local.r);

        // Initialize evaluation
        // e = c
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.value_at_r,
            local.coeff,
        );
        // e' = c + r * e
        assert_array_eq(
            &mut builder.when(is_transition.clone()),
            next.value_at_r,
            ext_field_add(next.coeff, ext_field_multiply(next.r, local.value_at_r)),
        );

        ///////////////////////////////////////////////////////////////////////
        // Transition index
        ///////////////////////////////////////////////////////////////////////

        builder
            .when(is_transition.clone())
            .assert_eq(next.tidx, local.tidx - AB::Expr::from_canonical_usize(D_EF));

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        // Sample r
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(D_EF),
            local.r,
            local.is_valid * local.is_first,
        );
        // Observe coefficients
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.coeff,
            local.is_valid,
        );

        // Receive initial tidx value
        self.univariate_sumcheck_input_bus.receive(
            builder,
            local.proof_idx,
            UnivariateSumcheckInputMessage { tidx: local.tidx },
            local.is_valid * is_last.clone(),
        );
        // Send tidx when there are no multilinear rounds
        self.stacking_module_bus.send(
            builder,
            local.proof_idx,
            StackingModuleMessage {
                // Skip r
                tidx: local.tidx + AB::Expr::from_canonical_usize(2 * D_EF),
            },
            local.is_valid * local.is_first,
        );

        self.claim_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: AB::Expr::ZERO,
                value: local.sum_at_roots.map(Into::into),
            },
            local.is_valid * is_last.clone(),
        );
        self.claim_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: AB::Expr::ZERO,
                value: local.value_at_r.map(Into::into),
            },
            local.is_valid * is_last,
        );
        self.randomness_bus.send(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: AB::Expr::ZERO,
                challenge: local.r.map(Into::into),
            },
            local.is_valid * local.is_first,
        );

        self.batch_constraint_conductor_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: AB::Expr::ZERO,
                value: local.r.map(Into::into),
            },
            local.is_first * AB::Expr::TWO,
        );
    }
}
