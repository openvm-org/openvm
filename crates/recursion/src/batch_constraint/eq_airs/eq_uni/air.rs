use std::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    SubAir,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, EqZeroNBus, EqZeroNMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{assert_one_ext, ext_field_add, ext_field_multiply, ext_field_one_minus},
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqUniCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub proof_idx: T,

    pub x: [T; D_EF],
    pub y: [T; D_EF],
    pub res: [T; D_EF],
    pub idx: T,
}

pub struct EqUniAir {
    pub zero_n_bus: EqZeroNBus,
    pub r_xi_bus: BatchConstraintConductorBus,
    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqUniAir {}
impl<F> PartitionedBaseAir<F> for EqUniAir {}

impl<F> BaseAir<F> for EqUniAir {
    fn width(&self) -> usize {
        EqUniCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqUniAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &EqUniCols<AB::Var> = (*local).borrow();
        let next: &EqUniCols<AB::Var> = (*next).borrow();

        // Summary:
        // - Proof loop: rely on the nested for-loop sub-AIR to enforce the standard
        //   `is_valid`/`is_first`/`proof_idx` sequencing across proofs.
        // - idx handling: start `idx` at zero, increment it on each transition within the proof,
        //   keep it zero on invalid rows, and ensure the last valid row reaches `l_skip`.
        // - Values recalculation: during transitions, square both `x` and `y`, and update `res` via
        //   `(x + y) * res + (1 - x) * (1 - y)`.

        // Enforce the standard proof loop flags (is_valid/is_first/proof_idx).
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

        let local_is_last = next.is_first + not(next.is_valid);
        builder.assert_bool(local_is_last.clone());

        self.r_xi_bus.lookup_key(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: AB::Expr::ZERO,
                value: local.x.map(|x| x.into()),
            },
            local.is_valid * local.is_first,
        );
        self.r_xi_bus.lookup_key(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: AB::Expr::ZERO,
                value: local.y.map(|x| x.into()),
            },
            local.is_valid * local.is_first,
        );

        let inv_p2 = AB::F::ONE.halve().exp_u64(self.l_skip as u64);
        self.zero_n_bus.send(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ZERO,
                value: local.res.map(|x| x * inv_p2),
            },
            local.is_valid * local_is_last.clone(),
        );

        let is_transition = next.is_valid * (AB::Expr::ONE - next.is_first);

        // ======================== idx handling ==========================
        builder.when(local.is_first).assert_zero(local.idx);
        builder
            .when(is_transition.clone())
            .assert_eq(next.idx, local.idx + AB::Expr::ONE);
        builder.when(not(local.is_valid)).assert_zero(local.idx);
        builder
            .when(local_is_last)
            .when(local.is_valid)
            .assert_eq(local.idx, AB::Expr::from_usize(self.l_skip));

        // ======================== Values recalculation ==========================
        assert_one_ext(&mut builder.when(local.is_first), local.res);

        let mut when_transition = builder.when(is_transition);
        assert_array_eq(
            &mut when_transition,
            next.x.map(Into::into),
            ext_field_multiply::<AB::Expr>(local.x, local.x),
        );
        assert_array_eq(
            &mut when_transition,
            next.y.map(Into::into),
            ext_field_multiply::<AB::Expr>(local.y, local.y),
        );

        let x_plus_y = ext_field_add::<AB::Expr>(local.x, local.y);
        let one_minus_x = ext_field_one_minus::<AB::Expr>(local.x);
        let one_minus_y = ext_field_one_minus::<AB::Expr>(local.y);
        let next_res_expected = ext_field_add::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(x_plus_y, local.res),
            ext_field_multiply::<AB::Expr>(one_minus_x, one_minus_y),
        );
        assert_array_eq(
            &mut when_transition,
            next.res.map(Into::into),
            next_res_expected,
        );
    }
}
