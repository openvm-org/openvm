use std::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, EqSharpUniBus, EqSharpUniMessage, EqZeroNBus,
        EqZeroNMessage,
    },
    bus::{XiRandomnessBus, XiRandomnessMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{
        base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_one_minus, ext_field_subtract,
    },
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqSharpUniCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub proof_idx: T,

    pub root: T,
    pub root_pow: T,
    pub root_half_order: T,

    pub xi_idx: T,
    pub xi: [T; D_EF],
    pub product_before: [T; D_EF],

    pub iter_idx: T,
    pub is_first_iter: T,
}

pub struct EqSharpUniAir {
    pub xi_bus: XiRandomnessBus,
    pub eq_bus: EqSharpUniBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,
    pub l_skip: usize,
    pub canonical_inverse_generator: F,
}

impl<F> BaseAirWithPublicValues<F> for EqSharpUniAir {}
impl<F> PartitionedBaseAir<F> for EqSharpUniAir {}

impl<F> BaseAir<F> for EqSharpUniAir {
    fn width(&self) -> usize {
        EqSharpUniCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqSharpUniAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
    AB::Expr: From<F>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &EqSharpUniCols<AB::Var> = (*local).borrow();
        let next: &EqSharpUniCols<AB::Var> = (*next).borrow();

        // Summary:
        // - iter_idx consistency: the nested-loop sub-AIR models proof -> round -> iter loops. Here
        //   round_idx is derived from xi_idx, and iter uses is_first=is_valid so transitions force
        //   iter_idx += 1 within enabled rounds while allowing wrap at round boundaries. The first
        //   valid row initializes is_first_iter and iter_idx; invalid rows force iter_idx = 0.
        // - Root consistency: continuing iterations multiply `root_pow` by `root` while preserving
        //   `root` and its half order; when a wrap happens, the root squares and the half order
        //   doubles; the first row fixes `root = -1`, `root_half_order = 1`, and `root_pow = 1`,
        //   and the final row forces the root to equal `canonical_inverse_generator`.
        // - Xi/product consistency: the initial product equals one and `xi_idx` starts at `l_skip -
        //   1`, wraps decrement `xi_idx` until reaching zero on the last row, and bus
        //   communications consume and emit products based on `xi`, `1 - xi`, and `xi * root_pow`
        //   combinations.

        let local_round_idx = AB::Expr::from_usize(self.l_skip - 1) - local.xi_idx;
        let next_round_idx = AB::Expr::from_usize(self.l_skip - 1) - next.xi_idx;

        type LoopSubAir = NestedForLoopSubAir<3>;
        LoopSubAir {}.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_valid.into(),
                    counter: [
                        local.proof_idx.into(),
                        local_round_idx,
                        local.iter_idx.into(),
                    ],
                    is_first: [
                        local.is_first.into(),
                        local.is_first_iter.into(),
                        local.is_valid.into(),
                    ],
                },
                NestedForLoopIoCols {
                    is_enabled: next.is_valid.into(),
                    counter: [next.proof_idx.into(), next_round_idx, next.iter_idx.into()],
                    is_first: [
                        next.is_first.into(),
                        next.is_first_iter.into(),
                        next.is_valid.into(),
                    ],
                },
            ),
        );

        let local_is_last = LoopSubAir::local_is_last(local.is_valid, next.is_valid, next.is_first);

        // =========================== idx consistency =============================
        // NestedForLoopSubAir enforces proof/round/iter nested-loop shape; remaining constraints
        // pin initialization and wrap semantics specific to this AIR.
        builder
            .when(local.is_first_iter)
            .assert_zero(local.iter_idx);

        // Either iter_idx becomes zero, or increases by one.
        builder.assert_zero(next.iter_idx * (next.iter_idx - local.iter_idx - AB::Expr::ONE));
        // iter_idx is always zero on invalid rows
        builder
            .when(not(local.is_valid))
            .assert_zero(local.iter_idx);
        // If becomes zero, then it would have become root_half_order
        builder
            .when(LoopSubAir::local_is_last(
                local.is_valid,
                next.is_valid,
                next.is_first_iter,
            ))
            .assert_eq(local.iter_idx + AB::Expr::ONE, local.root_half_order);
        // =========================== Root consistency =============================
        // If iter_idx doesn't become zero (increases by one), then:
        // - root_pow is multiplied by root,
        // - root is preserved.
        builder
            .when(next.iter_idx)
            .assert_eq(next.root_pow, local.root_pow * local.root);
        builder.when(next.iter_idx).assert_eq(local.root, next.root);
        builder
            .when(next.iter_idx)
            .assert_eq(local.root_half_order, next.root_half_order);
        // Otherwise the new root is the square of the current one, and the root order doubles.
        builder
            .when(next.is_first_iter * not(next.is_first))
            .assert_eq(local.root_half_order * AB::Expr::TWO, next.root_half_order);
        builder
            .when(next.is_first_iter * not(next.is_first))
            .assert_eq(local.root, next.root * next.root);
        // Important: we need to enforce dropping iter_idx to zero if root_pow is going to become
        // one. We can leave this to inner interactions, but then we need to guarantee that
        // the first layer has size 1.
        builder
            .when(next.is_valid * local.is_first)
            .assert_one(next.is_first_iter);
        // Also, on the first row we have some conditions
        builder
            .when(local.is_valid * local.is_first)
            .assert_eq(local.root, AB::Expr::NEG_ONE);
        builder
            .when(local.is_valid * local.is_first)
            .assert_eq(local.root_half_order, AB::Expr::ONE);
        // If is_first_iter, then root_pow is 1
        builder
            .when(local.is_first_iter)
            .assert_eq(local.root_pow, AB::Expr::ONE);
        // Finally, the final root must equal some specific generator
        builder
            .when(local.is_valid * local_is_last.clone())
            .assert_eq::<AB::Var, AB::Expr>(local.root, self.canonical_inverse_generator.into());

        // =========================== Xi and product consistency =============================
        // Boundary conditions
        assert_array_eq(
            &mut builder.when(local.is_valid * local.is_first),
            local.product_before,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        builder.when(local.is_valid * local.is_first).assert_eq(
            local.xi_idx,
            AB::Expr::from_usize(self.l_skip) - AB::Expr::ONE,
        );
        assert_array_eq(
            &mut builder
                .when(local.is_valid)
                .when(next.is_valid)
                .when(not(next.is_first_iter)),
            local.xi,
            next.xi,
        );
        // When we drop iter_idx, xi_idx decreases
        builder
            .when(next.is_first_iter * not(next.is_first))
            .assert_one(local.xi_idx - next.xi_idx);
        // The last one must be 0
        builder
            .when(local_is_last.clone())
            .assert_eq(local.xi_idx, AB::Expr::ZERO);

        self.xi_bus.receive(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.xi_idx,
                xi: local.xi,
            },
            local.is_first_iter,
        );
        // Here idx < l_skip and all idx are different within one proof_idx
        self.batch_constraint_conductor_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: local.xi_idx.into(),
                value: local.xi.map(|x| x.into()),
            },
            local.is_valid * local_is_last,
        );

        self.eq_bus.receive(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: local.xi_idx + AB::Expr::ONE,
                iter_idx: local.iter_idx.into(),
                product: local.product_before.map(|x| x.into()),
            },
            local.is_valid * (AB::Expr::ONE - local.is_first),
        );
        let second: [AB::Expr; D_EF] = ext_field_multiply_scalar(local.xi, local.root_pow);
        let one_minus_xi: [AB::Expr; D_EF] = ext_field_one_minus(local.xi);

        self.eq_bus.send(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: local.xi_idx.into(),
                iter_idx: local.iter_idx.into(),
                product: ext_field_multiply(
                    local.product_before,
                    ext_field_add::<AB::Expr>(one_minus_xi.clone(), second.clone()),
                ),
            },
            local.is_valid,
        );
        self.eq_bus.send(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: local.xi_idx.into(),
                iter_idx: local.iter_idx + local.root_half_order,
                product: ext_field_multiply(
                    local.product_before,
                    ext_field_subtract::<AB::Expr>(one_minus_xi.clone(), second.clone()),
                ),
            },
            local.is_valid,
        );
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqSharpUniReceiverCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub is_last: T,
    pub proof_idx: T,

    pub idx: T,
    pub coeff: [T; D_EF],
    pub r: [T; D_EF],
    pub cur_sum: [T; D_EF],
}

pub struct EqSharpUniReceiverAir {
    pub r_bus: BatchConstraintConductorBus,
    pub eq_bus: EqSharpUniBus,
    pub zero_n_bus: EqZeroNBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqSharpUniReceiverAir {}
impl<F> PartitionedBaseAir<F> for EqSharpUniReceiverAir {}

impl<F> BaseAir<F> for EqSharpUniReceiverAir {
    fn width(&self) -> usize {
        EqSharpUniReceiverCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqSharpUniReceiverAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &EqSharpUniReceiverCols<AB::Var> = (*local).borrow();
        let next: &EqSharpUniReceiverCols<AB::Var> = (*next).borrow();

        // Summary:
        // - idx consistency: start with `idx = 0` on the first row and increment by one on each
        //   subsequent non-header row while the proof continues.
        // - EF values consistency: keep the `r` coefficients constant across transitions and update
        //   `cur_sum` via `coeff + r * next.cur_sum` on every step past the first.

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

        builder.assert_bool(local.is_last);
        let is_same_proof = next.is_valid - next.is_first;

        // ============================= idx consistency ============================
        builder.when(local.is_first).assert_zero(local.idx);
        builder
            .when(is_same_proof.clone())
            .assert_one(next.idx - local.idx);
        // ============================= EF values consistency ==========================
        assert_array_eq(&mut builder.when(is_same_proof.clone()), next.r, local.r);
        assert_array_eq(
            &mut builder
                .when(local.is_valid)
                .when(not::<AB::Expr>(is_same_proof.clone())),
            local.cur_sum,
            local.coeff,
        );
        assert_array_eq(
            &mut builder.when(is_same_proof),
            local.cur_sum,
            ext_field_add(local.coeff, ext_field_multiply(local.r, next.cur_sum)),
        );

        self.r_bus.lookup_key(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: AB::Expr::ZERO,
                value: local.r.map(|x| x.into()),
            },
            local.is_valid * local.is_first,
        );
        self.eq_bus.receive(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: AB::Expr::ZERO,
                iter_idx: local.idx.into(),
                product: local.coeff.map(|x| x.into()),
            },
            local.is_valid,
        );
        let l_skip_inv: AB::Expr = AB::F::from_usize(1 << self.l_skip).inverse().into();
        self.zero_n_bus.send(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ONE,
                value: local.cur_sum.map(|x| l_skip_inv.clone() * x),
            },
            local.is_valid * local.is_first,
        );
    }
}
