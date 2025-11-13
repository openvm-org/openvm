use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    SubAir,
    utils::{assert_array_eq, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, extension::BinomiallyExtendable};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, EqZeroNBus, EqZeroNMessage,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{ext_field_add, ext_field_multiply, ext_field_one_minus},
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqUniCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    x: [T; D_EF],
    y: [T; D_EF],
    res: [T; D_EF],
    idx: T,
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
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqUniCols<AB::Var> = (*local).borrow();
        let next: &EqUniCols<AB::Var> = (*next).borrow();

        // Summary:
        // - Proof loop: rely on the nested for-loop sub-AIR to enforce the standard
        //   `is_valid`/`is_first`/`is_last`/`proof_idx` sequencing across proofs.
        // - idx handling: start `idx` at zero, increment it on each transition within the proof,
        //   keep it zero on invalid rows, and ensure the last valid row reaches `l_skip`.
        // - Values recalculation: during transitions, square both `x` and `y`, and update `res` via
        //   `(x + y) * res + (1 - x) * (1 - y)`.

        // Enforce the standard proof loop flags (is_valid/is_first/is_last/proof_idx).
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

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);

        self.r_xi_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: AB::Expr::ZERO,
                value: local.x.map(|x| x.into()),
            },
            local.is_valid * local.is_first,
        );
        self.r_xi_bus.receive(
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
            local.is_valid * local.is_last,
        );

        let is_transition = next.is_valid * (AB::Expr::ONE - next.is_first);

        // ======================== idx handling ==========================
        builder.when(local.is_first).assert_zero(local.idx);
        builder
            .when(is_transition.clone())
            .assert_eq(next.idx, local.idx + AB::Expr::ONE);
        builder.when(not(local.is_valid)).assert_zero(local.idx);
        builder
            .when(local.is_last)
            .when(local.is_valid)
            .assert_eq(local.idx, AB::Expr::from_canonical_usize(self.l_skip));

        // ======================== Values recalculation ==========================
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

#[tracing::instrument(name = "generate_trace(EqUniAir)", skip_all)]
pub(crate) fn generate_eq_uni_trace(
    vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = EqUniCols::<F>::width();
    let l_skip = vk.inner.params.l_skip;
    let one_height = l_skip + 1;
    let total_height = one_height * preflights.len();
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    for (pidx, preflight) in preflights.iter().enumerate() {
        let mut x = preflight.batch_constraint.xi[0];
        let mut y = preflight.batch_constraint.sumcheck_rnd[0];
        let mut res = EF::ONE;
        trace[pidx * one_height * width..(pidx + 1) * one_height * width]
            .chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut EqUniCols<_> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.is_first = F::from_bool(i == 0);
                cols.is_last = F::from_bool(i + 1 == one_height);

                cols.idx = F::from_canonical_usize(i);
                cols.x.copy_from_slice(x.as_base_slice());
                cols.y.copy_from_slice(y.as_base_slice());
                cols.res.copy_from_slice(res.as_base_slice());

                res = (x + y) * res + (EF::ONE - x) * (EF::ONE - y);
                x *= x;
                y *= y;
            });
    }

    trace[total_height * width..]
        .chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut EqUniCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
