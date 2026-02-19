use std::{
    borrow::{Borrow, BorrowMut},
    collections::HashMap,
};

#[cfg(all(test, feature = "cuda"))]
use itertools::Itertools;
use openvm_circuit_primitives::{
    utils::{and, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
    PrimeField32, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    stacking::{
        bus::{
            EqBitsInternalBus, EqBitsInternalMessage, EqBitsLookupBus, EqBitsLookupMessage,
            EqRandValuesLookupBus, EqRandValuesLookupMessage,
        },
        utils::get_stacked_slice_data,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    tracegen::{RowMajorChip, StandardTracegenCtx},
    utils::{
        assert_one_ext, assert_zeros, ext_field_add_scalar, ext_field_multiply,
        ext_field_multiply_scalar, ext_field_subtract, ext_field_subtract_scalar,
    },
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct EqBitsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,

    // Multiplicities of internal and external lookups
    pub internal_child_flag: F,
    pub external_mult: F,

    // Parent row's b_value and evaluation
    pub sub_b_value: F,
    pub sub_eval: [F; D_EF],

    // This row's b_value's LSB, number of bits, and u_{n_stack - num_bits + 1}
    pub b_lsb: F,
    pub num_bits: F,
    pub u_val: [F; D_EF],
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct EqBitsAir {
    // Internal buses
    pub eq_bits_internal_bus: EqBitsInternalBus,
    pub eq_bits_lookup_bus: EqBitsLookupBus,
    pub eq_rand_values_bus: EqRandValuesLookupBus,

    // Other fields
    pub n_stack: usize,
    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for EqBitsAir {}
impl PartitionedBaseAir<F> for EqBitsAir {}

impl<F> BaseAir<F> for EqBitsAir {
    fn width(&self) -> usize {
        EqBitsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqBitsAir
where
    AB::F: PrimeField32 + TwoAdicField,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &EqBitsCols<AB::Var> = (*local).borrow();
        let next: &EqBitsCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1, 0> {}.eval(
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

        /*
         * Constrain that the root evaluation is correct, i.e. eq_bits([], []) = 1.
         */
        let mut when_first = builder.when(local.is_first);

        when_first.assert_zero(local.sub_b_value);
        when_first.assert_zero(local.num_bits);
        when_first.assert_zero(local.b_lsb);

        assert_zeros(&mut when_first, local.u_val);
        assert_one_ext(&mut when_first, local.sub_eval);

        /*
         * Receive the parent b_value and eq_bits(u[0..k - 1], b[0..k - 1]) eval, as
         * well as the value of u_{n_stack - num_bits + 1}.
         */
        self.eq_rand_values_bus.lookup_key(
            builder,
            local.proof_idx,
            EqRandValuesLookupMessage {
                idx: AB::Expr::from_usize(self.n_stack + 1) - local.num_bits,
                u: local.u_val.map(Into::into),
            },
            and(not(local.is_first), local.is_valid),
        );

        self.eq_bits_internal_bus.receive(
            builder,
            local.proof_idx,
            EqBitsInternalMessage {
                b_value: local.sub_b_value.into(),
                num_bits: local.num_bits.into() - AB::Expr::ONE,
                eval: local.sub_eval.map(Into::into),
                child_lsb: local.b_lsb.into(),
            },
            and(not(local.is_first), local.is_valid),
        );

        let b_value = AB::Expr::TWO * local.sub_b_value + local.b_lsb;
        builder.assert_bool(local.b_lsb);

        /*
         * Compute eq_bits(u, b) and send it to the appropriate internal and external
         * buses. Field internal_child_flag is odd iff it has a child with lsb 0, and
         * is >= 2 iff it has a child with new bit 1. Because internal message field
         * num_bits must be strictly increasing, each (b_value, num_bits) pair must be
         * unique by induction.
         */
        let eval = ext_field_multiply(
            local.sub_eval,
            ext_field_subtract_scalar::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(
                    ext_field_add_scalar::<AB::Expr>(
                        ext_field_multiply_scalar(local.u_val, AB::Expr::TWO * local.b_lsb),
                        AB::Expr::ONE,
                    ),
                    local.u_val,
                ),
                local.b_lsb,
            ),
        );

        let three = AB::F::from_u8(3);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.internal_child_flag);
        builder
            .when_ne(local.internal_child_flag, three)
            .when_ne(local.internal_child_flag, AB::Expr::TWO)
            .assert_bool(local.internal_child_flag);

        // Multiplicity is f(x) = 1/3 * x * (x - 2) * (2x - 5), which is such that
        // f(0), f(2) = 0 and f(1), f(3) = 1.
        self.eq_bits_internal_bus.send(
            builder,
            local.proof_idx,
            EqBitsInternalMessage {
                b_value: b_value.clone(),
                num_bits: local.num_bits.into(),
                eval: eval.clone(),
                child_lsb: AB::Expr::ZERO,
            },
            local.internal_child_flag
                * (local.internal_child_flag - AB::Expr::TWO)
                * (local.internal_child_flag * AB::Expr::TWO - AB::Expr::from_u8(5))
                * three.inverse(),
        );

        // Multiplicity is f(x) = -1/6 * x * (x - 1) * (2x - 7), which is such that
        // f(0), f(1) = 0 and f(2), f(3) = 1.
        self.eq_bits_internal_bus.send(
            builder,
            local.proof_idx,
            EqBitsInternalMessage {
                b_value: b_value.clone(),
                num_bits: local.num_bits.into(),
                eval: eval.clone(),
                child_lsb: AB::Expr::ONE,
            },
            local.internal_child_flag
                * (AB::Expr::ONE - local.internal_child_flag)
                * (local.internal_child_flag * AB::Expr::TWO - AB::Expr::from_u8(7))
                * AB::F::from_u8(6).inverse(),
        );

        self.eq_bits_lookup_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            EqBitsLookupMessage {
                b_value: b_value * AB::Expr::from_usize(1 << self.l_skip),
                num_bits: local.num_bits.into(),
                eval,
            },
            local.is_valid * local.external_mult,
        );
    }
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct EqBitsTraceGenerator;

impl RowMajorChip<F> for EqBitsTraceGenerator {
    type Ctx<'a> = StandardTracegenCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let vk = ctx.vk;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        debug_assert_eq!(proofs.len(), preflights.len());

        let width = EqBitsCols::<usize>::width();

        let traces = preflights
            .par_iter()
            .enumerate()
            .map(|(proof_idx, preflight)| {
                let stacked_slices =
                    get_stacked_slice_data(vk, &preflight.proof_shape.sorted_trace_vdata);

                // (b_value, num_bits) -> (sub_eval, eval, internal_mult, external_mult)
                let mut b_value_map = HashMap::<(usize, usize), (EF, EF, usize, usize)>::new();
                let mut base_internal_mult = 0usize;
                let mut base_external_mult = 0usize;
                let u = &preflight.stacking.sumcheck_rnd[1..];

                /*
                 * Suppose we have some b_value b[0..k], where k is total_num_bits. Then
                 * eq_bits(u, b) is a function of eq_bits(u[0..k - 1], b[0..k - 1]), u[k],
                 * and b[k]. This AIR uses that property to compute each eq_bits(u, b) via
                 * a tree structure + internal interactions.
                 */
                for slice in stacked_slices {
                    let n_lift = slice.n.max(0) as usize;
                    let b_value = slice.row_idx >> (n_lift + vk.inner.params.l_skip);
                    let total_num_bits = vk.inner.params.n_stack - n_lift;

                    if total_num_bits == 0 {
                        base_external_mult += 1;
                        continue;
                    }

                    let (mut latest_eval, latest_num_bits) = {
                        let mut ret = (EF::ONE, 0);
                        for num_bits in (1..=total_num_bits).rev() {
                            let shifted_b_value = b_value >> (total_num_bits - num_bits);
                            if let Some((_, eval, internal_mult, external_mult)) =
                                b_value_map.get_mut(&(shifted_b_value, num_bits))
                            {
                                if num_bits < total_num_bits {
                                    let child_b_value = b_value >> (total_num_bits - num_bits - 1);
                                    *internal_mult += 1 + (child_b_value & 1);
                                } else {
                                    *external_mult += 1;
                                }
                                ret = (*eval, num_bits);
                                break;
                            }
                        }
                        ret
                    };

                    if latest_num_bits == total_num_bits {
                        continue;
                    } else if latest_num_bits == 0 {
                        let b_value_msb = b_value >> (total_num_bits - 1);
                        base_internal_mult += 1 + b_value_msb;
                    }

                    for num_bits in latest_num_bits + 1..=total_num_bits {
                        let shifted_b_value = b_value >> (total_num_bits - num_bits);
                        let b_lsb = EF::from_usize(shifted_b_value & 1);
                        let u_val = u[vk.inner.params.n_stack - num_bits];
                        let next_eval =
                            latest_eval * (EF::ONE + EF::TWO * b_lsb * u_val - b_lsb - u_val);
                        let is_last = num_bits == total_num_bits;
                        b_value_map.insert(
                            (shifted_b_value, num_bits),
                            (latest_eval, next_eval, !is_last as usize, is_last as usize),
                        );
                        latest_eval = next_eval;
                    }
                }

                let num_rows = b_value_map.len() + 1;
                let proof_idx_value = F::from_usize(proof_idx);

                let mut trace = vec![F::ZERO; num_rows * width];

                {
                    let first_cols: &mut EqBitsCols<F> = trace[..width].borrow_mut();
                    first_cols.proof_idx = proof_idx_value;
                    first_cols.is_valid = F::ONE;
                    first_cols.is_first = F::ONE;

                    first_cols.sub_eval[0] = F::ONE;

                    first_cols.internal_child_flag = F::from_usize(base_internal_mult);
                    first_cols.external_mult = F::from_usize(base_external_mult);
                }

                #[cfg(all(test, feature = "cuda"))]
                let b_value_iter = b_value_map.iter().sorted();
                #[cfg(any(not(test), not(feature = "cuda")))]
                let b_value_iter = b_value_map.iter();

                for ((&(b_value, num_bits), &(sub_eval, _, internal_mult, external_mult)), chunk) in
                    b_value_iter.zip(trace.chunks_mut(width).skip(1).take(b_value_map.len()))
                {
                    let cols: &mut EqBitsCols<F> = chunk.borrow_mut();
                    cols.proof_idx = proof_idx_value;
                    cols.is_valid = F::ONE;

                    cols.internal_child_flag = F::from_usize(internal_mult);
                    cols.external_mult = F::from_usize(external_mult);

                    cols.sub_b_value = F::from_usize(b_value >> 1);
                    cols.num_bits = F::from_usize(num_bits);

                    cols.b_lsb = F::from_usize(b_value & 1);
                    cols.u_val.copy_from_slice(
                        u[vk.inner.params.n_stack - num_bits].as_basis_coefficients_slice(),
                    );
                    cols.sub_eval
                        .copy_from_slice(sub_eval.as_basis_coefficients_slice());
                }

                (trace, num_rows)
            })
            .collect::<Vec<_>>();

        let num_valid_rows = traces.iter().map(|(_trace, num_rows)| *num_rows).sum();
        let height = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };

        let mut combined_trace = Vec::with_capacity(height * width);
        for (trace, _num_rows) in traces {
            combined_trace.extend(trace);
        }

        let padding_proof_idx = F::from_usize(proofs.len());
        combined_trace.resize(height * width, F::ZERO);
        for chunk in combined_trace[num_valid_rows * width..].chunks_mut(width) {
            let cols: &mut EqBitsCols<F> = chunk.borrow_mut();
            cols.proof_idx = padding_proof_idx;
        }

        Some(RowMajorMatrix::new(combined_trace, width))
    }
}
