use std::{
    borrow::{Borrow, BorrowMut},
    collections::HashMap,
};

#[cfg(all(test, feature = "cuda"))]
use itertools::Itertools;
use openvm_circuit_primitives::{
    SubAir,
    utils::{and, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField,
    extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
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
    system::Preflight,
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
    pub internal_mult: F,
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
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct EqBitsTraceGenerator;

impl EqBitsTraceGenerator {
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn generate_trace(
        vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> RowMajorMatrix<F> {
        debug_assert_eq!(proofs.len(), preflights.len());

        let width = EqBitsCols::<usize>::width();

        if proofs.is_empty() {
            return RowMajorMatrix::new(vec![F::ZERO; width], width);
        }

        let mut combined_trace = Vec::<F>::new();
        let mut total_rows = 0usize;

        for (proof_idx, preflight) in preflights.iter().enumerate() {
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
                                *internal_mult += 1;
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
                    base_internal_mult += 1;
                }

                for num_bits in latest_num_bits + 1..=total_num_bits {
                    let shifted_b_value = b_value >> (total_num_bits - num_bits);
                    let b_lsb = EF::from_canonical_usize(shifted_b_value & 1);
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
            let proof_idx_value = F::from_canonical_usize(proof_idx);

            let mut trace = vec![F::ZERO; num_rows * width];

            for chunk in trace.chunks_mut(width) {
                let cols: &mut EqBitsCols<F> = chunk.borrow_mut();
                cols.proof_idx = proof_idx_value;
            }

            {
                let first_cols: &mut EqBitsCols<F> = trace[..width].borrow_mut();
                first_cols.is_valid = F::ONE;
                first_cols.is_first = F::ONE;

                first_cols.sub_eval[0] = F::ONE;

                first_cols.internal_mult = F::from_canonical_usize(base_internal_mult);
                first_cols.external_mult = F::from_canonical_usize(base_external_mult);
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

                cols.internal_mult = F::from_canonical_usize(internal_mult);
                cols.external_mult = F::from_canonical_usize(external_mult);

                cols.sub_b_value = F::from_canonical_usize(b_value >> 1);
                cols.num_bits = F::from_canonical_usize(num_bits);

                cols.b_lsb = F::from_canonical_usize(b_value & 1);
                cols.u_val
                    .copy_from_slice(u[vk.inner.params.n_stack - num_bits].as_base_slice());
                cols.sub_eval.copy_from_slice(sub_eval.as_base_slice());
            }

            combined_trace.extend(trace);
            total_rows += num_rows;
        }

        let padded_rows = total_rows.next_power_of_two();
        if padded_rows > total_rows {
            let padding_start = combined_trace.len();
            combined_trace.resize(padded_rows * width, F::ZERO);

            let padding_proof_idx = F::from_canonical_usize(proofs.len());
            for chunk in combined_trace[padding_start..].chunks_mut(width) {
                let cols: &mut EqBitsCols<F> = chunk.borrow_mut();
                cols.proof_idx = padding_proof_idx;
            }
        }

        RowMajorMatrix::new(combined_trace, width)
    }
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
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

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
        self.eq_rand_values_bus.receive(
            builder,
            local.proof_idx,
            EqRandValuesLookupMessage {
                idx: AB::Expr::from_canonical_usize(self.n_stack + 1) - local.num_bits,
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
            },
            and(not(local.is_first), local.is_valid),
        );

        let b_value = AB::Expr::TWO * local.sub_b_value + local.b_lsb;
        builder.assert_bool(local.b_lsb);

        /*
         * Compute eq_bits(u, b) and send it to the appropriate internal and external
         * buses.
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

        self.eq_bits_internal_bus.send(
            builder,
            local.proof_idx,
            EqBitsInternalMessage {
                b_value: b_value.clone(),
                num_bits: local.num_bits.into(),
                eval: eval.clone(),
            },
            local.is_valid * local.internal_mult,
        );

        self.eq_bits_lookup_bus.send(
            builder,
            local.proof_idx,
            EqBitsLookupMessage {
                b_value: b_value * AB::Expr::from_canonical_usize(1 << self.l_skip),
                num_bits: local.num_bits.into(),
                eval,
            },
            local.is_valid * local.external_mult,
        );
    }
}
