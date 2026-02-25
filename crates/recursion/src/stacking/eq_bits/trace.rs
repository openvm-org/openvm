use std::{borrow::BorrowMut, collections::HashMap};

#[cfg(all(test, feature = "cuda"))]
use itertools::Itertools;
use openvm_stark_sdk::config::baby_bear_poseidon2::{EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    stacking::{eq_bits::air::EqBitsCols, utils::get_stacked_slice_data},
    tracegen::{RowMajorChip, StandardTracegenCtx},
};

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
