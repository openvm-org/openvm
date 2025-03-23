pub use domain::*;
use openvm_native_compiler::{
    ir::{
        Array, ArrayLike, Builder, Config, Ext, ExtensionOperand, Felt, RVar, SymbolicVar, Usize,
        Var,
    },
    prelude::MemVariable,
};
use openvm_native_compiler_derive::iter_zip;
use openvm_stark_backend::p3_field::{FieldAlgebra, TwoAdicField};
use tracing::debug;
pub use two_adic_pcs::*;

use self::types::{DimensionsVariable, FriConfigVariable, FriQueryProofVariable};
use crate::{
    digest::{CanPoseidon2Digest, DigestVariable},
    outer_poseidon2::Poseidon2CircuitBuilder,
    utils::cond_eval,
    vars::{HintSlice, OuterDigestVariable},
};

pub mod domain;
pub mod hints;
pub mod two_adic_pcs;
pub mod types;
pub mod witness;

/// Verifies a FRI query.
///
/// Currently assumes the index that is accessed is constant.
///
/// Reference: <https://github.com/Plonky3/Plonky3/blob/4809fa7bedd9ba8f6f5d3267b1592618e3776c57/fri/src/verifier.rs#L101>
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)]
pub fn verify_query<C: Config>(
    builder: &mut Builder<C>,
    config: &FriConfigVariable<C>,
    commit_phase_commits: &Array<C, DigestVariable<C>>,
    index_bits: &Array<C, Var<C::N>>,
    proof: &FriQueryProofVariable<C>,
    betas: &Array<C, Ext<C::F, C::EF>>,
    reduced_openings: &Array<C, Ext<C::F, C::EF>>,
    log_max_lde_height: RVar<C::N>,
) -> Ext<C::F, C::EF>
where
    C::F: TwoAdicField,
    C::EF: TwoAdicField,
{
    builder.cycle_tracker_start("verify-query");

    let folded_eval: Ext<C::F, C::EF> = builder.eval(C::F::ZERO);
    let subgroup_size: Var<C::N> =
        builder.eval(log_max_lde_height + RVar::from(config.arity_bits) - C::N::ONE);
    let two_adic_generator_f = config.get_two_adic_generator(builder, subgroup_size);

    let two_adic_gen_ext = two_adic_generator_f.to_operand().symbolic();
    let two_adic_generator_ef: Ext<_, _> = builder.eval(two_adic_gen_ext);

    let base = two_adic_generator_ef;

    let index_bits_truncated = index_bits.slice(builder, 0, log_max_lde_height);

    let log_folded_height: Var<C::N> = builder.eval(log_max_lde_height);

    let get_idx = |builder: &mut Builder<C>, start: Var<C::N>, end: Var<C::N>| {
        let idx: Var<C::N> = builder.eval(C::N::ZERO);
        builder.range(start, end).for_each(|i_vec, builder| {
            let i = i_vec[0];
            let bit = builder.get(&index_bits, i);
            builder.assign(&idx, idx * C::N::TWO + bit);
        });
        idx
    };

    let assert_opened_row = |builder: &mut Builder<C>,
                             opened_rows: &Array<C, Array<C, Ext<C::F, C::EF>>>,
                             row_idx: RVar<C::N>,
                             col_idx_start: Var<C::N>,
                             col_idx_end: SymbolicVar<C::N>,
                             expected_val: Ext<C::F, C::EF>| {
        let col_idx_end = builder.eval(col_idx_end);
        let col_idx = get_idx(builder, col_idx_start, col_idx_end);
        let row = builder.get(opened_rows, row_idx);
        let opened_val = builder.get(&row, col_idx);
        builder.assert_ext_eq(opened_val, expected_val);
    };

    let add_one = |builder: &mut Builder<C>, counter: &Array<C, Var<C::N>>| {
        let zero = builder.eval(C::N::ZERO);
        let one = builder.eval(C::N::ONE);

        let carry_over: Var<C::N> = builder.eval(C::N::ONE);
        builder.range(0, counter.len()).for_each(|k_vec, builder| {
            let k = k_vec[0];

            let old_c_val = builder.get(&counter, k);
            let new_c_val: Var<C::N> = builder.eval(old_c_val + carry_over);

            builder.if_eq(new_c_val, zero).then(|builder| {
                builder.set_value(&counter, k, zero);
                builder.assign(&carry_over, zero);
            });
            builder.if_eq(new_c_val, one).then(|builder| {
                builder.set_value(&counter, k, one);
                builder.assign(&carry_over, zero);
            });
            builder.if_eq(new_c_val, C::N::TWO).then(|builder| {
                builder.set_value(&counter, k, zero);
                builder.assign(&carry_over, one);
            });
        });
    };

    let fold_row = |builder: &mut Builder<C>,
                    x_0: Ext<C::F, C::EF>,
                    x_1: Ext<C::F, C::EF>,
                    e_0: Ext<C::F, C::EF>,
                    e_1: Ext<C::F, C::EF>,
                    beta: Ext<C::F, C::EF>| {
        let ret = builder.eval(e_0 + (beta - x_0) * (e_1 - e_0) / (x_1 - x_0));
        ret
    };

    // This returns 1 if the provided value is between [-log_blowup - log_final_poly_len, 0)
    let check_if_negative = |builder: &mut Builder<C>, val: Var<C::N>| {
        let threshold: Var<C::N> =
            builder.eval(RVar::from(config.log_blowup + config.log_final_poly_len));
        let ret: Var<C::N> = builder.eval(C::N::ZERO);
        builder.range(0, threshold).for_each(|i_vec, builder| {
            builder.assign(&val, val + C::N::ONE);
            builder.if_eq(val, C::N::ZERO).then(|builder| {
                builder.assign(&ret, C::N::ONE);
            });
        });
        ret
    };

    // Returns base ^ (ls_part || ms_part), where both parts are big endian
    let get_power = |builder: &mut Builder<C>,
                     base: Ext<C::F, C::EF>,
                     ls_part: &Array<C, Var<C::N>>,
                     ms_part: &Array<C, Var<C::N>>,
                     ms_len: Var<C::N>| {
        let result: Ext<C::F, C::EF> = builder.eval(C::F::ONE);
        let power_f: Ext<C::F, C::EF> = builder.eval(base);
        let one_var: Ext<C::F, C::EF> = builder.eval(C::F::ONE);

        iter_zip!(builder, ls_part).for_each(|ptr_vec, builder| {
            let bit = builder.iter_ptr_get(ls_part, ptr_vec[0]);
            builder.assign(&result, result * result);
            let mul = builder.select_ef(bit, power_f, one_var);
            builder.assign(&result, result * mul);
        });

        builder.range(0, ms_len).for_each(|i_vec, builder| {
            let i = i_vec[0];
            let bit = builder.get(ms_part, i);
            builder.assign(&result, result * result);
            let mul = builder.select_ef(bit, power_f, one_var);
            builder.assign(&result, result * mul);
        });

        result
    };

    let index_bits_offset: Var<C::N> = builder.eval(C::N::ZERO);

    // proof.commit_phase_openings.len() == log_max_lde_height - log_blowup
    builder.assert_usize_eq(
        proof.commit_phase_openings.len(),
        commit_phase_commits.len(),
    );
    builder
        .range(0, commit_phase_commits.len())
        .for_each(|i_vec, builder| {
            let i = i_vec[0];
            let i_var: Var<C::N> = builder.eval(i);

            let cur_arity_bits: Var<C::N> = builder.eval(RVar::from(config.arity_bits));

            let last_round_idx: Var<C::N> = builder.eval(commit_phase_commits.len() - C::N::ONE);
            builder.if_eq(i, last_round_idx).then(|builder| {
                // Here, we want to minimize cur_arity_bits with log_folded_height
                let next_log_folded_height: Var<C::N> =
                    builder.eval(log_folded_height - cur_arity_bits);
                let is_negative = check_if_negative(builder, next_log_folded_height);
                builder.if_eq(is_negative, C::N::ONE).then(|builder| {
                    builder.assign(&cur_arity_bits, log_folded_height);
                });
            });

            let beta = builder.get(betas, i);
            let commit = builder.get(commit_phase_commits, i);
            let opening = builder.get(&proof.commit_phase_openings, i);

            let reduced_opening = builder.get(reduced_openings, log_folded_height);
            builder.assign(&folded_eval, folded_eval + reduced_opening);

            let index_row = index_bits.shift(builder, cur_arity_bits);

            assert_opened_row(
                builder,
                &opening.opened_rows,
                RVar::from(0),
                index_bits_offset,
                index_bits_offset + RVar::from(cur_arity_bits),
                folded_eval,
            );

            let opened_row_index: Var<C::N> = builder.eval(C::N::ONE);
            builder.range(1, cur_arity_bits).for_each(|j_vec, builder| {
                let j = j_vec[0];
                let lh: Var<C::N> = builder.eval(log_folded_height - j);
                let ro = builder.get(reduced_openings, lh);

                // TODO[osama]: the following should be inside an if condition testing if a new polynomial enters

                let opened_row = builder.get(&opening.opened_rows, opened_row_index);
                builder.assign(&opened_row_index, opened_row_index + C::N::ONE);

                // Make sure the opened row is of the correct length
                let log_row_len: Var<C::N> =
                    builder.eval(lh + RVar::from(cur_arity_bits) - log_folded_height);
                let row_len: Var<C::N> = builder.sll(C::N::ONE, log_row_len.into());
                builder.assert_eq::<Var<C::N>>(opened_row.len(), row_len);

                builder.assign(&index_bits_offset, index_bits_offset + C::N::ONE);

                assert_opened_row(
                    builder,
                    &opening.opened_rows,
                    opened_row_index.into(),
                    index_bits_offset,
                    index_bits_offset + log_row_len,
                    ro,
                );
            });

            let log_height: Var<C::N> =
                builder.eval(log_folded_height - RVar::from(cur_arity_bits));
            let dim = DimensionsVariable::<C> {
                log_height: log_height.into(),
            };
            let dims_slice: Array<C, DimensionsVariable<C>> =
                builder.array(opening.opened_rows.len());
            builder
                .range(0, opening.opened_rows.len())
                .for_each(|i_vec, builder| {
                    let i = i_vec[0];
                    builder.set_value(&dims_slice, i, dim.clone());
                });

            builder.assign(&index_bits_offset, index_bits_offset + C::N::ONE);
            let index_row = index_bits.shift(builder, index_bits_offset);

            // verify_batch::<C>(
            //     builder,
            //     &commit,
            //     dims_slice,
            //     index_row,
            //     &NestedOpenedValues::Ext(opening.opened_rows),
            //     &opening.opening_proof,
            // );

            // Do the folding logic

            let folded_row = builder.get(&opening.opened_rows, 0);
            let opened_row_index: Var<C::N> = builder.eval(C::N::ONE);

            let folded_lens = builder.dyn_array(cur_arity_bits);
            let cur_len: Var<C::N> = builder.eval(C::N::ONE);
            builder.range(0, cur_arity_bits).for_each(|i_vec, builder| {
                let i = i_vec[0];
                let len_idx: Var<C::N> = builder.eval(cur_arity_bits - i - C::N::ONE);

                builder.set_value(&folded_lens, len_idx, cur_len);
                builder.assign(&cur_len, cur_len + cur_len);
            });

            builder.range(0, cur_arity_bits).for_each(|i_vec, builder| {
                let i = i_vec[0];

                builder.assign(&log_folded_height, log_folded_height - C::N::ONE);

                let log_new_folded_row_len: Var<C::N> =
                    builder.eval(cur_arity_bits - i - C::N::ONE);
                let counter = builder.dyn_array(log_new_folded_row_len);

                let new_folded_row_len = builder.get(&folded_lens, i);
                let new_folded_row = builder.dyn_array(new_folded_row_len);

                builder
                    .range(0, new_folded_row_len)
                    .for_each(|j_vec, builder| {
                        let j = j_vec[0];

                        let x = get_power(builder, base, &counter, &index_row, log_folded_height);

                        let k_0: Var<C::N> = builder.eval(j.clone() + j.clone());
                        let k_1: Var<C::N> = builder.eval(k_0 + C::N::ONE);

                        let e_0 = builder.get(&folded_row, k_0);
                        let e_1 = builder.get(&folded_row, k_1);

                        let two_adic_generator_one =
                            config.get_two_adic_generator(builder, Usize::from(1));

                        let x_0 = x;
                        let x_1 = builder.eval(x * two_adic_generator_one);

                        let folded_val = fold_row(builder, x_0, x_1, e_0, e_1, beta);
                        builder.set_value(&new_folded_row, j, folded_val);

                        // increment counter
                        add_one(builder, &counter);
                    });

                builder.assign(&base, base * base);

                builder.assign(&folded_row, new_folded_row);
                builder.assign(&beta, beta * beta);

                builder
                    .if_ne(opened_row_index, opening.opened_rows.len())
                    .then(|builder| {
                        let opened_row = builder.get(&opening.opened_rows, opened_row_index);
                        builder
                            .if_eq(opened_row.len(), folded_row.len())
                            .then(|builder| {
                                builder
                                    .range(0, opened_row.len())
                                    .for_each(|j_vec, builder| {
                                        let j = j_vec[0];
                                        let folded_val = builder.get(&folded_row, j);
                                        let opened_val = builder.get(&opened_row, j);
                                        let sum = builder.eval(folded_val + opened_val);
                                        builder.set_value(&folded_row, j, sum);
                                    });
                                builder.assign(&opened_row_index, opened_row_index + C::N::ONE);
                            })
                    });
            });

            let new_folded_eval = builder.get(&folded_row, 0);
            builder.assign(&folded_eval, new_folded_eval);

            // TODO[osama]: maybe figure out how to move this
            verify_batch::<C>(
                builder,
                &commit,
                dims_slice,
                index_row,
                &NestedOpenedValues::Ext(opening.opened_rows),
                &opening.opening_proof,
            );
        });

    builder.cycle_tracker_end("verify-query");
    folded_eval
}

#[allow(clippy::type_complexity)]
pub enum NestedOpenedValues<C: Config> {
    Felt(Array<C, Array<C, Felt<C::F>>>),
    Ext(Array<C, Array<C, Ext<C::F, C::EF>>>),
}

/// Verifies a batch opening.
///
/// Assumes the dimensions have already been sorted by tallest first.
///
/// Reference: <https://github.com/Plonky3/Plonky3/blob/4809fa7bedd9ba8f6f5d3267b1592618e3776c57/merkle-tree/src/mmcs.rs#L92>
#[allow(clippy::type_complexity)]
#[allow(unused_variables)]
pub fn verify_batch<C: Config>(
    builder: &mut Builder<C>,
    commit: &DigestVariable<C>,
    dimensions: Array<C, DimensionsVariable<C>>,
    index_bits: Array<C, Var<C::N>>,
    opened_values: &NestedOpenedValues<C>,
    proof: &HintSlice<C>,
) {
    if builder.flags.static_only {
        verify_batch_static(
            builder,
            commit,
            dimensions,
            index_bits,
            opened_values,
            proof,
        );
        return;
    }

    let dimensions = match dimensions {
        Array::Dyn(ptr, len) => Array::Dyn(ptr, len.clone()),
        _ => panic!("Expected a dynamic array of felts"),
    };
    let commit = match commit {
        DigestVariable::Felt(arr) => arr,
        _ => panic!("Expected a dynamic array of felts"),
    };
    match opened_values {
        NestedOpenedValues::Felt(opened_values) => builder.verify_batch_felt(
            &dimensions,
            opened_values,
            proof.id.get_var(),
            &index_bits,
            commit,
        ),
        NestedOpenedValues::Ext(opened_values) => builder.verify_batch_ext(
            &dimensions,
            opened_values,
            proof.id.get_var(),
            &index_bits,
            commit,
        ),
    };
}

/// [static version] Verifies a batch opening.
///
/// Assumes the dimensions have already been sorted by tallest first.
///
/// Reference: <https://github.com/Plonky3/Plonky3/blob/4809fa7bedd9ba8f6f5d3267b1592618e3776c57/merkle-tree/src/mmcs.rs#L92>
#[allow(clippy::type_complexity)]
#[allow(unused_variables)]
pub fn verify_batch_static<C: Config>(
    builder: &mut Builder<C>,
    commit: &DigestVariable<C>,
    dimensions: Array<C, DimensionsVariable<C>>,
    index_bits: Array<C, Var<C::N>>,
    opened_values: &NestedOpenedValues<C>,
    proof: &HintSlice<C>,
) {
    let commit: OuterDigestVariable<C> = if let DigestVariable::Var(commit) = commit {
        commit.vec().try_into().unwrap()
    } else {
        panic!("Expected a Var commitment");
    };
    // The index of which table to process next.
    let index: Usize<C::N> = builder.eval(C::N::ZERO);
    // The height of the current layer (padded).
    let mut current_log_height = builder.get(&dimensions, index.clone()).log_height.value();
    // Reduce all the tables that have the same height to a single root.
    let reducer = opened_values.create_reducer(builder);
    let mut root = reducer
        .reduce_fast(
            builder,
            index.clone(),
            &dimensions,
            current_log_height,
            opened_values,
        )
        .into_outer_digest();

    // For each sibling in the proof, reconstruct the root.
    let witness_refs = builder.get_witness_refs(proof.id.clone()).to_vec();
    for (i, &witness_ref) in witness_refs.iter().enumerate() {
        let sibling: OuterDigestVariable<C> = [witness_ref.into()];
        let bit = builder.get(&index_bits, i);

        let [left, right]: [Var<_>; 2] = cond_eval(builder, bit, root[0], sibling[0]);
        root = builder.p2_compress([[left], [right]]);
        current_log_height -= 1;

        builder
            .if_ne(index.clone(), dimensions.len())
            .then(|builder| {
                let next_log_height = builder.get(&dimensions, index.clone()).log_height;
                builder
                    .if_eq(next_log_height, Usize::from(current_log_height))
                    .then(|builder| {
                        let next_height_openings_digest = reducer
                            .reduce_fast(
                                builder,
                                index.clone(),
                                &dimensions,
                                current_log_height,
                                opened_values,
                            )
                            .into_outer_digest();
                        root = builder.p2_compress([root, next_height_openings_digest]);
                    });
            })
    }

    builder.assert_var_eq(root[0], commit[0]);
}

#[allow(clippy::type_complexity)]
fn reduce_fast<C: Config, V: MemVariable<C>>(
    builder: &mut Builder<C>,
    dim_idx: Usize<C::N>,
    dims: &Array<C, DimensionsVariable<C>>,
    cur_log_height: usize,
    opened_values: &Array<C, Array<C, V>>,
    nested_opened_values_buffer: &Array<C, Array<C, V>>,
) -> DigestVariable<C>
where
    Array<C, Array<C, V>>: CanPoseidon2Digest<C>,
{
    builder.cycle_tracker_start("verify-batch-reduce-fast");

    // `nested_opened_values_buffer` will be truncated in this function. We want to avoid modifying
    // the original buffer object, so we create a new one or clone it.
    let nested_opened_values_buffer = if builder.flags.static_only {
        builder.array(REDUCER_BUFFER_SIZE)
    } else {
        // This points to the same memory. Only the length of this object will change when truncating.
        let ret = builder.uninit();
        builder.assign(&ret, nested_opened_values_buffer.clone());
        ret
    };

    let nb_opened_values: Usize<_> = builder.eval(C::N::ZERO);
    let start_dim_idx: Usize<_> = builder.eval(dim_idx.clone());
    builder.cycle_tracker_start("verify-batch-reduce-fast-setup");
    let dims_shifted = dims.shift(builder, start_dim_idx.clone());
    let opened_values_shifted = opened_values.shift(builder, start_dim_idx);
    iter_zip!(builder, dims_shifted, opened_values_shifted).for_each(|ptr_vec, builder| {
        let log_height = builder.iter_ptr_get(&dims_shifted, ptr_vec[0]).log_height;
        builder
            .if_eq(log_height, Usize::from(cur_log_height))
            .then(|builder| {
                let opened_values = builder.iter_ptr_get(&opened_values_shifted, ptr_vec[1]);
                builder.set_value(
                    &nested_opened_values_buffer,
                    nb_opened_values.clone(),
                    opened_values.clone(),
                );
                builder.assign(&nb_opened_values, nb_opened_values.clone() + C::N::ONE);
            });
    });
    builder.assign(&dim_idx, dim_idx.clone() + nb_opened_values.clone());
    builder.cycle_tracker_end("verify-batch-reduce-fast-setup");

    nested_opened_values_buffer.truncate(builder, nb_opened_values);
    let h = nested_opened_values_buffer.p2_digest(builder);
    builder.cycle_tracker_end("verify-batch-reduce-fast");
    h
}

struct NestedOpenedValuesReducerVar<C: Config> {
    buffer: NestedOpenedValues<C>,
}
impl<C: Config> NestedOpenedValuesReducerVar<C> {
    fn reduce_fast(
        &self,
        builder: &mut Builder<C>,
        dim_idx: Usize<C::N>,
        dims: &Array<C, DimensionsVariable<C>>,
        cur_log_height: usize,
        nested_opened_values: &NestedOpenedValues<C>,
    ) -> DigestVariable<C> {
        match nested_opened_values {
            NestedOpenedValues::Felt(opened_values) => {
                let buffer = match &self.buffer {
                    NestedOpenedValues::Felt(buffer) => buffer,
                    NestedOpenedValues::Ext(_) => unreachable!(),
                };
                reduce_fast(
                    builder,
                    dim_idx,
                    dims,
                    cur_log_height,
                    opened_values,
                    buffer,
                )
            }
            NestedOpenedValues::Ext(opened_values) => {
                let buffer = match &self.buffer {
                    NestedOpenedValues::Felt(_) => unreachable!(),
                    NestedOpenedValues::Ext(buffer) => buffer,
                };
                reduce_fast(
                    builder,
                    dim_idx,
                    dims,
                    cur_log_height,
                    opened_values,
                    buffer,
                )
            }
        }
    }
}

/// 8192 is just a random large enough number.
const REDUCER_BUFFER_SIZE: usize = 8192;

impl<C: Config> NestedOpenedValues<C> {
    fn create_reducer(&self, builder: &mut Builder<C>) -> NestedOpenedValuesReducerVar<C> {
        NestedOpenedValuesReducerVar {
            buffer: match self {
                NestedOpenedValues::Felt(_) => {
                    NestedOpenedValues::Felt(builder.array(REDUCER_BUFFER_SIZE))
                }
                NestedOpenedValues::Ext(_) => {
                    NestedOpenedValues::Ext(builder.array(REDUCER_BUFFER_SIZE))
                }
            },
        }
    }
}
