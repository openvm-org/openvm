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
fn verify_query<C: Config>(
    builder: &mut Builder<C>,
    config: &FriConfigVariable<C>,
    commit_phase_commits: &Array<C, DigestVariable<C>>,
    index_bits: &Array<C, Var<C::N>>,
    proof: &FriQueryProofVariable<C>,
    betas: &Array<C, Ext<C::F, C::EF>>,
    betas_squared: &Array<C, Ext<C::F, C::EF>>,
    reduced_openings: &Array<C, Ext<C::F, C::EF>>,
    log_max_lde_height: RVar<C::N>,
    i_plus_one_arr: &Array<C, Usize<C::N>>,
) -> Ext<C::F, C::EF>
where
    C::F: TwoAdicField,
    C::EF: TwoAdicField,
{
    builder.cycle_tracker_start("verify-query");
    // reduced_openings.len() == MAX_TWO_ADICITY >= log_max_lde_height
    let folded_eval: Ext<C::F, C::EF> = builder.get(reduced_openings, log_max_lde_height);
    let two_adic_generator_f = config.get_two_adic_generator(builder, log_max_lde_height);

    let two_adic_gen_ext = two_adic_generator_f.to_operand().symbolic();
    let two_adic_generator_ef: Ext<_, _> = builder.eval(two_adic_gen_ext);

    let index_bits_truncated = index_bits.slice(builder, 0, log_max_lde_height);
    let x = builder.exp_bits_big_endian(two_adic_generator_ef, &index_bits_truncated);

    // assert proof.commit_phase_openings.len() == log_max_height
    // where
    //   - commit_phase_commits.len() = log_max_height is compile-time constant (assuming
    //     log_final_poly_len = 0)
    //   - log_max_lde_height = log_max_height + log_blowup by definition in verify_two_adic_pcs
    builder.assert_usize_eq(
        proof.commit_phase_openings.len(),
        commit_phase_commits.len(),
    );
    // By definition in verify_two_adic_pcs:
    // - betas, betas_squared, i_plus_one_arr have length log_max_height
    // - index_bits has length log_max_lde_height
    iter_zip!(
        builder,
        commit_phase_commits,
        proof.commit_phase_openings,
        betas,
        betas_squared,
        i_plus_one_arr,
        index_bits
    )
    .for_each(|ptr_vec, builder| {
        let [comm_ptr, opening_ptr, beta_ptr, beta_sq_ptr, i_plus_one_ptr, index_bit_ptr] =
            ptr_vec.try_into().unwrap();

        let i_plus_one = builder.iter_ptr_get(i_plus_one_arr, i_plus_one_ptr);
        let i_plus_one = RVar::from(i_plus_one);
        let log_folded_height = builder.eval_expr(log_max_lde_height - i_plus_one);
        let commit = builder.iter_ptr_get(commit_phase_commits, comm_ptr);
        let step = builder.iter_ptr_get(&proof.commit_phase_openings, opening_ptr);
        let beta = builder.iter_ptr_get(betas, beta_ptr);
        let beta_sq = builder.iter_ptr_get(betas_squared, beta_sq_ptr);

        let index_bit = builder.iter_ptr_get(index_bits, index_bit_ptr);
        let index_sibling_mod_2: Var<C::N> = builder.eval(SymbolicVar::from(C::N::ONE) - index_bit);
        let index_pair = index_bits.shift(builder, i_plus_one);

        let evals: Array<C, Ext<C::F, C::EF>> = builder.array(2);
        let eval_0: Ext<C::F, C::EF>;
        let eval_1: Ext<C::F, C::EF>;
        if builder.flags.static_only {
            [eval_0, eval_1] = cond_eval(
                builder,
                index_sibling_mod_2,
                step.sibling_value,
                folded_eval,
            );
            builder.set_value(&evals, 0, eval_0);
            builder.set_value(&evals, 1, eval_1);
        } else {
            builder.set_value(&evals, 0, folded_eval);
            builder.set_value(&evals, 1, folded_eval);
            // This is faster than branching.
            builder.set_value(&evals, index_sibling_mod_2, step.sibling_value);
            eval_0 = builder.get(&evals, 0);
            eval_1 = builder.get(&evals, 1);
        }

        let dims = DimensionsVariable::<C> {
            log_height: builder.eval(log_folded_height),
        };
        let dims_slice: Array<C, DimensionsVariable<C>> = builder.array(1);
        builder.set_value(&dims_slice, 0, dims);

        let opened_values = builder.array(1);
        builder.set_value(&opened_values, 0, evals);
        builder.cycle_tracker_start("verify-batch-ext");
        verify_batch::<C>(
            builder,
            &commit,
            dims_slice,
            index_pair,
            &NestedOpenedValues::Ext(opened_values),
            &step.opening_proof,
        );
        builder.cycle_tracker_end("verify-batch-ext");

        let two_adic_generator_one = config.get_two_adic_generator(builder, Usize::from(1));

        let [xs_0, xs_1]: [Ext<_, _>; 2] =
            cond_eval(builder, index_sibling_mod_2, x * two_adic_generator_one, x);

        builder.assign(
            &folded_eval,
            eval_0 + (beta - xs_0) * (eval_1 - eval_0) / (xs_1 - xs_0),
        );

        builder.assign(&x, x * x);

        // reduced_openings.len() == MAX_TWO_ADICITY >= log_max_lde_height >= log_folded_height
        let reduced_opening = builder.get(reduced_openings, log_folded_height);
        // Roll in new reduced opening polynomial at the folded height. This is 0 if there is no
        // reduced opening at the folded height.
        //
        // Each `reduced_opening` is the evaluation of a reduced opening polynomial, which is itself
        // a random linear combination `f_{i, 0}(x) + alpha f_{i, 1}(x) + ...`, but when we add it
        // to the current folded polynomial evaluation claim, we need to multiply by a new random
        // factor since `f_{i, 0}` has no leading coefficient.
        //
        // We use `beta^2` as the random factor since `beta` is already used in the folding.
        //
        // Note: this will include the case `reduced_opening = reduced_openings[log_blowup]` in the
        // last iteration of the loop. Since we roll this in with the beta^2 factor, the low degree
        // test will also include this case (corresponding to `log_height = 0`, which is not done in
        // the Plonky3 implementation).
        builder.assign(&folded_eval, folded_eval + beta_sq * reduced_opening);
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
        // This points to the same memory. Only the length of this object will change when
        // truncating.
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
