use axvm_native_compiler::{
    ir::{
        Array, Builder, Config, Ext, ExtensionOperand, Felt, Ptr, RVar, SymbolicVar, Usize, Var,
        DIGEST_SIZE,
    },
    prelude::MemVariable,
};
pub use domain::*;
use p3_field::{AbstractField, Field, TwoAdicField};
pub use two_adic_pcs::*;

use self::types::{
    DimensionsVariable, FriChallengesVariable, FriConfigVariable, FriProofVariable,
    FriQueryProofVariable,
};
use crate::{
    challenger::ChallengerVariable,
    digest::{CanPoseidon2Digest, DigestVariable},
    outer_poseidon2::Poseidon2CircuitBuilder,
    utils::cond_eval,
    vars::OuterDigestVariable,
};

pub mod domain;
pub mod hints;
pub mod two_adic_pcs;
pub mod types;
pub mod witness;

/// Reference: https://github.com/Plonky3/Plonky3/blob/4809fa7bedd9ba8f6f5d3267b1592618e3776c57/fri/src/verifier.rs#L27
pub fn verify_shape_and_sample_challenges<C: Config>(
    builder: &mut Builder<C>,
    config: &FriConfigVariable<C>,
    proof: &FriProofVariable<C>,
    challenger: &mut impl ChallengerVariable<C>,
) -> FriChallengesVariable<C> {
    let betas: Array<C, Ext<C::F, C::EF>> = builder.array(proof.commit_phase_commits.len());

    builder
        .range(0, proof.commit_phase_commits.len())
        .for_each(|i, builder| {
            let comm = builder.get(&proof.commit_phase_commits, i);
            challenger.observe_digest(builder, comm);
            let sample = challenger.sample_ext(builder);
            builder.set(&betas, i, sample);
        });

    let final_poly_felts = builder.ext2felt(proof.final_poly);
    challenger.observe_slice(builder, final_poly_felts);

    let num_query_proofs = proof.query_proofs.len().clone();
    builder
        .if_ne(num_query_proofs, RVar::from(config.num_queries))
        .then(|builder| {
            builder.error();
        });

    challenger.check_witness(builder, config.proof_of_work_bits, proof.pow_witness);

    let log_max_height =
        builder.eval_expr(proof.commit_phase_commits.len() + RVar::from(config.log_blowup));
    let query_indices = builder.array(config.num_queries);
    builder.range(0, config.num_queries).for_each(|i, builder| {
        let index_bits = challenger.sample_bits(builder, log_max_height);
        builder.set(&query_indices, i, index_bits);
    });

    FriChallengesVariable {
        query_indices,
        betas,
    }
}

/// Verifies a set of FRI challenges.
///
/// Reference: https://github.com/Plonky3/Plonky3/blob/4809fa7bedd9ba8f6f5d3267b1592618e3776c57/fri/src/verifier.rs#L67
#[allow(clippy::type_complexity)]
pub fn verify_challenges<C: Config>(
    builder: &mut Builder<C>,
    config: &FriConfigVariable<C>,
    proof: &FriProofVariable<C>,
    challenges: &FriChallengesVariable<C>,
    reduced_openings: &Array<C, Array<C, Ext<C::F, C::EF>>>,
) where
    C::F: TwoAdicField,
    C::EF: TwoAdicField,
{
    let log_max_height =
        builder.eval_expr(proof.commit_phase_commits.len() + RVar::from(config.log_blowup));
    builder
        .range(0, challenges.query_indices.len())
        .for_each(|i, builder| {
            let index_bits = builder.get(&challenges.query_indices, i);
            let query_proof = builder.get(&proof.query_proofs, i);
            let ro = builder.get(reduced_openings, i);

            let folded_eval = verify_query(
                builder,
                config,
                &proof.commit_phase_commits,
                &index_bits,
                &query_proof,
                &challenges.betas,
                &ro,
                log_max_height,
            );

            builder.assert_ext_eq(folded_eval, proof.final_poly);
        });
}

/// Verifies a FRI query.
///
/// Currently assumes the index that is accessed is constant.
///
/// Reference: https://github.com/Plonky3/Plonky3/blob/4809fa7bedd9ba8f6f5d3267b1592618e3776c57/fri/src/verifier.rs#L101
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
    log_max_height: RVar<C::N>,
) -> Ext<C::F, C::EF>
where
    C::F: TwoAdicField,
    C::EF: TwoAdicField,
{
    builder.cycle_tracker_start("verify-query");
    let folded_eval: Ext<C::F, C::EF> = builder.eval(C::F::ZERO);
    let two_adic_generator_f = config.get_two_adic_generator(builder, log_max_height);

    let two_adic_gen_ext = two_adic_generator_f.to_operand().symbolic();
    let two_adic_generator_ef: Ext<_, _> = builder.eval(two_adic_gen_ext);

    let x = builder.exp_reverse_bits_len(two_adic_generator_ef, index_bits, log_max_height);

    builder
        .range(0, commit_phase_commits.len())
        .for_each(|i, builder| {
            let log_folded_height = builder.eval_expr(log_max_height - i - C::N::ONE);
            let log_folded_height_plus_one = builder.eval_expr(log_folded_height + C::N::ONE);
            let commit = builder.get(commit_phase_commits, i);
            let step = builder.get(&proof.commit_phase_openings, i);
            let beta = builder.get(betas, i);

            let reduced_opening = builder.get(reduced_openings, log_folded_height_plus_one);
            builder.assign(&folded_eval, folded_eval + reduced_opening);

            let index_bit = builder.get(index_bits, i);
            let index_sibling_mod_2: Var<C::N> =
                builder.eval(SymbolicVar::from(C::N::ONE) - index_bit);
            let i_plus_one = builder.eval_expr(i + RVar::one());
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
                height: builder.sll(C::N::ONE, log_folded_height),
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
/// Reference: https://github.com/Plonky3/Plonky3/blob/4809fa7bedd9ba8f6f5d3267b1592618e3776c57/merkle-tree/src/mmcs.rs#L92
#[allow(clippy::type_complexity)]
#[allow(unused_variables)]
pub fn verify_batch<C: Config>(
    builder: &mut Builder<C>,
    commit: &DigestVariable<C>,
    dimensions: Array<C, DimensionsVariable<C>>,
    index_bits: Array<C, Var<C::N>>,
    opened_values: &NestedOpenedValues<C>,
    proof: &Array<C, DigestVariable<C>>,
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
    let reducer = opened_values.create_reducer(builder);

    let commit = if let DigestVariable::Felt(commit) = commit {
        commit
    } else {
        panic!("Expected a Felt commitment");
    };
    // Cast DigestVariable into the concrete type.
    let proof: Array<C, Array<C, Felt<C::F>>> = if let Array::Dyn(ptr, len) = proof {
        Array::Dyn(*ptr, len.clone())
    } else {
        panic!("Expected a dynamic array of Felt commitments");
    };

    // The index of which table to process next.
    let index: Usize<C::N> = builder.eval(C::N::ZERO);
    // The height of the current layer (padded).
    let current_height = builder.get(&dimensions, index.clone()).height;
    // Reduce all the tables that have the same height to a single root.
    let root = reducer
        .reduce_fast(
            builder,
            index.clone(),
            &dimensions,
            current_height.clone(),
            opened_values,
        )
        .into_inner_digest();
    let root_ptr = root.ptr();

    // For each sibling in the proof, reconstruct the root.
    let left: Ptr<C::N> = builder.uninit();
    let right: Ptr<C::N> = builder.uninit();
    builder.range(0, proof.len()).for_each(|i, builder| {
        let sibling = builder.get_ptr(&proof, i);
        let bit = builder.get(&index_bits, i);

        builder.if_eq(bit, C::N::ONE).then_or_else(
            |builder| {
                builder.assign(&left, sibling);
                builder.assign(&right, root_ptr);
            },
            |builder| {
                builder.assign(&left, root_ptr);
                builder.assign(&right, sibling);
            },
        );

        builder.poseidon2_compress_x(
            &Array::Dyn(root_ptr, Usize::from(0)),
            &Array::Dyn(left, Usize::from(0)),
            &Array::Dyn(right, Usize::from(0)),
        );
        builder.assign(
            &current_height,
            current_height.clone() * (C::N::TWO.inverse()),
        );

        builder
            .if_ne(index.clone(), dimensions.len())
            .then(|builder| {
                let next_height = builder.get(&dimensions, index.clone()).height;
                builder
                    .if_eq(next_height, current_height.clone())
                    .then(|builder| {
                        let next_height_openings_digest = reducer
                            .reduce_fast(
                                builder,
                                index.clone(),
                                &dimensions,
                                current_height.clone(),
                                opened_values,
                            )
                            .into_inner_digest();
                        builder.poseidon2_compress_x(
                            &root.clone(),
                            &root.clone(),
                            &next_height_openings_digest,
                        );
                    });
            })
    });

    // Assert that the commitments match.
    for i in 0..DIGEST_SIZE {
        let e1 = builder.get(commit, i);
        let e2 = builder.get(&root, i);
        builder.assert_felt_eq(e1, e2);
    }
}

/// [static version] Verifies a batch opening.
///
/// Assumes the dimensions have already been sorted by tallest first.
///
/// Reference: https://github.com/Plonky3/Plonky3/blob/4809fa7bedd9ba8f6f5d3267b1592618e3776c57/merkle-tree/src/mmcs.rs#L92
#[allow(clippy::type_complexity)]
#[allow(unused_variables)]
pub fn verify_batch_static<C: Config>(
    builder: &mut Builder<C>,
    commit: &DigestVariable<C>,
    dimensions: Array<C, DimensionsVariable<C>>,
    index_bits: Array<C, Var<C::N>>,
    opened_values: &NestedOpenedValues<C>,
    proof: &Array<C, DigestVariable<C>>,
) {
    let commit: OuterDigestVariable<C> = if let DigestVariable::Var(commit) = commit {
        commit.vec().try_into().unwrap()
    } else {
        panic!("Expected a Var commitment");
    };
    // The index of which table to process next.
    let index: Usize<C::N> = builder.eval(C::N::ZERO);
    // The height of the current layer (padded).
    let current_height = builder.get(&dimensions, index.clone()).height;
    // Reduce all the tables that have the same height to a single root.
    let reducer = opened_values.create_reducer(builder);
    let mut root = reducer
        .reduce_fast(
            builder,
            index.clone(),
            &dimensions,
            current_height.clone(),
            opened_values,
        )
        .into_outer_digest();

    // For each sibling in the proof, reconstruct the root.
    builder.range(0, proof.len()).for_each(|i, builder| {
        let sibling: OuterDigestVariable<C> = if let DigestVariable::Var(d) = builder.get(proof, i)
        {
            d.vec().try_into().unwrap()
        } else {
            panic!("Expected a Var commitment");
        };
        let bit = builder.get(&index_bits, i);

        let [left, right]: [Var<_>; 2] = cond_eval(builder, bit, root[0], sibling[0]);
        root = builder.p2_compress([[left], [right]]);
        builder.assign(
            &current_height,
            current_height.clone() * (C::N::TWO.inverse()),
        );

        builder
            .if_ne(index.clone(), dimensions.len())
            .then(|builder| {
                let next_height = builder.get(&dimensions, index.clone()).height;
                builder
                    .if_eq(next_height, current_height.clone())
                    .then(|builder| {
                        let next_height_openings_digest = reducer
                            .reduce_fast(
                                builder,
                                index.clone(),
                                &dimensions,
                                current_height.clone(),
                                opened_values,
                            )
                            .into_outer_digest();
                        root = builder.p2_compress([root, next_height_openings_digest]);
                    });
            })
    });

    builder.assert_var_eq(root[0], commit[0]);
}

#[allow(clippy::type_complexity)]
fn reduce_fast<C: Config, V: MemVariable<C>>(
    builder: &mut Builder<C>,
    dim_idx: Usize<C::N>,
    dims: &Array<C, DimensionsVariable<C>>,
    curr_height_padded: Usize<C::N>,
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
    builder
        .range(start_dim_idx, dims.len())
        .for_each(|i, builder| {
            let height = builder.get(dims, i).height;
            builder
                .if_eq(height, curr_height_padded.clone())
                .then(|builder| {
                    let opened_values = builder.get(opened_values, i);
                    builder.set_value(
                        &nested_opened_values_buffer,
                        nb_opened_values.clone(),
                        opened_values.clone(),
                    );
                    builder.assign(&nb_opened_values, nb_opened_values.clone() + C::N::ONE);
                    builder.assign(&dim_idx, dim_idx.clone() + C::N::ONE);
                });
        });
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
        curr_height: Usize<C::N>,
        nested_opened_values: &NestedOpenedValues<C>,
    ) -> DigestVariable<C> {
        match nested_opened_values {
            NestedOpenedValues::Felt(opened_values) => {
                let buffer = match &self.buffer {
                    NestedOpenedValues::Felt(buffer) => buffer,
                    NestedOpenedValues::Ext(_) => unreachable!(),
                };
                reduce_fast(builder, dim_idx, dims, curr_height, opened_values, buffer)
            }
            NestedOpenedValues::Ext(opened_values) => {
                let buffer = match &self.buffer {
                    NestedOpenedValues::Felt(_) => unreachable!(),
                    NestedOpenedValues::Ext(buffer) => buffer,
                };
                reduce_fast(builder, dim_idx, dims, curr_height, opened_values, buffer)
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
