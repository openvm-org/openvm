use core::borrow::Borrow;
use std::{array::from_fn, sync::Arc};

use itertools::{fold, Itertools};
use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, SubAir};
use openvm_poseidon2_air::{
    Poseidon2Config, Poseidon2SubAir, Poseidon2SubChip, Poseidon2SubCols,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE, POSEIDON2_WIDTH,
};
use openvm_stark_backend::{air_builders::sub::SubAirBuilder, interaction::InteractionBuilder};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, InjectiveMonomial, PrimeCharacteristicRing, PrimeField};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::expr_eval::{
        CachedRecord, CachedSymbolicExpressionColumns, FLAG_MODULUS, NUM_FLAGS,
    },
    utils::assert_zeros,
};

pub(in crate::batch_constraint) const SBOX_REGISTERS: usize = 1;

pub fn default_poseidon2_sub_chip<
    F: PrimeField + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>,
>() -> Poseidon2SubChip<F, SBOX_REGISTERS> {
    Poseidon2SubChip::<F, SBOX_REGISTERS>::new(Poseidon2Config::default().constants)
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DagCommitCols<T> {
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub flags: [T; NUM_FLAGS],
    pub is_constraint: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DagCommitPvs<T> {
    pub commit: [T; DIGEST_SIZE],
}

/// Sub-AIR to compute the onion hash of one digest per row. Expects each AIR
/// that uses it to have DagCommitPvs as its public value representation.
pub struct DagCommitSubAir<F: Field> {
    pub subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
}

impl<F: PrimeField + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>> DagCommitSubAir<F> {
    pub fn new() -> Self {
        Self {
            subair: default_poseidon2_sub_chip().air,
        }
    }
}

impl<F: Field> BaseAir<F> for DagCommitSubAir<F> {
    fn width(&self) -> usize {
        DagCommitCols::<u8>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> SubAir<AB>
    for DagCommitSubAir<AB::F>
{
    type AirContext<'a>
        = (&'a [AB::Var], &'a [AB::Var])
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, (local, next): (&'a [AB::Var], &'a [AB::Var]))
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let local: &DagCommitCols<AB::Var> = (*local).borrow();
        let next: &DagCommitCols<AB::Var> = (*next).borrow();

        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);

        debug_assert_eq!(FLAG_MODULUS, 3);
        builder.assert_bool(local.is_constraint);
        for flag in local.flags {
            builder.assert_tern(flag);
        }

        let first_digest_element =
            collapse_flags(local.flags.map(Into::into), local.is_constraint.into());
        builder.assert_eq(local.inner.inputs[0], first_digest_element);

        assert_zeros::<_, DIGEST_SIZE>(
            &mut builder.when_first_row(),
            from_fn(|i| local.inner.inputs[i + DIGEST_SIZE]),
        );
        assert_array_eq::<_, _, _, DIGEST_SIZE>(
            &mut builder.when_transition(),
            from_fn(|i| local.inner.ending_full_rounds.last().unwrap().post[i + DIGEST_SIZE]),
            from_fn(|i| next.inner.inputs[i + DIGEST_SIZE]),
        );

        let &DagCommitPvs::<_> { commit: pvs_commit } = builder.public_values().borrow();

        assert_array_eq::<_, _, _, DIGEST_SIZE>(
            &mut builder.when_last_row(),
            from_fn(|i| local.inner.ending_full_rounds.last().unwrap().post[i]),
            pvs_commit,
        );
    }
}

#[derive(Debug, Clone)]
pub struct DagCommitInfo<F: Clone> {
    pub commit: [F; DIGEST_SIZE],
    pub poseidon2_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

/// Collapse flags and is_constraint into a single field element, which should be the
/// first element of a row's input digest.
///
/// WARNING: To use this in an AIR you MUST constrain that is_constraint is boolean
/// and that each flag is in [0, FLAG_MODULUS). This ensures that the element doesn't
/// overflow for any 13-bit field or higher.
pub fn collapse_flags<F: PrimeCharacteristicRing>(flags: [F; NUM_FLAGS], is_constraint: F) -> F {
    fold(
        flags.iter().enumerate(),
        is_constraint.clone(),
        |acc, (pow_exp, flag)| {
            acc + (flag.clone() * F::from_u32(FLAG_MODULUS.pow(pow_exp as u32) << 1))
        },
    )
}

/// Compresses a CachedSymbolicExpressionColumns row into a digest. Uses collapse_flags for
/// the first element.
pub fn cached_symbolic_expr_cols_to_digest<F: PrimeCharacteristicRing>(
    cached_cols: &[F],
) -> [F; DIGEST_SIZE] {
    let cached_cols: &CachedSymbolicExpressionColumns<_> = cached_cols.borrow();
    let mut ret = [F::ZERO; DIGEST_SIZE];
    ret[0] = collapse_flags(cached_cols.flags.clone(), cached_cols.is_constraint.clone());
    ret[1] = cached_cols.air_idx.clone();
    ret[2] = cached_cols.node_or_interaction_idx.clone();
    ret[3] = cached_cols.attrs[0].clone();
    ret[4] = cached_cols.attrs[1].clone();
    ret[5] = cached_cols.attrs[2].clone();
    ret[6] = cached_cols.fanout.clone();
    ret[7] = cached_cols.constraint_idx.clone();
    ret
}

/// Converts an input digest (+ flags and is_constraint) into cached columns.
pub fn digest_to_cached_symbolic_expr_cols<F: Copy>(
    digest: [F; DIGEST_SIZE],
    flags: [F; NUM_FLAGS],
    is_constraint: F,
) -> CachedSymbolicExpressionColumns<F> {
    CachedSymbolicExpressionColumns {
        flags,
        air_idx: digest[1],
        node_or_interaction_idx: digest[2],
        attrs: from_fn(|i| digest[3..][i]),
        fanout: digest[6],
        is_constraint,
        constraint_idx: digest[7],
    }
}

pub fn dag_commit_cols_to_cached_cols<F: Copy>(
    dag_commit_cols: &[F],
) -> CachedSymbolicExpressionColumns<F> {
    let cols: &DagCommitCols<_> = dag_commit_cols.borrow();
    let digest = from_fn(|i| cols.inner.inputs[i]);
    digest_to_cached_symbolic_expr_cols(digest, cols.flags, cols.is_constraint)
}

pub(crate) fn generate_dag_commit_info(
    cached_records: &[CachedRecord],
    encoder: Encoder,
) -> DagCommitInfo<F> {
    let sub_chip = Poseidon2SubChip::<F, SBOX_REGISTERS>::new(Poseidon2Config::default().constants);
    let mut state = [F::ZERO; POSEIDON2_WIDTH];

    let height = cached_records.len().next_power_of_two();
    let poseidon2_inputs = (0..height)
        .map(|row_idx| {
            let digest: [F; DIGEST_SIZE] = if row_idx < cached_records.len() {
                let r = &cached_records[row_idx];
                let encoder_flag = encoder.get_flag_pt(r.kind as usize);
                let columns = CachedSymbolicExpressionColumns {
                    flags: from_fn(|i| F::from_u32(encoder_flag[i])),
                    air_idx: F::from_usize(r.air_idx),
                    node_or_interaction_idx: F::from_usize(r.node_idx),
                    attrs: r.attrs.map(F::from_usize),
                    fanout: F::from_usize(r.fanout),
                    is_constraint: F::from_bool(r.is_constraint),
                    constraint_idx: F::from_usize(r.constraint_idx),
                };
                cached_symbolic_expr_cols_to_digest(&columns.to_vec())
            } else {
                [F::ZERO; DIGEST_SIZE]
            };

            state[..DIGEST_SIZE].copy_from_slice(&digest);
            let input = state;
            sub_chip.permute_mut(&mut state);
            input
        })
        .collect_vec();

    DagCommitInfo {
        commit: from_fn(|i| state[i]),
        poseidon2_inputs,
    }
}
