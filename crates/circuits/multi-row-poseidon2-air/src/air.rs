use std::{borrow::Borrow, marker::PhantomData};

use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::Matrix,
};
use p3_poseidon2::GenericPoseidon2LinearLayers;

use crate::{
    columns::{num_cols, MultiRowPoseidon2Cols, SBox},
    constants::RoundConstants,
};

/// Assumes the field size is at least 16 bits.
#[derive(Debug)]
pub struct MultiRowPoseidon2Air<
    F: Field,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    _phantom: PhantomData<LinearLayers>,
}

impl<
        F: Field,
        LinearLayers,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    >
    MultiRowPoseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    pub fn new(constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>) -> Self {
        Self {
            constants,
            _phantom: PhantomData,
        }
    }
}

impl<
        F: Field,
        LinearLayers: Sync,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > BaseAir<F>
    for MultiRowPoseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
    }
}

impl<
        AB: AirBuilder,
        LinearLayers: GenericPoseidon2LinearLayers<AB::Expr, WIDTH>,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > Air<AB>
    for MultiRowPoseidon2Air<
        AB::F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MultiRowPoseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = (*local).borrow();
        let next = main.row_slice(1);
        let next_cols: &MultiRowPoseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = (*next).borrow();
        let mut state: [AB::Expr; WIDTH] = local.cur_state.map(|x| x.into());
        for (s, c) in state.iter_mut().zip(local.constants) {
            *s += c.into();
        }
        let pre_state = state.clone();
        for (state_i, sbox_post) in state.iter_mut().zip(local.sbox.iter()) {
            eval_sbox(&sbox_post.sbox, state_i, builder);
            builder.assert_eq(state_i.clone(), sbox_post.post_sbox);
            *state_i = sbox_post.post_sbox.into();
        }
        // state is now deg 2
        for (i, (s, p)) in state.iter_mut().zip(pre_state.iter()).enumerate() {
            if i == 0 {
                *s = (local.is_full + local.is_partial) * s.clone()
                    + (local.is_beginning) * p.clone();
            } else {
                *s =
                    local.is_full * s.clone() + (local.is_beginning + local.is_partial) * p.clone();
            }
        }
        let mut e_state = state.clone();
        let mut i_state = state.clone();
        LinearLayers::external_linear_layer(&mut e_state);
        LinearLayers::internal_linear_layer(&mut i_state);
        for ((e, i), p) in e_state
            .iter()
            .zip(i_state.iter())
            .zip(local.end_state.iter())
        {
            builder.assert_eq(
                (local.is_full + local.is_beginning) * e.clone() + local.is_partial * i.clone(),
                *p,
            );
        }
        for (l, n) in local.end_state.iter().zip(next_cols.cur_state.iter()) {
            builder.assert_zero((*l - *n) * (AB::Expr::ONE - next_cols.is_beginning));
        }
        for (l, n) in local.inputs.iter().zip(next_cols.inputs.iter()) {
            builder.assert_zero((*l - *n) * (AB::Expr::ONE - next_cols.is_beginning));
        }
        for (l, n) in local.inputs.iter().zip(local.cur_state.iter()) {
            builder.assert_zero((*l - *n) * local.is_beginning);
        }
    }
}

/// Evaluates the S-box over a degree-1 expression `x`.
///
/// # Panics
///
/// This method panics if the number of `REGISTERS` is not chosen optimally for the given
/// `DEGREE` or if the `DEGREE` is not supported by the S-box. The supported degrees are
/// `3`, `5`, `7`, and `11`.
#[inline]
fn eval_sbox<AB, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &SBox<AB::Var, DEGREE, REGISTERS>,
    x: &mut AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let committed_x3 = sbox.0[0].into();
            let x2 = x.square();
            builder.assert_eq(committed_x3.clone(), x2.clone() * x.clone());
            committed_x3 * x2
        }
        (7, 1) => {
            let committed_x3 = sbox.0[0].into();
            builder.assert_eq(committed_x3.clone(), x.cube());
            committed_x3.square() * x.clone()
        }
        (11, 2) => {
            let committed_x3 = sbox.0[0].into();
            let committed_x9 = sbox.0[1].into();
            let x2 = x.square();
            builder.assert_eq(committed_x3.clone(), x2.clone() * x.clone());
            builder.assert_eq(committed_x9.clone(), committed_x3.cube());
            committed_x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}
