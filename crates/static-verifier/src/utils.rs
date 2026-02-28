use halo2_base::{
    AssignedValue, Context,
    gates::{RangeInstructions, range::RangeChip},
};

use crate::circuit::Fr;

#[inline]
pub(crate) fn bits_for_u64(value: u64) -> usize {
    (64 - value.leading_zeros() as usize).max(1)
}

#[inline]
pub(crate) fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).expect("usize value does not fit in u64")
}

pub(crate) fn assign_and_range_u64(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    value: u64,
) -> AssignedValue<Fr> {
    let cell = ctx.load_witness(Fr::from(value));
    range.range_check(ctx, cell, bits_for_u64(value));
    cell
}

pub(crate) fn assign_and_range_usize(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    value: usize,
) -> AssignedValue<Fr> {
    assign_and_range_u64(ctx, range, usize_to_u64(value))
}
