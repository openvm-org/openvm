use p3_air::AirBuilder;
use p3_field::FieldAlgebra;

use crate::bus::TranscriptBusMessage;

pub fn transcript_messages<F, const N: usize>(
    tidx: impl Into<F>,
    value: [impl Into<F>; N],
    is_sample: impl Into<F>,
) -> [TranscriptBusMessage<F>; N]
where
    F: FieldAlgebra,
{
    let tidx = tidx.into();
    let value = value.map(Into::into);
    let is_sample = is_sample.into();

    core::array::from_fn(|j| TranscriptBusMessage {
        tidx: tidx.clone() + F::from_canonical_usize(j),
        value: value[j].clone(),
        is_sample: is_sample.clone(),
    })
}

// TODO(ayush): move to a custom air builder
pub fn assert_zeros<AB, const N: usize>(builder: &mut AB, array: [impl Into<AB::Expr>; N])
where
    AB: AirBuilder,
{
    for elem in array.into_iter() {
        builder.assert_zero(elem);
    }
}

pub fn assert_eq_array<AB, const N: usize>(
    builder: &mut AB,
    actual: [impl Into<AB::Expr>; N],
    expected: [impl Into<AB::Expr>; N],
) where
    AB: AirBuilder,
{
    for (a, e) in actual.into_iter().zip(expected.into_iter()) {
        builder.assert_eq(a, e);
    }
}
