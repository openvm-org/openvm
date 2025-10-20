use p3_air::AirBuilder;

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
