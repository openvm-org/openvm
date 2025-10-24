use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, extension::BinomiallyExtendable};
use stark_backend_v2::D_EF;

pub fn base_to_ext<FA>(x: impl Into<FA>) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    [x.into(), FA::ZERO, FA::ZERO, FA::ZERO]
}

// TODO(ayush): move somewhere else
pub const MAX_CONSTRAINT_DEGREE: usize = 4;

pub fn ext_field_one_minus<FA>(x: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    [FA::ONE - x0, -x1, -x2, -x3]
}

pub fn ext_field_add<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let [y0, y1, y2, y3] = y.map(Into::into);
    [x0 + y0, x1 + y1, x2 + y2, x3 + y3]
}

pub fn ext_field_subtract<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let [y0, y1, y2, y3] = y.map(Into::into);
    [x0 - y0, x1 - y1, x2 - y2, x3 - y3]
}

pub fn ext_field_multiply<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D_EF>,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let [y0, y1, y2, y3] = y.map(Into::into);

    let w = FA::from_f(FA::F::W);

    let z0_beta_terms = x1.clone() * y3.clone() + x2.clone() * y2.clone() + x3.clone() * y1.clone();
    let z1_beta_terms = x2.clone() * y3.clone() + x3.clone() * y2.clone();
    let z2_beta_terms = x3.clone() * y3.clone();

    [
        x0.clone() * y0.clone() + z0_beta_terms * w.clone(),
        x0.clone() * y1.clone() + x1.clone() * y0.clone() + z1_beta_terms * w.clone(),
        x0.clone() * y2.clone()
            + x1.clone() * y1.clone()
            + x2.clone() * y0.clone()
            + z2_beta_terms * w,
        x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0,
    ]
}

pub fn ext_field_add_scalar<FA>(x: [impl Into<FA>; D_EF], y: impl Into<FA>) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    [x0 + y.into(), x1, x2, x3]
}

pub fn ext_field_subtract_scalar<FA>(x: [impl Into<FA>; D_EF], y: impl Into<FA>) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    [x0 - y.into(), x1, x2, x3]
}

pub fn ext_field_multiply_scalar<FA>(x: [impl Into<FA>; D_EF], y: impl Into<FA>) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let y = y.into();
    [x0 * y.clone(), x1 * y.clone(), x2 * y.clone(), x3 * y]
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

pub fn assert_one_ext<AB>(builder: &mut AB, array: [impl Into<AB::Expr>; D_EF])
where
    AB: AirBuilder,
{
    for (i, elem) in array.into_iter().enumerate() {
        if i == 0 {
            builder.assert_one(elem);
        } else {
            builder.assert_zero(elem);
        }
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
