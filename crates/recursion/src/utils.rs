use std::ops::Index;

use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, extension::BinomiallyExtendable};
use stark_backend_v2::D_EF;

// TODO(ayush): move somewhere else
pub const MAX_CONSTRAINT_DEGREE: usize = 4;

pub fn base_to_ext<FA>(x: impl Into<FA>) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    [x.into(), FA::ZERO, FA::ZERO, FA::ZERO]
}

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

pub fn scalar_subtract_ext_field<FA>(x: impl Into<FA>, y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    let [y0, y1, y2, y3] = y.map(Into::into);
    [x.into() - y0, -y1, -y2, -y3]
}

pub fn ext_field_multiply_scalar<FA>(x: [impl Into<FA>; D_EF], y: impl Into<FA>) -> [FA; D_EF]
where
    FA: FieldAlgebra,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let y = y.into();
    [x0 * y.clone(), x1 * y.clone(), x2 * y.clone(), x3 * y]
}

pub fn eq_1<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D_EF>,
{
    let x = x.map(Into::into);
    let y = y.map(Into::into);

    let xy = ext_field_multiply::<FA>(x.clone(), y.clone());
    let two_xy = ext_field_multiply_scalar::<FA>(xy, FA::TWO);
    let x_plus_y = ext_field_add::<FA>(x, y);
    let one_minus_x_plus_y = scalar_subtract_ext_field::<FA>(FA::ONE, x_plus_y);
    ext_field_add(one_minus_x_plus_y, two_xy)
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

#[derive(Debug, Clone)]
pub(crate) struct MultiProofVecVec<T> {
    data: Vec<T>,
    bounds: Vec<usize>,
}

impl<T> MultiProofVecVec<T> {
    pub(crate) fn new() -> Self {
        Self {
            data: Vec::new(),
            bounds: vec![0],
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bounds: vec![0],
        }
    }

    pub(crate) fn push(&mut self, x: T) {
        self.data.push(x);
    }

    pub(crate) fn extend_from_slice(&mut self, slice: &[T])
    where
        T: Clone,
    {
        self.data.extend_from_slice(slice);
    }

    pub(crate) fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
        self.data.extend(iter);
    }

    pub(crate) fn end_proof(&mut self) {
        self.bounds.push(self.data.len());
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn num_proofs(&self) -> usize {
        self.bounds.len() - 1
    }

    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }
}

impl<T> Index<usize> for MultiProofVecVec<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.num_proofs());
        &self.data[self.bounds[index]..self.bounds[index + 1]]
    }
}

pub fn interpolate_quadratic<FA>(
    pre_claim: [impl Into<FA>; D_EF],
    ev1: [impl Into<FA>; D_EF],
    ev2: [impl Into<FA>; D_EF],
    alpha: [impl Into<FA>; D_EF],
) -> [FA; D_EF]
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D_EF>,
{
    let pre_claim = pre_claim.map(Into::into);
    let ev1 = ev1.map(Into::into);
    let ev2 = ev2.map(Into::into);
    let alpha = alpha.map(Into::into);

    let ev0 = ext_field_subtract::<FA>(pre_claim.clone(), ev1.clone());
    let s1 = ext_field_subtract::<FA>(ev1.clone(), ev0.clone());
    let s2 = ext_field_subtract::<FA>(ev2.clone(), ev1.clone());
    let p = ext_field_multiply::<FA>(
        ext_field_subtract::<FA>(s2, s1.clone()),
        base_to_ext::<FA>(FA::from_f(FA::F::TWO.inverse())),
    );
    let q = ext_field_subtract::<FA>(s1, p.clone());
    ext_field_add::<FA>(
        ev0,
        ext_field_multiply::<FA>(
            alpha.clone(),
            ext_field_add::<FA>(q, ext_field_multiply::<FA>(p, alpha)),
        ),
    )
}
