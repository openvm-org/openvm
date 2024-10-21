use derive_new::new;
use p3_air::AirBuilder;
use p3_field::Field;

use crate::{
    is_zero::{IsZeroAir, IsZeroAuxCols, IsZeroIo},
    SubAir, TraceSubRowGenerator,
};

#[cfg(test)]
pub mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, new)]
pub struct IsEqualIo<T> {
    pub x: T,
    pub y: T,
    /// The boolean output, constrained to equal (x == y).
    pub out: T,
}

pub type IsEqualAuxCols<T> = IsZeroAuxCols<T>;

/// An Air that constrains `out = (x == y)`.
#[derive(Copy, Clone)]
pub struct IsEqualAir;

impl<AB: AirBuilder> SubAir<AB> for IsEqualAir {
    /// (io, inv)
    type AirContext<'a> = (IsEqualIo<AB::Expr>, AB::Var) where AB::Expr: 'a, AB::Var: 'a, AB: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, (io, inv): (IsEqualIo<AB::Expr>, AB::Var))
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let is_zero_io = IsZeroIo::new(io.x - io.y, io.out);
        IsZeroAir.eval(builder, (is_zero_io, inv));
    }
}

impl<F: Field> TraceSubRowGenerator<F> for IsEqualAir {
    /// `x, y`
    type TraceContext<'a> = (F, F);
    /// `inv`
    type ColsMut<'a> = &'a mut F;

    fn generate_subrow<'a>(&'a self, (x, y): (F, F), inv: &'a mut F) {
        IsZeroAir.generate_subrow(x - y, inv);
    }
}
