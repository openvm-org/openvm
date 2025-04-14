use derive_new::new;
use openvm_stark_backend::{p3_air::AirBuilder, p3_field::Field};

use crate::{
    is_zero::{IsZeroAuxCols, IsZeroIo, IsZeroSubAir},
    SubAir, TraceSubRowGenerator,
};

#[cfg(test)]
pub mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, new)]
pub struct IsEqualIo<T> {
    pub x: T,
    pub y: T,
    /// The boolean output, constrained to equal (x == y), when `condition != 0`.
    pub out: T,
    /// Constraints only hold when `condition != 0`. When `condition == 0`, setting all trace
    /// values to zero still passes the constraints.
    pub condition: T,
}

pub type IsEqualAuxCols<T> = IsZeroAuxCols<T>;

/// An Air that constrains `out = (x == y)`.
#[derive(Copy, Clone)]
pub struct IsEqSubAir;

impl<AB: AirBuilder> SubAir<AB> for IsEqSubAir {
    /// (io, inv)
    type AirContext<'a>
        = (IsEqualIo<AB::Expr>, AB::Var)
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, (io, inv): (IsEqualIo<AB::Expr>, AB::Var))
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let is_zero_io = IsZeroIo::new(io.x - io.y, io.out, io.condition);
        IsZeroSubAir.eval(builder, (is_zero_io, inv));
    }
}

impl<F: Field> TraceSubRowGenerator<F> for IsEqSubAir {
    /// `(x, y)`
    type TraceContext<'a> = (F, F);
    /// `(inv, out)`
    type ColsMut<'a> = (&'a mut F, &'a mut F);

    fn generate_subrow<'a>(&'a self, (x, y): (F, F), (inv, out): (&'a mut F, &'a mut F)) {
        IsZeroSubAir.generate_subrow(x - y, (inv, out));
    }
}
