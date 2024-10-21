use afs_derive::AlignedBorrow;
use derive_new::new;
use p3_air::AirBuilder;
use p3_field::Field;

use crate::{SubAir, TraceSubRowGenerator};

#[cfg(test)]
pub mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, new)]
pub struct IsZeroIo<T> {
    pub x: T,
    /// The boolean output, constrained to equal (x == 0).
    pub out: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Debug, new)]
pub struct IsZeroAuxCols<T> {
    pub inv: T,
}

/// An Air that constraints that checks if a number equals 0
#[derive(Copy, Clone)]
pub struct IsZeroAir;

impl<AB: AirBuilder> SubAir<AB> for IsZeroAir {
    /// (io, inv)
    type AirContext<'a> = (IsZeroIo<AB::Expr>, AB::Var) where AB::Expr: 'a, AB::Var: 'a, AB: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, (io, inv): (IsZeroIo<AB::Expr>, AB::Var))
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        builder.assert_zero(io.x.clone() * io.out.clone());
        builder.assert_one(io.out + io.x * inv);
    }
}

impl<F: Field> TraceSubRowGenerator<F> for IsZeroAir {
    /// `x`
    type TraceContext<'a> = F;
    /// `inv`
    type ColsMut<'a> = &'a mut F;

    fn generate_subrow<'a>(&'a self, x: F, inv: &'a mut F) {
        *inv = x.try_inverse().unwrap_or(F::zero());
    }
}
