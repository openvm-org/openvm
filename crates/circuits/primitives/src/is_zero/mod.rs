use derive_new::new;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{p3_air::AirBuilder, p3_field::Field};

use crate::{SubAir, TraceSubRowGenerator};

#[cfg(test)]
pub mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, new)]
pub struct IsZeroIo<T> {
    pub x: T,
    /// The boolean output, constrained to equal (x == 0) when `condition != 0`..
    pub out: T,
    /// Constraints only hold when `condition != 0`. When `condition == 0`, setting all trace
    /// values to zero still passes the constraints.
    pub condition: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Debug, new)]
pub struct IsZeroAuxCols<T> {
    pub inv: T,
}

/// An Air that constraints that checks if a number equals 0
#[derive(Copy, Clone)]
pub struct IsZeroSubAir;

impl<AB: AirBuilder> SubAir<AB> for IsZeroSubAir {
    /// (io, inv)
    type AirContext<'a>
        = (IsZeroIo<AB::Expr>, AB::Var)
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, (io, inv): (IsZeroIo<AB::Expr>, AB::Var))
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        // We always assert this, even when `condition == 0`, because x = 0, out = 0 will pass.
        builder.assert_zero(io.x.clone() * io.out.clone());
        builder.when(io.condition).assert_one(io.out + io.x * inv);
    }
}

impl<F: Field> TraceSubRowGenerator<F> for IsZeroSubAir {
    /// `x`
    type TraceContext<'a> = F;
    /// `(inv, out)`
    type ColsMut<'a> = (&'a mut F, &'a mut F);

    fn generate_subrow<'a>(&'a self, x: F, (inv, out): (&'a mut F, &'a mut F)) {
        *out = F::from_bool(x.is_zero());
        *inv = x.try_inverse().unwrap_or(F::ZERO);
    }
}
