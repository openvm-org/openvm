use itertools::izip;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{p3_air::AirBuilder, p3_field::Field};

use crate::{SubAir, TraceSubRowGenerator};

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IsEqArrayIo<T, const NUM: usize> {
    pub x: [T; NUM],
    pub y: [T; NUM],
    /// The boolean output, constrained to equal (x == y) when `condition != 0`.
    pub out: T,
    /// Constraints only hold when `condition != 0`. When `condition == 0`, setting all trace
    /// values to zero still passes the constraints.
    pub condition: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct IsEqArrayAuxCols<T, const NUM: usize> {
    // `diff_inv_marker` is filled with 0 except at the lowest index i such that
    // `x[i] != y[i]`. If such an `i` exists `diff_inv_marker[i]` is the inverse of `x[i] - y[i]`.
    pub diff_inv_marker: [T; NUM],
}

#[derive(Clone, Copy, Debug)]
pub struct IsEqArraySubAir<const NUM: usize>;

impl<AB: AirBuilder, const NUM: usize> SubAir<AB> for IsEqArraySubAir<NUM> {
    /// `(io, diff_inv_marker)`
    type AirContext<'a>
        = (IsEqArrayIo<AB::Expr, NUM>, [AB::Var; NUM])
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    /// Constrain that out == (x == y) when condition != 0
    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, diff_inv_marker): (IsEqArrayIo<AB::Expr, NUM>, [AB::Var; NUM]),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let mut sum = io.out.clone();
        // If x == y: then sum == 1 implies out = 1.
        // If x != y: then out * (x[i] - y[i]) == 0 implies out = 0.
        //            to get the sum == 1 to be satisfied, we set diff_inv_marker[i] = (x[i] -
        // y[i])^{-1} at the first index i such that x[i] != y[i].
        for (x_i, y_i, inv_marker_i) in izip!(io.x, io.y, diff_inv_marker) {
            sum += (x_i.clone() - y_i.clone()) * inv_marker_i;
            builder.assert_zero(io.out.clone() * (x_i - y_i));
        }
        builder.when(io.condition).assert_one(sum);
        builder.assert_bool(io.out);
    }
}

impl<F: Field, const NUM: usize> TraceSubRowGenerator<F> for IsEqArraySubAir<NUM> {
    /// (x, y)
    type TraceContext<'a> = (&'a [F; NUM], &'a [F; NUM]);
    /// (diff_inv_marker, out)
    type ColsMut<'a> = (&'a mut [F; NUM], &'a mut F);

    #[inline(always)]
    fn generate_subrow<'a>(
        &'a self,
        (x, y): (&'a [F; NUM], &'a [F; NUM]),
        (diff_inv_marker, out): (&'a mut [F; NUM], &'a mut F),
    ) {
        let mut is_eq = true;
        for (x_i, y_i, diff_inv_marker_i) in izip!(x, y, diff_inv_marker) {
            if x_i != y_i && is_eq {
                is_eq = false;
                *diff_inv_marker_i = (*x_i - *y_i).inverse();
            } else {
                *diff_inv_marker_i = F::ZERO;
            }
        }
        *out = F::from_bool(is_eq);
    }
}
