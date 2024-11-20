mod fp12;
mod fp2;
mod fp6;

use axvm_algebra::{DivUnsafe, Field};
pub(crate) use fp12::*;
pub(crate) use fp2::*;
pub(crate) use fp6::*;

// Inverse z = x⁻¹ (mod p)
pub(crate) fn fp_invert_assign<F: Field + DivUnsafe<F>>(x: &mut F) {
    let res = F::ONE.div_unsafe(&*x);
    *x = res;
}
