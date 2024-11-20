use axvm_algebra::{Field, IntMod};

use super::fp_invert_assign;

pub(crate) fn fp2_invert_assign<F: Field + IntMod>(c: &mut [F; 2]) {
    let mut t1 = c[1].clone();
    <F as Field>::square_assign(&mut t1);
    let mut t0 = c[0].clone();
    <F as Field>::square_assign(&mut t0);
    t0 += &t1;
    fp_invert_assign(&mut t0);
    let mut tmp = [c[0].clone(), c[1].clone()];
    tmp[0] *= &t0;
    tmp[1] *= &t0;
    tmp[1].neg_assign();

    *c = tmp;
}
