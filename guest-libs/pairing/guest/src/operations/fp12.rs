use openvm_algebra_guest::{field::FieldExtension, Field, IntMod};

use super::{
    fp6_invert_assign, fp6_mul_assign, fp6_mul_by_nonresidue_assign, fp6_square_assign,
    fp6_sub_assign,
};

pub(crate) fn fp12_invert_assign<
    Fp: IntMod + Field,
    Fp2: Field + FieldExtension<Fp, Coeffs = [Fp; 2]>,
>(
    c: &mut [Fp2; 6],
    xi: &Fp2,
) {
    let mut c0s = [c[0].clone(), c[2].clone(), c[4].clone()];
    let mut c1s = [c[1].clone(), c[3].clone(), c[5].clone()];

    fp6_square_assign(&mut c0s, xi);
    fp6_square_assign(&mut c1s, xi);
    fp6_mul_by_nonresidue_assign(&mut c1s, xi);
    fp6_sub_assign(&mut c0s, &c1s);

    fp6_invert_assign(&mut c0s, xi);
    let mut t0 = c0s.clone();
    let mut t1 = c0s;
    fp6_mul_assign(&mut t0, &[c[0].clone(), c[2].clone(), c[4].clone()], xi);
    fp6_mul_assign(&mut t1, &[c[1].clone(), c[3].clone(), c[5].clone()], xi);
    c[0] = t0[0].clone();
    c[2] = t0[1].clone();
    c[4] = t0[2].clone();
    c[1] = t1[0].clone().neg();
    c[3] = t1[1].clone().neg();
    c[5] = t1[2].clone().neg();
}
