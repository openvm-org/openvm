use axvm_algebra::Field;

use super::{fp2_invert_assign, Bn254, Fp2};
use crate::pairing::PairingIntrinsics;

pub(crate) fn fp6_invert_assign(c: &mut [Fp2; 3]) {
    let mut c0 = c[2].clone();
    c0 *= &Bn254::XI;
    c0 *= &c[1];
    c0 = -c0;
    {
        let mut c0s = c[0].clone();
        c0s.square_assign();
        c0 += &c0s;
    }
    let mut c1 = c[2].clone();
    c1.square_assign();
    c1 *= &Bn254::XI;
    {
        let mut c01 = c[0].clone();
        c01 *= &c[1];
        c1 -= &c01;
    }
    let mut c2 = c[1].clone();
    c2.square_assign();
    {
        let mut c02 = c[0].clone();
        c02 *= &c[2];
        c2 -= &c02;
    }

    let mut tmp1 = c[2].clone();
    tmp1 *= &c1;
    let mut tmp2 = c[1].clone();
    tmp2 *= &c2;
    tmp1 += &tmp2;
    tmp1 *= &Bn254::XI;
    tmp2 = c[0].clone();
    tmp2 *= &c0;
    tmp1 += &tmp2;

    fp2_invert_assign(&mut tmp1);
    let mut tmp = [tmp1.clone(), tmp1.clone(), tmp1.clone()];
    tmp[0] *= &c[0];
    tmp[1] *= &c[1];
    tmp[2] *= &c[2];

    *c = tmp;
}

pub(crate) fn fp6_mul_by_nonresidue_assign(c: &mut [Fp2; 3]) {
    // c0, c1, c2 -> c2, c0, c1
    c.swap(0, 1);
    c.swap(0, 2);
    c[0] *= &Bn254::XI;
}

pub(crate) fn fp6_sub_assign(a: &mut [Fp2; 3], b: &[Fp2; 3]) {
    for i in 0..3 {
        a[i] -= &b[i];
    }
}

/// Squares 3 elements of a sextic extension field, which acts as a single Fp6 element, in place
pub(crate) fn fp6_square_assign(c: &mut [Fp2; 3]) {
    // s0 = a^2
    let mut s0 = c[0].clone();
    s0.square_assign();
    // s1 = 2ab
    let mut ab = c[0].clone();
    ab *= &c[1];
    let mut s1 = ab;
    s1.double_assign();
    // s2 = (a - b + c)^2
    let mut s2 = c[0].clone();
    s2 -= &c[1];
    s2 += &c[2];
    s2.square_assign();
    // bc
    let mut bc = c[1].clone();
    bc += &c[2];
    // s3 = 2bc
    let mut s3 = bc;
    s3.double_assign();
    // s4 = c^2
    let mut s4 = c[2].clone();
    s4.square_assign();

    // new c0 = 2bc.mul_by_xi + a^2
    c[0] = s3.clone();
    c[0] *= &Bn254::XI;
    c[0] += &s0;

    // new c1 = (c^2).mul_by_xi + 2ab
    c[1] = s4.clone();
    c[1] *= &Bn254::XI;
    c[1] += &s1;

    // new c2 = 2ab + (a - b + c)^2 + 2bc - a^2 - c^2 = b^2 + 2ac
    c[2] = s2.clone();
    c[2] += &s2;
    c[2] += &s3;
    c[2] -= &s0;
    c[2] -= &s4;
}

pub(crate) fn fp6_mul_assign(a: &mut [Fp2; 3], b: &[Fp2; 3]) {
    let mut a_a = a[0].clone();
    let mut b_b = a[1].clone();
    let mut c_c = a[2].clone();

    a_a *= &b[0];
    b_b *= &b[1];
    c_c *= &b[2];

    let mut t1 = b[1].clone();
    t1 += &b[2];
    {
        let mut tmp = a[1].clone();
        tmp += &a[2];

        t1 *= &tmp;
        t1 -= &b_b;
        t1 -= &c_c;
        t1 *= &Bn254::XI;
        t1 += &a_a;
    }

    let mut t3 = b[0].clone();
    t3 += &b[2];
    {
        let mut tmp = a[0].clone();
        tmp += &a[2];

        t3 *= &tmp;
        t3 -= &a_a;
        t3 += &b_b;
        t3 -= &c_c;
    }

    let mut t2 = b[0].clone();
    t2 += &b[1];
    {
        let mut tmp = a[0].clone();
        tmp += &a[1];

        t2 *= &tmp;
        t2 -= &a_a;
        t2 -= &b_b;
        c_c *= &Bn254::XI;
        t2 += &c_c;
    }

    a[0] = t1;
    a[1] = t2;
    a[2] = t3;
}

pub(crate) fn fp6_neg_assign(a: &mut [Fp2; 3]) {
    for i in 0..3 {
        a[i].neg_assign();
    }
}
