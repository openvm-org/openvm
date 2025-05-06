use openvm_algebra_guest::{field::FieldExtension, Field, IntMod};

use super::fp2_invert_assign;

pub(crate) fn fp6_invert_assign<
    Fp: IntMod + Field,
    Fp2: Field + FieldExtension<Fp, Coeffs = [Fp; 2]>,
>(
    c: &mut [Fp2; 3],
    xi: &Fp2,
) {
    let mut c0 = c[2].clone();
    c0 *= xi;
    c0 *= &c[1];
    c0 = c0.neg();
    {
        let mut c0s = c[0].clone();
        <Fp2 as Field>::square_assign(&mut c0s);
        c0 += &c0s;
    }
    let mut c1 = c[2].clone();
    <Fp2 as Field>::square_assign(&mut c1);
    c1 *= xi;
    {
        let mut c01 = c[0].clone();
        c01 *= &c[1];
        c1 -= &c01;
    }
    let mut c2 = c[1].clone();
    <Fp2 as Field>::square_assign(&mut c2);
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
    tmp1 *= xi;
    tmp2 = c[0].clone();
    tmp2 *= &c0;
    tmp1 += &tmp2;

    let mut coeffs = tmp1.clone().to_coeffs();
    fp2_invert_assign::<Fp>(&mut coeffs);
    let tmp = Fp2::from_coeffs(coeffs);
    let mut tmp = [tmp.clone(), tmp.clone(), tmp.clone()];
    tmp[0] *= &c0;
    tmp[1] *= &c1;
    tmp[2] *= &c2;

    *c = tmp;
}

pub(crate) fn fp6_mul_by_nonresidue_assign<
    Fp: IntMod + Field,
    Fp2: Field + FieldExtension<Fp, Coeffs = [Fp; 2]>,
>(
    c: &mut [Fp2; 3],
    xi: &Fp2,
) {
    // c0, c1, c2 -> c2, c0, c1
    c.swap(0, 1);
    c.swap(0, 2);
    c[0] *= xi;
}

pub(crate) fn fp6_sub_assign<
    Fp: IntMod + Field,
    Fp2: Field + FieldExtension<Fp, Coeffs = [Fp; 2]>,
>(
    a: &mut [Fp2; 3],
    b: &[Fp2; 3],
) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a -= b);
}

/// Squares 3 elements of `Fp2`, which represents as a single Fp6 element, in place
pub(crate) fn fp6_square_assign<
    Fp: IntMod + Field,
    Fp2: Field + FieldExtension<Fp, Coeffs = [Fp; 2]>,
>(
    c: &mut [Fp2; 3],
    xi: &Fp2,
) {
    // s0 = a^2
    let mut s0 = c[0].clone();
    <Fp2 as Field>::square_assign(&mut s0);
    // s1 = 2ab
    let mut ab = c[0].clone();
    ab *= &c[1];
    let mut s1 = ab;
    <Fp2 as Field>::double_assign(&mut s1);
    // s2 = (a - b + c)^2
    let mut s2 = c[0].clone();
    s2 -= &c[1];
    s2 += &c[2];
    <Fp2 as Field>::square_assign(&mut s2);
    // bc
    let mut bc = c[1].clone();
    bc *= &c[2];
    // s3 = 2bc
    let mut s3 = bc;
    <Fp2 as Field>::double_assign(&mut s3);
    // s4 = c^2
    let mut s4 = c[2].clone();
    <Fp2 as Field>::square_assign(&mut s4);

    // new c0 = 2bc.mul_by_xi + a^2
    c[0] = s3.clone();
    c[0] *= xi;
    c[0] += &s0;

    // new c1 = (c^2).mul_by_xi + 2ab
    c[1] = s4.clone();
    c[1] *= xi;
    c[1] += &s1;

    // new c2 = 2ab + (a - b + c)^2 + 2bc - a^2 - c^2 = b^2 + 2ac
    c[2] = s1;
    c[2] += &s2;
    c[2] += &s3;
    c[2] -= &s0;
    c[2] -= &s4;
}

pub(crate) fn fp6_mul_assign<
    Fp: IntMod + Field,
    Fp2: Field + FieldExtension<Fp, Coeffs = [Fp; 2]>,
>(
    a: &mut [Fp2; 3],
    b: &[Fp2; 3],
    xi: &Fp2,
) {
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
        t1 *= xi;
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
        c_c *= xi;
        t2 += &c_c;
    }

    a[0] = t1;
    a[1] = t2;
    a[2] = t3;
}
