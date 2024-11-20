use axvm_algebra::Field;

use super::{fp2_invert_assign, fp_sum_of_products, Bls12_381, Fp2};
use crate::pairing::PairingIntrinsics;

// pub(crate) fn fp6_invert_assign(c: &mut [Fp2; 3]) {
//     let mut c0 = &c[1] * &c[2];
//     c0 *= &Bls12_381::XI;
//     let c0_sub = c0.clone();
//     c0.square_assign();
//     c0 -= &c0_sub;

//     let mut c1 = c[2].clone();
//     c1.square_assign();
//     c1 *= &Bls12_381::XI;
//     let c0c1 = &c[0] * &c[1];
//     c1 -= &c0c1;

//     let mut c2 = c[1].clone();
//     c2.square_assign();
//     let c0c2 = &c[0] * &c[2];
//     c2 -= &c0c2;

//     let mut tmp = (&c[1] * &c2) + (&c[2] * &c1);
//     tmp *= &Bls12_381::XI;
//     let sc0c0 = &c[0] * &c0;
//     tmp += &sc0c0;

//     fp2_invert_assign(&mut tmp);
//     c[0] *= &tmp;
//     c[1] *= &tmp;
//     c[2] *= &tmp;
// }

pub(crate) fn fp6_invert_assign(c: &mut [Fp2; 3]) {
    let mut c0 = c[2].clone();
    c0 *= &Bls12_381::XI;
    c0 *= &c[1];
    c0.neg_assign();
    {
        let mut c0s = c[0].clone();
        c0s.square_assign();
        c0 += &c0s;
    }
    let mut c1 = c[2].clone();
    c1.square_assign();
    c1 *= &Bls12_381::XI;
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
    tmp1 *= &Bls12_381::XI;
    tmp2 = c[0].clone();
    tmp2 *= &c0;
    tmp1 += &tmp2;

    fp2_invert_assign(&mut tmp1);
    let mut tmp = [tmp1.clone(), tmp1.clone(), tmp1.clone()];
    tmp[0] *= &c0;
    tmp[1] *= &c1;
    tmp[2] *= &c2;

    *c = tmp;
}

pub(crate) fn fp6_mul_by_nonresidue_assign(c: &mut [Fp2; 3]) {
    // c0, c1, c2 -> c2, c0, c1
    c.swap(0, 1);
    c.swap(0, 2);
    c[0] *= &Bls12_381::XI;
}

pub(crate) fn fp6_sub_assign(a: &mut [Fp2; 3], b: &[Fp2; 3]) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a -= b);
}

/// Squares 3 elements of `Fp2`, which represents as a single Fp6 element, in place
pub(crate) fn fp6_square_assign(c: &mut [Fp2; 3]) {
    let mut s0 = c[0].clone();
    s0.square_assign();
    let mut ab = &c[0] * &c[1];
    ab.double_assign();
    let s1 = ab;
    let mut s2 = &c[0] - &c[1] + &c[2];
    s2.square_assign();
    let mut bc = &c[1] * &c[2];
    bc.double_assign();
    let mut s3 = bc;
    let mut s4 = c[2].clone();
    s4.square_assign();

    s3 *= &Bls12_381::XI;
    s3 += &s0;
    s4 *= &Bls12_381::XI;
    s4 += &s1;

    c[0] = s3.clone();
    c[1] = s4.clone();
    c[2] = &s1 + &s2 + &s3 - &s0 - &s4;
}

pub(crate) fn fp6_mul_assign(a: &mut [Fp2; 3], b: &[Fp2; 3]) {
    // let a0 = a[0].clone();
    // let a1 = a[1].clone();
    // let a2 = a[2].clone();

    // // Calculate intermediate values
    // let b10_p_b11 = &b[1].c0 + &b[1].c1;
    // let b10_m_b11 = &b[1].c0 - &b[1].c1;
    // let b20_p_b21 = &b[2].c0 + &b[2].c1;
    // let b20_m_b21 = &b[2].c0 - &b[2].c1;

    // // Calculate c0
    // a[0].c0 = fp_sum_of_products(
    //     [&a0.c0, &-a0.c1, &a1.c0, &-a1.c1, &a2.c0, &-a2.c1],
    //     [
    //         &b[0].c0, &b[0].c1, &b20_m_b21, &b20_p_b21, &b10_m_b11, &b10_p_b11,
    //     ],
    // );
    // a[0].c1 = fp_sum_of_products(
    //     [&a0.c0, &a0.c1, &a1.c0, &a1.c1, &a2.c0, &a2.c1],
    //     [
    //         &b[0].c1, &b[0].c0, &b20_p_b21, &b20_m_b21, &b10_p_b11, &b10_m_b11,
    //     ],
    // );

    // // Calculate c1
    // a[1].c0 = fp_sum_of_products(
    //     [&a0.c0, &-a0.c1, &a1.c0, &-a1.c1, &a2.c0, &-a2.c1],
    //     [
    //         &b[1].c0, &b[1].c1, &b[0].c0, &b[0].c1, &b20_m_b21, &b20_p_b21,
    //     ],
    // );
    // a[1].c1 = fp_sum_of_products(
    //     [&a0.c0, &a0.c1, &a1.c0, &a1.c1, &a2.c0, &a2.c1],
    //     [
    //         &b[1].c1, &b[1].c0, &b[0].c1, &b[0].c0, &b20_p_b21, &b20_m_b21,
    //     ],
    // );

    // // Calculate c2
    // a[2].c0 = fp_sum_of_products(
    //     [&a0.c0, &-a0.c1, &a1.c0, &-a1.c1, &a2.c0, &-a2.c1],
    //     [&b[2].c0, &b[2].c1, &b[1].c0, &b[1].c1, &b[0].c0, &b[0].c1],
    // );
    // a[2].c1 = fp_sum_of_products(
    //     [&a0.c0, &a0.c1, &a1.c0, &a1.c1, &a2.c0, &a2.c1],
    //     [&b[2].c1, &b[2].c0, &b[1].c1, &b[1].c0, &b[0].c1, &b[0].c0],
    // );
    todo!()
}

pub(crate) fn fp6_neg_assign(a: &mut [Fp2; 3]) {
    a.iter_mut().for_each(|x| x.neg_assign());
}
