use openvm_stark_sdk::openvm_stark_backend::p3_field::PrimeCharacteristicRing;

use crate::{
    chip_traits::BabyBearExt4Inst,
    field::baby_bear::{BabyBearExtWire, BabyBearWire},
    RootF,
};

#[allow(clippy::type_complexity)]
pub(crate) fn column_openings_by_rot_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    openings: &[BabyBearExtWire<B::F>],
    need_rot: bool,
) -> Vec<ExtWirePair<B::F>> {
    if need_rot {
        assert!(
            openings.len().is_multiple_of(2),
            "rotated opening vector must be even",
        );
        openings
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect::<Vec<_>>()
    } else {
        let zero = b.ext_zero();
        openings
            .iter()
            .map(|opening| (*opening, zero))
            .collect::<Vec<_>>()
    }
}

pub(crate) fn horner_eval_ext_poly_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    coeffs: &[BabyBearExtWire<B::F>],
    x: &BabyBearExtWire<B::F>,
) -> BabyBearExtWire<B::F> {
    if coeffs.is_empty() {
        return b.ext_zero();
    }
    // Pre-reduce x so that ext_mul doesn't redundantly reduce the same
    // high-max_bits components on every Horner step.
    let x_reduced = b.ext_reduce_max_bits(*x);
    let mut acc = *coeffs.last().unwrap();
    for coeff in coeffs.iter().rev().skip(1) {
        acc = b.ext_mul(acc, x_reduced);
        acc = b.ext_add(acc, *coeff);
    }
    acc
}

pub(crate) fn horner_eval_ext_poly_f_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    coeffs: &[BabyBearExtWire<B::F>],
    x: &BabyBearWire<B::F>,
) -> BabyBearExtWire<B::F> {
    if coeffs.is_empty() {
        return b.ext_zero();
    }
    // Pre-reduce x so that each mul_add step inside the loop doesn't redundantly
    // reduce the same high-max_bits value on every iteration.
    let x_reduced = b.bb_reduce_max_bits(*x);
    let mut acc = *coeffs.last().unwrap();
    for coeff in coeffs.iter().rev().skip(1) {
        acc = b.ext_scalar_mul_add(acc, x_reduced, *coeff);
    }
    acc
}

pub(crate) fn interpolate_quadratic_at_012_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    evals: [&BabyBearExtWire<B::F>; 3],
    x: &BabyBearExtWire<B::F>,
) -> BabyBearExtWire<B::F> {
    let one = b.ext_from_base_const(RootF::ONE);
    let two = b.ext_from_base_const(RootF::TWO);
    let inv_two = RootF::ONE.halve();

    let x_minus_one = b.ext_sub(*x, one);
    let x_minus_two = b.ext_sub(*x, two);
    let x_times_x_minus_one = b.ext_mul(*x, x_minus_one);
    let x_times_x_minus_two = b.ext_mul(*x, x_minus_two);
    let x_minus_one_times_x_minus_two = b.ext_mul(x_minus_one, x_minus_two);

    let l0 = b.ext_mul_base_const(x_minus_one_times_x_minus_two, inv_two);
    let l1 = b.ext_neg(x_times_x_minus_two);
    let l2 = b.ext_mul_base_const(x_times_x_minus_one, inv_two);

    let term0 = b.ext_mul(*evals[0], l0);
    let term1 = b.ext_mul(*evals[1], l1);
    let term2 = b.ext_mul(*evals[2], l2);
    let sum01 = b.ext_add(term0, term1);
    b.ext_add(sum01, term2)
}
