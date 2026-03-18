use halo2_base::Context;
use openvm_stark_sdk::openvm_stark_backend::p3_field::PrimeCharacteristicRing;

use crate::{field::baby_bear::BabyBearExtChip, ChildF, Fr};

pub(crate) type BabyBearExtWire = crate::field::baby_bear::BabyBearExtWire;

pub(crate) fn column_openings_by_rot_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    openings: &[BabyBearExtWire],
    need_rot: bool,
) -> Vec<(BabyBearExtWire, BabyBearExtWire)> {
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
        let zero = ext_chip.zero(ctx);
        openings
            .iter()
            .map(|opening| (*opening, zero))
            .collect::<Vec<_>>()
    }
}

pub(crate) fn horner_eval_ext_poly_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    coeffs: &[BabyBearExtWire],
    x: &BabyBearExtWire,
) -> BabyBearExtWire {
    let mut acc = ext_chip.zero(ctx);
    for coeff in coeffs.iter().rev() {
        acc = ext_chip.mul(ctx, acc, *x);
        acc = ext_chip.add(ctx, acc, *coeff);
    }
    acc
}

pub(crate) fn interpolate_quadratic_at_012_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    evals: [&BabyBearExtWire; 3],
    x: &BabyBearExtWire,
) -> BabyBearExtWire {
    let one = ext_chip.from_base_const(ctx, ChildF::ONE);
    let two = ext_chip.from_base_const(ctx, ChildF::TWO);
    let inv_two = ChildF::ONE.halve();

    let x_minus_one = ext_chip.sub(ctx, *x, one);
    let x_minus_two = ext_chip.sub(ctx, *x, two);
    let x_times_x_minus_one = ext_chip.mul(ctx, *x, x_minus_one);
    let x_times_x_minus_two = ext_chip.mul(ctx, *x, x_minus_two);
    let x_minus_one_times_x_minus_two = ext_chip.mul(ctx, x_minus_one, x_minus_two);

    let l0 = ext_chip.mul_base_const(ctx, x_minus_one_times_x_minus_two, inv_two);
    let l1 = ext_chip.neg(ctx, x_times_x_minus_two);
    let l2 = ext_chip.mul_base_const(ctx, x_times_x_minus_one, inv_two);

    let term0 = ext_chip.mul(ctx, *evals[0], l0);
    let term1 = ext_chip.mul(ctx, *evals[1], l1);
    let term2 = ext_chip.mul(ctx, *evals[2], l2);
    let sum01 = ext_chip.add(ctx, term0, term1);
    ext_chip.add(ctx, sum01, term2)
}
