use halo2_base::{Context, gates::range::RangeChip};
use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::F as NativeF;
use openvm_stark_sdk::openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{
    circuit::Fr,
    gadgets::baby_bear::{BabyBearArithmeticGadgets, BabyBearExtVar},
    stages::batch_constraints::{ext_from_base_const, ext_mul_base_const},
};

pub(crate) fn column_openings_by_rot_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    openings: &[BabyBearExtVar],
    need_rot: bool,
) -> Vec<(BabyBearExtVar, BabyBearExtVar)> {
    if need_rot {
        assert!(
            openings.len() % 2 == 0,
            "rotated opening vector must be even",
        );
        openings
            .chunks_exact(2)
            .map(|chunk| (chunk[0].clone(), chunk[1].clone()))
            .collect::<Vec<_>>()
    } else {
        let zero = baby_bear.ext_zero(ctx, range);
        openings
            .iter()
            .map(|opening| (opening.clone(), zero.clone()))
            .collect::<Vec<_>>()
    }
}

pub(crate) fn horner_eval_ext_poly_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    coeffs: &[BabyBearExtVar],
    x: &BabyBearExtVar,
) -> BabyBearExtVar {
    let mut acc = baby_bear.ext_zero(ctx, range);
    for coeff in coeffs.iter().rev() {
        acc = baby_bear.ext_mul(ctx, range, &acc, x);
        acc = baby_bear.ext_add(ctx, range, &acc, coeff);
    }
    acc
}

pub(crate) fn interpolate_quadratic_at_012_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    evals: [&BabyBearExtVar; 3],
    x: &BabyBearExtVar,
) -> BabyBearExtVar {
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let two = ext_from_base_const(ctx, range, baby_bear, 2);
    let inv_two = NativeF::ONE.halve().as_canonical_u64();

    let x_minus_one = baby_bear.ext_sub(ctx, range, x, &one);
    let x_minus_two = baby_bear.ext_sub(ctx, range, x, &two);
    let x_times_x_minus_one = baby_bear.ext_mul(ctx, range, x, &x_minus_one);
    let x_times_x_minus_two = baby_bear.ext_mul(ctx, range, x, &x_minus_two);
    let x_minus_one_times_x_minus_two = baby_bear.ext_mul(ctx, range, &x_minus_one, &x_minus_two);

    let l0 = ext_mul_base_const(
        ctx,
        range,
        baby_bear,
        &x_minus_one_times_x_minus_two,
        inv_two,
    );
    let zero = baby_bear.ext_zero(ctx, range);
    let l1 = baby_bear.ext_sub(ctx, range, &zero, &x_times_x_minus_two);
    let l2 = ext_mul_base_const(ctx, range, baby_bear, &x_times_x_minus_one, inv_two);

    let term0 = baby_bear.ext_mul(ctx, range, evals[0], &l0);
    let term1 = baby_bear.ext_mul(ctx, range, evals[1], &l1);
    let term2 = baby_bear.ext_mul(ctx, range, evals[2], &l2);
    let sum01 = baby_bear.ext_add(ctx, range, &term0, &term1);
    baby_bear.ext_add(ctx, range, &sum01, &term2)
}
