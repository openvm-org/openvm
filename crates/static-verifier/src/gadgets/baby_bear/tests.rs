use halo2_base::{
    gates::circuit::{CircuitBuilderStage, builder::BaseCircuitBuilder},
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::{
        BasedVectorSpace, PrimeCharacteristicRing, PrimeField64,
        extension::BinomialExtensionField,
    },
    p3_baby_bear::BabyBear,
};

use crate::config::{
    STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
};

use super::*;

fn run_mock(build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(12)
        .use_lookup_bits(8)
        .use_instance_columns(1);
    build(&mut builder);

    let params = builder.calculate_params(Some(20));
    assert_eq!(
        params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default(),
        STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0
    );
    assert_eq!(
        params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default(),
        STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0
    );

    MockProver::run(12, &builder, vec![vec![]])
        .expect("mock prover should initialize")
        .assert_satisfied();
}

fn ef_from_u64(coeffs: [u64; BABY_BEAR_EXT_DEGREE]) -> BinomialExtensionField<BabyBear, 4> {
    BinomialExtensionField::from_basis_coefficients_fn(|i| BabyBear::from_u64(coeffs[i]))
}

#[test]
fn baby_bear_base_ops_match_native_mod_arithmetic() {
    run_mock(|builder| {
        let gadgets = BabyBearArithmeticGadgets;
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let gate = range.gate();

        let cases = [
            (0, 0),
            (1, 2),
            (5, 13),
            (BABY_BEAR_MODULUS_U64 - 1, 1),
            (BABY_BEAR_MODULUS_U64 - 2, BABY_BEAR_MODULUS_U64 - 3),
        ];

        for (a_u64, b_u64) in cases {
            let a = gadgets.load_witness(ctx, &range, a_u64);
            let b = gadgets.load_witness(ctx, &range, b_u64);

            let sum = gadgets.add(ctx, &range, &a, &b);
            let diff = gadgets.sub(ctx, &range, &a, &b);
            let prod = gadgets.mul(ctx, &range, &a, &b);
            let neg = gadgets.neg(ctx, &range, &a);
            let by_const = gadgets.mul_const(ctx, &range, &a, 11);

            let expected_sum = (a_u64 + b_u64) % BABY_BEAR_MODULUS_U64;
            let expected_diff = (a_u64 + BABY_BEAR_MODULUS_U64 - b_u64) % BABY_BEAR_MODULUS_U64;
            let expected_prod =
                ((a_u64 as u128 * b_u64 as u128) % BABY_BEAR_MODULUS_U64 as u128) as u64;
            let expected_neg = if a_u64 == 0 {
                0
            } else {
                BABY_BEAR_MODULUS_U64 - a_u64
            };
            let expected_by_const =
                ((a_u64 as u128 * 11u128) % BABY_BEAR_MODULUS_U64 as u128) as u64;

            gate.assert_is_const(ctx, &sum.cell, &Fr::from(expected_sum));
            gate.assert_is_const(ctx, &diff.cell, &Fr::from(expected_diff));
            gate.assert_is_const(ctx, &prod.cell, &Fr::from(expected_prod));
            gate.assert_is_const(ctx, &neg.cell, &Fr::from(expected_neg));
            gate.assert_is_const(ctx, &by_const.cell, &Fr::from(expected_by_const));
        }
    });
}

#[test]
fn baby_bear_ext_mul_matches_native_binomial_extension() {
    run_mock(|builder| {
        let gadgets = BabyBearArithmeticGadgets;
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let gate = range.gate();

        let lhs_coeffs = [3, 5, 7, 11];
        let rhs_coeffs = [2, 4, 6, 8];
        let lhs = gadgets.load_ext_witness(ctx, &range, lhs_coeffs);
        let rhs = gadgets.load_ext_witness(ctx, &range, rhs_coeffs);

        let sum = gadgets.ext_add(ctx, &range, &lhs, &rhs);
        let diff = gadgets.ext_sub(ctx, &range, &lhs, &rhs);
        let prod = gadgets.ext_mul(ctx, &range, &lhs, &rhs);
        let sqr = gadgets.ext_square(ctx, &range, &lhs);

        let lhs_native = ef_from_u64(lhs_coeffs);
        let rhs_native = ef_from_u64(rhs_coeffs);
        let sum_native = lhs_native + rhs_native;
        let diff_native = lhs_native - rhs_native;
        let prod_native = lhs_native * rhs_native;
        let sqr_native = lhs_native * lhs_native;

        let expected_sum = core::array::from_fn::<_, 4, _>(|i| {
            <BinomialExtensionField<BabyBear, 4> as BasedVectorSpace<BabyBear>>::as_basis_coefficients_slice(&sum_native)
                [i]
                .as_canonical_u64()
        });
        let expected_diff = core::array::from_fn::<_, 4, _>(|i| {
            <BinomialExtensionField<BabyBear, 4> as BasedVectorSpace<BabyBear>>::as_basis_coefficients_slice(&diff_native)
                [i]
                .as_canonical_u64()
        });
        let expected_prod = core::array::from_fn::<_, 4, _>(|i| {
            <BinomialExtensionField<BabyBear, 4> as BasedVectorSpace<BabyBear>>::as_basis_coefficients_slice(&prod_native)
                [i]
                .as_canonical_u64()
        });
        let expected_sqr = core::array::from_fn::<_, 4, _>(|i| {
            <BinomialExtensionField<BabyBear, 4> as BasedVectorSpace<BabyBear>>::as_basis_coefficients_slice(&sqr_native)
                [i]
                .as_canonical_u64()
        });

        for i in 0..BABY_BEAR_EXT_DEGREE {
            gate.assert_is_const(ctx, &sum.coeffs[i].cell, &Fr::from(expected_sum[i]));
            gate.assert_is_const(ctx, &diff.coeffs[i].cell, &Fr::from(expected_diff[i]));
            gate.assert_is_const(ctx, &prod.coeffs[i].cell, &Fr::from(expected_prod[i]));
            gate.assert_is_const(ctx, &sqr.coeffs[i].cell, &Fr::from(expected_sqr[i]));
        }
    });
}
