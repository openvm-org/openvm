use std::sync::Arc;

use halo2_base::{
    gates::{
        circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
        GateInstructions, RangeInstructions,
    },
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::openvm_stark_backend::p3_field::{
    BasedVectorSpace, PrimeCharacteristicRing, PrimeField64,
};

use super::*;
use crate::{
    config::{STATIC_VERIFIER_LOOKUP_ADVICE_COLS, STATIC_VERIFIER_NUM_ADVICE_COLS},
    Fr, RootEF, RootF,
};

fn run_mock(build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(12)
        .use_lookup_bits(11)
        .use_instance_columns(1);
    build(&mut builder);

    let params = builder.calculate_params(Some(20));
    assert_eq!(
        params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default(),
        STATIC_VERIFIER_NUM_ADVICE_COLS
    );
    assert_eq!(
        params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default(),
        STATIC_VERIFIER_LOOKUP_ADVICE_COLS
    );

    MockProver::run(12, &builder, vec![vec![]])
        .expect("mock prover should initialize")
        .assert_satisfied();
}

#[test]
fn baby_bear_base_ops_match_native_mod_arithmetic() {
    run_mock(|builder| {
        let range = builder.range_chip();
        let chip = BabyBearChip::new(Arc::new(range.clone()));
        let ctx = builder.main(0);
        let gate = range.gate();

        let cases = [
            (0u64, 0u64),
            (1, 2),
            (5, 13),
            (BABY_BEAR_MODULUS_U64 - 1, 1),
            (BABY_BEAR_MODULUS_U64 - 2, BABY_BEAR_MODULUS_U64 - 3),
        ];

        for (a_u64, b_u64) in cases {
            let a = chip.load_witness(ctx, RootF::from_u64(a_u64));
            let b = chip.load_witness(ctx, RootF::from_u64(b_u64));

            // Reduce results to canonical form before comparing, since the new
            // BabyBearChip uses lazy reduction and wire values may be unreduced.
            let sum = chip.add(ctx, a, b);
            let sum = chip.reduce(ctx, sum);
            let diff = chip.sub(ctx, a, b);
            let diff = chip.reduce(ctx, diff);
            let prod = chip.mul(ctx, a, b);
            let prod = chip.reduce(ctx, prod);
            let neg = chip.neg(ctx, a);
            let neg = chip.reduce(ctx, neg);
            let by_const = chip.mul_const(ctx, a, RootF::from_u64(11));
            let by_const = chip.reduce(ctx, by_const);

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

            gate.assert_is_const(ctx, &sum.value, &Fr::from(expected_sum));
            gate.assert_is_const(ctx, &diff.value, &Fr::from(expected_diff));
            gate.assert_is_const(ctx, &prod.value, &Fr::from(expected_prod));
            gate.assert_is_const(ctx, &neg.value, &Fr::from(expected_neg));
            gate.assert_is_const(ctx, &by_const.value, &Fr::from(expected_by_const));
        }
    });
}

#[test]
fn baby_bear_ext_mul_matches_native_binomial_extension() {
    run_mock(|builder| {
        let range = builder.range_chip();
        let base_chip = BabyBearChip::new(Arc::new(range));
        let ext_chip = BabyBearExt5Chip::new(base_chip);
        let ctx = builder.main(0);
        let gate = ext_chip.base().gate();

        let lhs_native =
            RootEF::from_basis_coefficients_fn(|i| RootF::from_u64([3, 5, 7, 11, 13][i]));
        let rhs_native =
            RootEF::from_basis_coefficients_fn(|i| RootF::from_u64([2, 4, 6, 8, 10][i]));

        let lhs = ext_chip.load_witness(ctx, lhs_native);
        let rhs = ext_chip.load_witness(ctx, rhs_native);

        let sum = ext_chip.add(ctx, lhs, rhs);
        let sum = ext_chip.reduce_max_bits(ctx, sum);
        let diff = ext_chip.sub(ctx, lhs, rhs);
        let diff = ext_chip.reduce_max_bits(ctx, diff);
        let prod = ext_chip.mul(ctx, lhs, rhs);
        let prod = ext_chip.reduce_max_bits(ctx, prod);
        let sqr = ext_chip.square(ctx, lhs);
        let sqr = ext_chip.reduce_max_bits(ctx, sqr);

        let sum_native = lhs_native + rhs_native;
        let diff_native = lhs_native - rhs_native;
        let prod_native = lhs_native * rhs_native;
        let sqr_native = lhs_native * lhs_native;

        let expected_sum = core::array::from_fn::<_, BABY_BEAR_EXT_DEGREE, _>(|i| {
            <RootEF as BasedVectorSpace<RootF>>::as_basis_coefficients_slice(&sum_native)[i]
                .as_canonical_u64()
        });
        let expected_diff = core::array::from_fn::<_, BABY_BEAR_EXT_DEGREE, _>(|i| {
            <RootEF as BasedVectorSpace<RootF>>::as_basis_coefficients_slice(&diff_native)[i]
                .as_canonical_u64()
        });
        let expected_prod = core::array::from_fn::<_, BABY_BEAR_EXT_DEGREE, _>(|i| {
            <RootEF as BasedVectorSpace<RootF>>::as_basis_coefficients_slice(&prod_native)[i]
                .as_canonical_u64()
        });
        let expected_sqr = core::array::from_fn::<_, BABY_BEAR_EXT_DEGREE, _>(|i| {
            <RootEF as BasedVectorSpace<RootF>>::as_basis_coefficients_slice(&sqr_native)[i]
                .as_canonical_u64()
        });

        for i in 0..BABY_BEAR_EXT_DEGREE {
            gate.assert_is_const(ctx, &sum.0[i].value, &Fr::from(expected_sum[i]));
            gate.assert_is_const(ctx, &diff.0[i].value, &Fr::from(expected_diff[i]));
            gate.assert_is_const(ctx, &prod.0[i].value, &Fr::from(expected_prod[i]));
            gate.assert_is_const(ctx, &sqr.0[i].value, &Fr::from(expected_sqr[i]));
        }
    });
}
