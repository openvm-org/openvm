use afs_compiler::{asm::AsmBuilder, ir::Var, util::execute_program};
use ax_sdk::utils::create_seeded_rng;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, One, Zero};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use rand::RngCore;

fn secp256k1_coord_prime() -> BigUint {
    let mut result = BigUint::one() << 256;
    for power in [32, 9, 8, 7, 6, 4, 0] {
        result -= BigUint::one() << power;
    }
    result
}

#[test]
fn test_compiler_modular_arithmetic_1() {
    let a = BigUint::from_isize(31).unwrap();
    let b = BigUint::from_isize(115).unwrap();

    let r = BigUint::from_isize(31 * 115).unwrap();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let a_var = builder.eval_biguint(a);
    let b_var = builder.eval_biguint(b);
    let r_var = builder.secp256k1_coord_mul(&a_var, &b_var);
    let r_check_var = builder.eval_biguint(r);
    builder.assert_secp256k1_coord_eq(&r_var, &r_check_var);
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_modular_arithmetic_2() {
    let num_digits = 8;

    let mut rng = create_seeded_rng();
    let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let a = BigUint::new(a_digits);
    let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let b = BigUint::new(b_digits);
    // if these are not true then trace is not guaranteed to be verifiable
    assert!(a < secp256k1_coord_prime());
    assert!(b < secp256k1_coord_prime());

    let r = (a.clone() * b.clone()) % secp256k1_coord_prime();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let a_var = builder.eval_biguint(a);
    let b_var = builder.eval_biguint(b);
    let r_var = builder.secp256k1_coord_mul(&a_var, &b_var);
    let r_check_var = builder.eval_biguint(r);
    builder.assert_secp256k1_coord_eq(&r_var, &r_check_var);
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_modular_arithmetic_conditional() {
    let a = BigUint::from_isize(23).unwrap();
    let b = BigUint::from_isize(41).unwrap();

    let r = BigUint::from_isize(23 * 41).unwrap();
    let s = BigUint::from_isize(1000).unwrap();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let a_var = builder.eval_biguint(a);
    let b_var = builder.eval_biguint(b);
    let product_var = builder.secp256k1_coord_mul(&a_var, &b_var);
    let r_var = builder.eval_biguint(r);
    let s_var = builder.eval_biguint(s);

    let should_be_1: Var<F> = builder.uninit();
    let should_be_2: Var<F> = builder.uninit();

    builder
        .if_secp256k1_coord_eq(&product_var, &r_var)
        .then_or_else(
            |builder| builder.assign(&should_be_1, F::one()),
            |builder| builder.assign(&should_be_1, F::two()),
        );
    builder
        .if_secp256k1_coord_eq(&product_var, &s_var)
        .then_or_else(
            |builder| builder.assign(&should_be_2, F::one()),
            |builder| builder.assign(&should_be_2, F::two()),
        );

    builder.assert_var_eq(should_be_1, F::one());
    builder.assert_var_eq(should_be_2, F::two());

    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
#[should_panic]
fn test_compiler_modular_arithmetic_negative() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let one = builder.eval_biguint(BigUint::one());
    let one_times_one = builder.secp256k1_coord_mul(&one, &one);
    let zero = builder.eval_biguint(BigUint::zero());
    builder.assert_secp256k1_coord_eq(&one_times_one, &zero);
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_modular_scalar_arithmetic_conditional() {
    let a = BigUint::from_isize(23).unwrap();
    let b = BigUint::from_isize(41).unwrap();

    let r = BigUint::from_isize(23 * 41).unwrap();
    let s = BigUint::from_isize(1000).unwrap();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let a_var = builder.eval_biguint(a);
    let b_var = builder.eval_biguint(b);
    let product_var = builder.secp256k1_scalar_mul(&a_var, &b_var);
    let r_var = builder.eval_biguint(r);
    let s_var = builder.eval_biguint(s);

    let should_be_1: Var<F> = builder.uninit();
    let should_be_2: Var<F> = builder.uninit();

    builder
        .if_secp256k1_scalar_eq(&product_var, &r_var)
        .then_or_else(
            |builder| builder.assign(&should_be_1, F::one()),
            |builder| builder.assign(&should_be_1, F::two()),
        );
    builder
        .if_secp256k1_scalar_eq(&product_var, &s_var)
        .then_or_else(
            |builder| builder.assign(&should_be_2, F::one()),
            |builder| builder.assign(&should_be_2, F::two()),
        );

    builder.assert_var_eq(should_be_1, F::one());
    builder.assert_var_eq(should_be_2, F::two());

    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
#[should_panic]
fn test_compiler_modular_scalar_arithmetic_negative() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let one = builder.eval_biguint(BigUint::one());
    let one_times_one = builder.secp256k1_scalar_mul(&one, &one);
    let zero = builder.eval_biguint(BigUint::zero());
    builder.assert_secp256k1_scalar_eq(&one_times_one, &zero);
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}
