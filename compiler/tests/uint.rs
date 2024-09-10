use afs_compiler::{asm::AsmBuilder, conversion::CompilerOptions, util::execute_program};
use ax_sdk::utils::create_seeded_rng;
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use rand::RngCore;

#[test]
fn test_huest() {
    let num_digits = 8;

    let mut rng = create_seeded_rng();
    let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let a = BigUint::new(a_digits);
    let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let b = BigUint::new(b_digits);

    let c = (a.clone() + b.clone()) % (BigUint::from(1u32) << 256);

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let a_var = builder.eval_biguint(a);
    let b_var = builder.eval_biguint(b);
    let c_var = builder.u256_add(&a_var, &b_var);
    let c_check_var = builder.eval_biguint(c);
    builder.assert_var_array_eq(&c_var, &c_check_var);
    builder.halt();

    let program = builder.clone().compile_isa_with_options(CompilerOptions {
        compile_prints: false,
        enable_cycle_tracker: false,
        field_arithmetic_enabled: false,
        field_extension_enabled: false,
        field_less_than_enabled: false,
    });
    execute_program(program, vec![]);
}
