use afs_compiler::{asm::AsmBuilder, util::execute_program};
use ax_sdk::utils::create_seeded_rng;
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use rand::{Rng, RngCore};

#[test]
fn test_compiler_u256_add_sub() {
    let num_digits = 8;
    let num_ops = 15;
    let mut rng = create_seeded_rng();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();
    let u256_modulus = BigUint::from(1u32) << 256;

    for _ in 0..num_ops {
        let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
        let a = BigUint::new(a_digits);
        let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
        let b = BigUint::new(b_digits);

        let a_var = builder.eval_biguint(a.clone());
        let b_var = builder.eval_biguint(b.clone());

        let add_flag = rng.gen_bool(0.5);

        let c = if add_flag {
            (a.clone() + b.clone()) % u256_modulus.clone()
        } else {
            (a.clone() + (u256_modulus.clone() - b.clone())) % u256_modulus.clone()
        };

        let c_var = if add_flag {
            builder.u256_add(&a_var, &b_var)
        } else {
            builder.u256_sub(&a_var, &b_var)
        };
        let c_check_var = builder.eval_biguint(c);
        builder.assert_var_array_eq(&c_var, &c_check_var);
    }
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}
