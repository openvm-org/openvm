use std::iter;

use ax_sdk::utils::create_seeded_rng;
use axvm_circuit::{
    arch::ExecutorName,
    system::{
        program::util::{execute_program, execute_program_with_config},
        vm::config::VmConfig,
    },
};
use axvm_native_compiler::{asm::AsmBuilder, conversion::CompilerOptions, ir::Var};
use num_bigint_dig::BigUint;
use num_traits::Zero;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use rand::{Rng, RngCore};
#[test]
fn test_compiler_256_add_sub() {
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
            builder.add_256(&a_var, &b_var)
        } else {
            builder.sub_256(&a_var, &b_var)
        };
        let c_check_var = builder.eval_biguint(c);
        builder.assert_var_array_eq(&c_var, &c_check_var);
    }
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_256_mul() {
    let num_digits = 8;
    let num_ops = 10;
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

        let c = (a.clone() * b.clone()) % u256_modulus.clone();
        let c_var = builder.mul_256(&a_var, &b_var);

        let c_check_var = builder.eval_biguint(c);
        builder.assert_var_array_eq(&c_var, &c_check_var);
    }
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program_with_config(
        VmConfig {
            num_public_values: 4,
            max_segment_len: (1 << 25) - 100,
            ..Default::default()
        }
        .add_executor(ExecutorName::Phantom)
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::FieldArithmetic)
        .add_executor(ExecutorName::U256Multiplication),
        program,
        vec![],
    );
}

#[test]
fn test_compiler_256_sltu_eq() {
    let num_digits = 8;
    let num_ops = 15;
    let mut rng = create_seeded_rng();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    for _ in 0..num_ops {
        let lt_flag = rng.gen_bool(0.5);

        let a_digits: Vec<u32> = (0..num_digits).map(|_| rng.next_u32()).collect();
        let a = BigUint::new(a_digits.clone());
        let b_digits = if lt_flag || rng.gen_bool(0.5) {
            (0..num_digits).map(|_| rng.next_u32()).collect()
        } else {
            a_digits.clone()
        };
        let b = BigUint::new(b_digits);

        let a_var = builder.eval_biguint(a.clone());
        let b_var = builder.eval_biguint(b.clone());

        let c = if lt_flag {
            a.clone() < b.clone()
        } else {
            a.clone() == b.clone()
        };

        let c_var = if lt_flag {
            builder.sltu_256(&a_var, &b_var)
        } else {
            builder.eq_256(&a_var, &b_var)
        };

        let c_check_var: Var<_> = builder.eval(F::from_bool(c));
        builder.assert_var_eq(c_var, c_check_var);
    }
    builder.halt();

    let program = builder.clone().compile_isa_with_options(CompilerOptions {
        word_size: 32,
        ..Default::default()
    });
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_256_slt_eq() {
    let num_digits = 8;
    let num_ops = 15;
    let mut rng = create_seeded_rng();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();
    let msb_mask: u32 = 1 << 31;

    for _ in 0..num_ops {
        let a_digits: Vec<u32> = (0..num_digits).map(|_| rng.next_u32()).collect();
        let a = BigUint::new(a_digits.clone());
        let b_digits: Vec<u32> = (0..num_digits).map(|_| rng.next_u32()).collect();
        let b = BigUint::new(b_digits.clone());

        let a_var = builder.eval_biguint(a.clone());
        let b_var = builder.eval_biguint(b.clone());

        let same_sign =
            (a_digits[num_digits - 1] & msb_mask) == (b_digits[num_digits - 1] & msb_mask);

        let c = if same_sign {
            a.clone() < b.clone()
        } else {
            a_digits[num_digits - 1] & msb_mask == msb_mask
        };

        let c_var = builder.slt_256(&a_var, &b_var);
        let c_check_var: Var<_> = builder.eval(F::from_bool(c));
        builder.assert_var_eq(c_var, c_check_var);
    }
    builder.halt();

    let program = builder.clone().compile_isa_with_options(CompilerOptions {
        word_size: 32,
        ..Default::default()
    });
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_256_xor_and_or() {
    let num_digits = 8;
    let num_ops = 20;
    let mut rng = create_seeded_rng();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    for _ in 0..num_ops {
        let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
        let a = BigUint::new(a_digits);
        let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
        let b = BigUint::new(b_digits);

        let a_var = builder.eval_biguint(a.clone());
        let b_var = builder.eval_biguint(b.clone());

        // xor = 0, and = 1, or = 2
        let flag: u8 = rng.gen_range(0..=2);

        let c = if flag == 0 {
            a.clone() ^ b.clone()
        } else if flag == 1 {
            a.clone() & b.clone()
        } else {
            a.clone() | b.clone()
        };

        let c_var = if flag == 0 {
            builder.xor_256(&a_var, &b_var)
        } else if flag == 1 {
            builder.and_256(&a_var, &b_var)
        } else {
            builder.or_256(&a_var, &b_var)
        };
        let c_check_var = builder.eval_biguint(c);
        builder.assert_var_array_eq(&c_var, &c_check_var);
    }
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_256_sll_srl() {
    let num_digits = 8;
    let num_ops = 15;
    let mut rng = create_seeded_rng();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    for _ in 0..num_ops {
        let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect::<Vec<_>>();
        let a = BigUint::new(a_digits.clone());

        let b_shift = rng.gen_range(0..=64);
        let b_digits = iter::once(b_shift as u32)
            .chain(iter::repeat(0u32))
            .take(num_digits)
            .collect::<Vec<_>>();
        let b = BigUint::new(b_digits);

        let a_var = builder.eval_biguint(a.clone());
        let b_var = builder.eval_biguint(b.clone());

        // sll = 0, srl = 1
        let sll_flag = rng.gen_bool(0.5);

        let c = if sll_flag {
            a.clone() << b_shift
        } else {
            a.clone() >> b_shift
        };

        let c_var = if sll_flag {
            builder.sll_256(&a_var, &b_var)
        } else {
            builder.srl_256(&a_var, &b_var)
        };
        let c_check_var = builder.eval_biguint(c);
        builder.assert_var_array_eq(&c_var, &c_check_var);
    }
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program_with_config(
        VmConfig {
            num_public_values: 4,
            max_segment_len: (1 << 25) - 100,
            ..Default::default()
        }
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::Shift256)
        .add_executor(ExecutorName::FieldArithmetic),
        program,
        vec![],
    );
}

#[test]
fn test_compiler_256_sra() {
    let num_digits = 8;
    let num_ops = 10;
    let mut rng = create_seeded_rng();

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();
    let msb_mask: u32 = 1 << 31;

    for _ in 0..num_ops {
        let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect::<Vec<_>>();
        let a_sign = a_digits[num_digits - 1] & msb_mask == msb_mask;
        let a = BigUint::new(a_digits.clone());

        let b_shift = rng.gen_range(0..=256);
        let b_digits = iter::once(b_shift as u32)
            .chain(iter::repeat(0u32))
            .take(num_digits)
            .collect::<Vec<_>>();
        let b = BigUint::new(b_digits);

        let a_var = builder.eval_biguint(a.clone());
        let b_var = builder.eval_biguint(b.clone());

        let ones = iter::repeat(0)
            .take((256 - b_shift) / 32)
            .chain(iter::once(u32::MAX << (32 - (b_shift % 32))))
            .chain(iter::repeat(u32::MAX))
            .take(num_digits)
            .collect::<Vec<_>>();

        let c = (a.clone() >> b_shift)
            + if a_sign {
                BigUint::new(ones)
            } else {
                BigUint::zero()
            };

        let c_var = builder.sra_256(&a_var, &b_var);
        let c_check_var = builder.eval_biguint(c);
        builder.assert_var_array_eq(&c_var, &c_check_var);
    }
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program_with_config(
        VmConfig {
            num_public_values: 4,
            max_segment_len: (1 << 25) - 100,
            ..Default::default()
        }
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::Shift256)
        .add_executor(ExecutorName::FieldArithmetic),
        program,
        vec![],
    );
}
