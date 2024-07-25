use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, Field};
use rand::{thread_rng, Rng};

use afs_compiler::asm::AsmBuilder;
use afs_compiler::conversion::CompilerOptions;
use afs_compiler::ir::{Ext, Felt, SymbolicExt};
use afs_compiler::ir::{ExtConst, Var};
use afs_compiler::util::execute_program;

#[allow(dead_code)]
const WORD_SIZE: usize = 1;

#[test]
fn test_compiler_arithmetic() {
    let num_tests = 3;
    let mut rng = thread_rng();
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let zero: Felt<_> = builder.eval(F::zero());
    let one: Felt<_> = builder.eval(F::one());

    builder.assert_felt_eq(zero * one, F::zero());
    builder.assert_felt_eq(one * one, F::one());
    builder.assert_felt_eq(one + one, F::two());

    let zero_ext: Ext<_, _> = builder.eval(EF::zero().cons());
    let one_ext: Ext<_, _> = builder.eval(EF::one().cons());
    let two_ext: Ext<_, _> = builder.eval(EF::two().cons());

    // Check Val() vs Const() equality
    builder.assert_ext_eq(zero_ext, EF::zero().cons());
    builder.assert_ext_eq(one_ext, EF::one().cons());
    builder.assert_ext_eq(two_ext, EF::two().cons());

    // Check Val() vs Const() inequality
    builder.assert_ext_ne(one_ext, EF::two().cons());

    builder.assert_ext_eq(zero_ext * one_ext, EF::zero().cons());
    builder.assert_ext_eq(one_ext * one_ext, EF::one().cons());
    builder.assert_ext_eq(one_ext + one_ext, EF::two().cons());
    builder.assert_ext_eq(one_ext - one_ext, EF::zero().cons());

    builder.assert_ext_eq(two_ext / one_ext, (EF::two() / EF::one()).cons());

    for _ in 0..num_tests {
        let a_var_val = rng.gen::<F>();
        let b_var_val = rng.gen::<F>();
        let a_var: Var<_> = builder.eval(a_var_val);
        let b_var: Var<_> = builder.eval(b_var_val);
        builder.assert_var_eq(a_var + b_var, a_var_val + b_var_val);
        builder.assert_var_eq(a_var * b_var, a_var_val * b_var_val);
        builder.assert_var_eq(a_var - b_var, a_var_val - b_var_val);
        builder.assert_var_eq(-a_var, -a_var_val);

        let a_felt_val = rng.gen::<F>();
        let b_felt_val = rng.gen::<F>();
        let a: Felt<_> = builder.eval(a_felt_val);
        let b: Felt<_> = builder.eval(b_felt_val);
        builder.assert_felt_eq(a + b, a_felt_val + b_felt_val);
        builder.assert_felt_eq(a + b, a + b_felt_val);
        builder.assert_felt_eq(a * b, a_felt_val * b_felt_val);
        builder.assert_felt_eq(a - b, a_felt_val - b_felt_val);
        builder.assert_felt_eq(a / b, a_felt_val / b_felt_val);
        builder.assert_felt_eq(-a, -a_felt_val);

        let a_ext_val = rng.gen::<EF>();
        let b_ext_val = rng.gen::<EF>();

        let a_ext: Ext<_, _> = builder.eval(a_ext_val.cons());
        let b_ext: Ext<_, _> = builder.eval(b_ext_val.cons());
        builder.assert_ext_eq(a_ext + b_ext, (a_ext_val + b_ext_val).cons());
        builder.assert_ext_eq(
            -a_ext / b_ext + (a_ext * b_ext) * (a_ext * b_ext),
            (-a_ext_val / b_ext_val + (a_ext_val * b_ext_val) * (a_ext_val * b_ext_val)).cons(),
        );
        let mut a_expr = SymbolicExt::from(a_ext);
        let mut a_val = a_ext_val;
        for _ in 0..10 {
            a_expr += b_ext * a_val + EF::one();
            a_val += b_ext_val * a_val + EF::one();
            builder.assert_ext_eq(a_expr.clone(), a_val.cons())
        }
        builder.assert_ext_eq(a_ext * b_ext, (a_ext_val * b_ext_val).cons());
        builder.assert_ext_eq(a_ext - b_ext, (a_ext_val - b_ext_val).cons());
        builder.assert_ext_eq(a_ext / b_ext, (a_ext_val / b_ext_val).cons());
        builder.assert_ext_eq(-a_ext, (-a_ext_val).cons());
    }

    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE, _>(program, vec![]);

    let program = builder.compile_isa_with_options::<WORD_SIZE>(CompilerOptions {
        field_extension_enabled: false,
        ..Default::default()
    });
    execute_program::<WORD_SIZE, _>(program, vec![]);
}

#[test]
fn test_compiler_arithmetic_2() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let ef = EF::from_base_slice(&[
        F::from_canonical_u32(1163664312),
        F::from_canonical_u32(1251518712),
        F::from_canonical_u32(1133200680),
        F::from_canonical_u32(1689596134),
    ]);

    let x: Ext<_, _> = builder.constant(ef);
    let xinv: Ext<_, _> = builder.constant(ef.inverse());
    builder.assert_ext_eq(x.inverse(), xinv);

    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE, _>(program, vec![]);

    let program = builder.compile_isa_with_options::<WORD_SIZE>(CompilerOptions {
        field_extension_enabled: false,
        ..Default::default()
    });
    execute_program::<WORD_SIZE, _>(program, vec![]);
}

#[test]
fn test_in_place_arithmetic() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let ef = EF::from_base_slice(&[
        F::from_canonical_u32(1163664312),
        F::from_canonical_u32(1251518712),
        F::from_canonical_u32(1133200680),
        F::from_canonical_u32(1689596134),
    ]);

    let x: Ext<_, _> = builder.constant(ef);
    builder.assign(x, x + x);
    builder.assert_ext_eq(x, (ef + ef).cons());

    let x: Ext<_, _> = builder.constant(ef);
    builder.assign(x, x - x);
    builder.assert_ext_eq(x, EF::zero().cons());

    let x: Ext<_, _> = builder.constant(ef);
    builder.assign(x, x * x);
    builder.assert_ext_eq(x, (ef * ef).cons());

    let x: Ext<_, _> = builder.constant(ef);
    builder.assign(x, x / x);
    builder.assert_ext_eq(x, EF::one().cons());

    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE, _>(program, vec![]);

    let program = builder.compile_isa_with_options::<WORD_SIZE>(CompilerOptions {
        field_extension_enabled: false,
        ..Default::default()
    });
    execute_program::<WORD_SIZE, _>(program, vec![]);
}

#[test]
fn test_ext_immediate() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::default();

    let ef = EF::from_base_slice(&[
        F::from_canonical_u32(1163664312),
        F::from_canonical_u32(1251518712),
        F::from_canonical_u32(1133200680),
        F::from_canonical_u32(1689596134),
    ]);

    let ext: Ext<_, _> = builder.constant(ef);

    let x: Ext<_, _> = builder.uninit();
    builder.assign(x, ext + ef);
    builder.assert_ext_eq(x, (ef + ef).cons());

    builder.assign(x, ext - ef);
    builder.assert_ext_eq(x, EF::zero().cons());

    builder.assign(x, ext * ef);
    builder.assert_ext_eq(x, (ef * ef).cons());

    builder.assign(x, ext / ef);
    builder.assert_ext_eq(x, EF::one().cons());

    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE, _>(program, vec![]);
}
