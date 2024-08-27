use afs_compiler::{
    asm::{AsmBuilder, AsmCompiler, AsmConfig},
    conversion::{convert_program, CompilerOptions},
    ir::{Builder, Ext, ExtConst, Felt, SymbolicExt, Var},
    util::execute_program,
};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField, Field};
use rand::{thread_rng, Rng};
use stark_vm::{
    cpu::trace::ExecutionError::Fail,
    vm::{config::VmConfig, VirtualMachine},
};

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

    builder.assert_felt_eq(one / F::two(), F::two().inverse());

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

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
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

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
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
    builder.assign(&x, x + x);
    builder.assert_ext_eq(x, (ef + ef).cons());

    let x: Ext<_, _> = builder.constant(ef);
    builder.assign(&x, x - x);
    builder.assert_ext_eq(x, EF::zero().cons());

    let x: Ext<_, _> = builder.constant(ef);
    builder.assign(&x, x * x);
    builder.assert_ext_eq(x, (ef * ef).cons());

    let x: Ext<_, _> = builder.constant(ef);
    builder.assign(&x, x / x);
    builder.assert_ext_eq(x, EF::one().cons());

    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_field_immediate() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let mut rng = thread_rng();

    let a = rng.gen();
    let b = rng.gen();

    let v: Felt<_> = builder.constant(a);

    builder.assert_felt_eq(v + b, a + b);
    builder.assert_felt_eq(v - b, a - b);
    builder.assert_felt_eq(v * b, a * b);
    builder.assert_felt_eq(v / b, a / b);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_ext_immediate() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let f = F::from_canonical_u32(314159265);

    let ef = EF::from_base_slice(&[
        F::from_canonical_u32(1163664312),
        F::from_canonical_u32(1251518712),
        F::from_canonical_u32(1133200680),
        F::from_canonical_u32(1689596134),
    ]);

    let ext: Ext<_, _> = builder.constant(ef);

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext + ef);
    builder.assert_ext_eq(x, (ef + ef).cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ef.cons() + ext);
    builder.assert_ext_eq(x, (ef + ef).cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext + f);
    builder.assert_ext_eq(x, (ef + f).cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext - ef);
    builder.assert_ext_eq(x, EF::zero().cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ef.cons() - ext);
    builder.assert_ext_eq(x, EF::zero().cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext - f);
    builder.assert_ext_eq(x, (ef - f).cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext * ef);
    builder.assert_ext_eq(x, (ef * ef).cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ef.cons() * ext);
    builder.assert_ext_eq(x, (ef * ef).cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext * f);
    builder.assert_ext_eq(x, (ef * f).cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext / ef);
    builder.assert_ext_eq(x, EF::one().cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ef.cons() / ext);
    builder.assert_ext_eq(x, EF::one().cons());

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext / f);
    builder.assert_ext_eq(x, (ef / f.into()).cons());

    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);

    let program = builder.compile_isa_with_options(CompilerOptions {
        compile_prints: false,
        enable_cycle_tracker: false,
        field_arithmetic_enabled: true,
        field_extension_enabled: true,
        field_less_than_enabled: false,
    });
    execute_program(program, vec![]);
}

#[test]
fn test_ext_felt_arithmetic() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let f = F::from_canonical_u32(314159265);

    let ef = EF::from_base_slice(&[
        F::from_canonical_u32(1163664312),
        F::from_canonical_u32(1251518712),
        F::from_canonical_u32(1133200680),
        F::from_canonical_u32(1689596134),
    ]);

    let felt: Felt<_> = builder.constant(f);
    let ext: Ext<_, _> = builder.constant(ef);

    let x: Ext<_, _> = builder.uninit();
    builder.assign(&x, ext + felt);
    builder.assert_ext_eq(x, (ef + f).cons());

    builder.assign(&x, ext + f);
    builder.assert_ext_eq(x, (ef + f).cons());

    builder.assign(&x, ext - felt);
    builder.assert_ext_eq(x, (ef - f).cons());

    builder.assign(&x, ext - f);
    builder.assert_ext_eq(x, (ef - f).cons());

    builder.assign(&x, ext * felt);
    builder.assert_ext_eq(x, (ef * f).cons());

    builder.assign(&x, ext * f);
    builder.assert_ext_eq(x, (ef * f).cons());

    builder.assign(&x, ext / felt);
    builder.assert_ext_eq(x, (ef / EF::from_base(f)).cons());

    builder.assign(&x, ext / f);
    builder.assert_ext_eq(x, (ef / EF::from_base(f)).cons());

    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);

    let program = builder.compile_isa_with_options(CompilerOptions {
        compile_prints: false,
        enable_cycle_tracker: false,
        field_arithmetic_enabled: true,
        field_extension_enabled: true,
        field_less_than_enabled: false,
    });
    execute_program(program, vec![]);
}

#[test]
fn test_felt_equality() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut rng = thread_rng();
    let f = rng.gen::<F>();

    let mut builder = AsmBuilder::<F, EF>::default();

    let a: Felt<_> = builder.constant(f);
    builder.assert_felt_eq(a, f);
    builder.assert_felt_eq(f, a);
    builder.assert_felt_eq(a, a);
    builder.assert_ext_eq(a, a);

    builder.assert_felt_ne(a, a + F::one());
    builder.assert_felt_ne(a, f + F::one());
    builder.assert_felt_ne(a + F::one(), a);
    builder.assert_felt_ne(f + F::one(), a);

    builder.halt();

    let mut compiler = AsmCompiler::new(WORD_SIZE);
    compiler.build(builder.operations);
    let asm_code = compiler.code();
    println!("{}", asm_code);

    let program = convert_program::<F, EF>(asm_code, CompilerOptions::default());
    execute_program(program, vec![]);
}

#[test]
fn test_felt_equality_negative() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut rng = thread_rng();
    let f = rng.gen::<F>();

    let mut builder = AsmBuilder::<F, EF>::default();
    let a: Felt<_> = builder.constant(f);
    builder.assert_felt_ne(a, a);
    builder.halt();
    assert_failed_assertion(builder);

    let mut builder = AsmBuilder::<F, EF>::default();
    let a: Felt<_> = builder.constant(f);
    builder.assert_felt_eq(a, a + F::one());
    builder.halt();

    assert_failed_assertion(builder);
}

#[test]
fn test_ext_equality() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut rng = thread_rng();
    let a_ext = rng.gen::<EF>();

    let mut builder = AsmBuilder::<F, EF>::default();

    let a: Ext<_, _> = builder.constant(a_ext);
    builder.assert_ext_eq(a, a);
    builder.assert_ext_eq(a, a_ext.cons());
    builder.assert_ext_eq(a_ext.cons(), a);

    builder.assert_ext_ne(a, a + EF::one());
    builder.assert_ext_ne(a + EF::one(), a);
    builder.assert_ext_ne(a, (a_ext + EF::one()).cons());
    builder.assert_ext_ne((a_ext + EF::one()).cons(), a);

    for i in 0..4 {
        let mut base = a_ext.as_base_slice().to_vec();
        base[i] = rng.gen::<F>();
        let b_ext = EF::from_base_slice(&base);
        builder.assert_ext_ne(a, b_ext.cons());
        builder.assert_ext_ne(b_ext.cons(), a);
    }

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_ext_equality_negative() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut rng = thread_rng();
    let a_ext = rng.gen::<EF>();

    let mut builder = AsmBuilder::<F, EF>::default();
    let a: Ext<_, _> = builder.constant(a_ext);
    builder.assert_ext_ne(a, a);
    builder.halt();
    assert_failed_assertion(builder);

    let mut builder = AsmBuilder::<F, EF>::default();
    let a: Ext<_, _> = builder.constant(a_ext);
    builder.assert_ext_eq(a, a + EF::one());
    builder.halt();
    assert_failed_assertion(builder);
}

fn assert_failed_assertion(
    builder: Builder<AsmConfig<BabyBear, BinomialExtensionField<BabyBear, 4>>>,
) {
    let program = builder.compile_isa();
    let vm = VirtualMachine::new(VmConfig::default(), program, vec![]);
    let result = vm.execute();
    assert!(matches!(result, Err(Fail(_))));
}
