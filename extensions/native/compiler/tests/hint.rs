use ax_stark_backend::p3_field::{extension::BinomialExtensionField, AbstractField, Field};
use ax_stark_sdk::p3_baby_bear::BabyBear;
use axvm_native_circuit::execute_program;
use axvm_native_compiler::{
    asm::AsmBuilder,
    ir::{Felt, RVar, Var},
};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_hint_bits_felt() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let felt: Felt<_> = builder.constant(F::from_canonical_u32(5));
    let bits = builder.num2bits_f(felt, F::bits() as u32);

    let x = builder.get(&bits, 0);
    builder.assert_var_eq(x, F::ONE);
    let x = builder.get(&bits, 1);
    builder.assert_var_eq(x, F::ZERO);
    let x = builder.get(&bits, 2);
    builder.assert_var_eq(x, F::ONE);

    for i in 3..31 {
        let x = builder.get(&bits, i);
        builder.assert_var_eq(x, F::ZERO);
    }

    builder.halt();

    let program = builder.compile_isa();
    println!("{}", program);
    execute_program(program, vec![]);
}

#[test]
fn test_hint_bits_var() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let var: Var<_> = builder.constant(F::from_canonical_u32(5));
    let bits = builder.num2bits_v(var, F::bits() as u32);

    let x = builder.get(&bits, RVar::zero());
    builder.assert_var_eq(x, F::ONE);
    let x = builder.get(&bits, RVar::one());
    builder.assert_var_eq(x, F::ZERO);
    let x = builder.get(&bits, 2);
    builder.assert_var_eq(x, F::ONE);

    for i in 3..31 {
        let x = builder.get(&bits, i);
        builder.assert_var_eq(x, F::ZERO);
    }

    builder.halt();

    let program = builder.compile_isa();
    println!("{}", program);
    execute_program(program, vec![]);
}