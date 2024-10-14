use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, RVar, Var},
};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField, Field};
use stark_vm::system::program::util::execute_program;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_hint_bits_felt() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let felt: Felt<_> = builder.constant(F::from_canonical_u32(5));
    let bits = builder.num2bits_f(felt, F::bits() as u32);

    let x = builder.get(&bits, 0);
    builder.assert_var_eq(x, F::one());
    let x = builder.get(&bits, 1);
    builder.assert_var_eq(x, F::zero());
    let x = builder.get(&bits, 2);
    builder.assert_var_eq(x, F::one());

    for i in 3..31 {
        let x = builder.get(&bits, i);
        builder.assert_var_eq(x, F::zero());
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
    builder.assert_var_eq(x, F::one());
    let x = builder.get(&bits, RVar::one());
    builder.assert_var_eq(x, F::zero());
    let x = builder.get(&bits, 2);
    builder.assert_var_eq(x, F::one());

    for i in 3..31 {
        let x = builder.get(&bits, i);
        builder.assert_var_eq(x, F::zero());
    }

    builder.halt();

    let program = builder.compile_isa();
    println!("{}", program);
    execute_program(program, vec![]);
}
