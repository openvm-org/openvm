use afs_compiler::{asm::AsmBuilder, ir::Var};
use axvm_circuit::system::program::util::execute_program;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};

#[test]
fn test_compiler_loop() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let n = F::from_canonical_usize(100);

    let var: Var<_> = builder.constant(F::zero());
    builder.do_loop(|builder| {
        builder.assign(&var, var + F::one());
        builder
            .if_eq(var, n)
            .then_may_break(|builder| builder.break_loop())
    });
    builder.assert_var_eq(var, n);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}
