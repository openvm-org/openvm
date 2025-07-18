use openvm_native_circuit::execute_program;
use openvm_native_compiler::{asm::AsmBuilder, ir::Var};
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_compiler_conditionals() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let zero: Var<_> = builder.eval(F::ZERO);
    let one: Var<_> = builder.eval(F::ONE);
    let two: Var<_> = builder.eval(F::TWO);
    let three: Var<_> = builder.eval(F::from_canonical_u32(3));
    let four: Var<_> = builder.eval(F::from_canonical_u32(4));

    let c: Var<_> = builder.eval(F::ZERO);
    builder.if_eq(zero, zero).then(|builder| {
        builder.if_eq(one, one).then(|builder| {
            builder.if_eq(two, two).then(|builder| {
                builder.if_eq(three, three).then(|builder| {
                    builder
                        .if_eq(four, four)
                        .then(|builder| builder.assign(&c, F::ONE))
                })
            })
        })
    });
    builder.assert_var_eq(c, F::ONE);

    let c: Var<_> = builder.eval(F::ZERO);
    builder.if_eq(zero, one).then_or_else(
        |builder| {
            builder.if_eq(one, one).then(|builder| {
                builder
                    .if_eq(two, two)
                    .then(|builder| builder.assign(&c, F::ONE))
            })
        },
        |builder| {
            builder
                .if_ne(three, four)
                .then_or_else(|_| {}, |builder| builder.assign(&c, F::ZERO))
        },
    );
    builder.assert_var_eq(c, F::ZERO);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_conditionals_v2() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let zero: Var<_> = builder.eval(F::ZERO);
    let one: Var<_> = builder.eval(F::ONE);
    let two: Var<_> = builder.eval(F::TWO);
    let three: Var<_> = builder.eval(F::from_canonical_u32(3));
    let four: Var<_> = builder.eval(F::from_canonical_u32(4));

    let c: Var<_> = builder.eval(F::ZERO);
    builder.if_eq(zero, zero).then(|builder| {
        builder.if_eq(one, one).then(|builder| {
            builder.if_eq(two, two).then(|builder| {
                builder.if_eq(three, three).then(|builder| {
                    builder
                        .if_eq(four, four)
                        .then(|builder| builder.assign(&c, F::ONE))
                })
            })
        })
    });

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_conditionals_const() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let zero = builder.eval_expr(F::ZERO);
    let one = builder.eval_expr(F::ONE);
    let two = builder.eval_expr(F::from_canonical_u32(2));
    let three = builder.eval_expr(F::from_canonical_u32(3));
    let four = builder.eval_expr(F::from_canonical_u32(4));

    // 1 instruction to evaluate the variable.
    let c: Var<_> = builder.eval(F::ZERO);
    builder.if_ne(zero, one).then(|builder| {
        builder.if_eq(zero, zero).then(|builder| {
            builder.if_eq(one, one).then(|builder| {
                builder.if_eq(two, two).then(|builder| {
                    builder.if_eq(three, three).then(|builder| {
                        builder
                            .if_eq(four, four)
                            // 1 instruction to assign the variable.
                            .then(|builder| builder.assign(&c, F::ONE))
                    })
                })
            })
        })
    });

    assert_eq!(
        builder.operations.vec.len(),
        2,
        "Constant conditionals should be optimized"
    );
}
