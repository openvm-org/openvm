use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractField;

use afs_compiler::asm::AsmBuilder;
use afs_compiler::ir::{Usize, Var};
use afs_compiler::util::execute_program;
use stark_vm::cpu::WORD_SIZE;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_compiler_conditionals() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let zero: Var<_> = builder.eval(F::zero());
    let one: Var<_> = builder.eval(F::one());
    let two: Var<_> = builder.eval(F::two());
    let three: Var<_> = builder.eval(F::from_canonical_u32(3));
    let four: Var<_> = builder.eval(F::from_canonical_u32(4));

    let c: Var<_> = builder.eval(F::zero());
    builder.if_eq(zero, zero).then(|builder| {
        builder.if_eq(one, one).then(|builder| {
            builder.if_eq(two, two).then(|builder| {
                builder.if_eq(three, three).then(|builder| {
                    builder
                        .if_eq(four, four)
                        .then(|builder| builder.assign(c, F::one()))
                })
            })
        })
    });
    builder.assert_var_eq(c, F::one());

    let c: Var<_> = builder.eval(F::zero());
    builder.if_eq(zero, one).then_or_else(
        |builder| {
            builder.if_eq(one, one).then(|builder| {
                builder
                    .if_eq(two, two)
                    .then(|builder| builder.assign(c, F::one()))
            })
        },
        |builder| {
            builder
                .if_ne(three, four)
                .then_or_else(|_| {}, |builder| builder.assign(c, F::zero()))
        },
    );
    builder.assert_var_eq(c, F::zero());

    builder.halt();

    let program = builder.compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}

#[test]
fn test_compiler_conditionals_v2() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let zero: Var<_> = builder.eval(F::zero());
    let one: Var<_> = builder.eval(F::one());
    let two: Var<_> = builder.eval(F::two());
    let three: Var<_> = builder.eval(F::from_canonical_u32(3));
    let four: Var<_> = builder.eval(F::from_canonical_u32(4));

    let c: Var<_> = builder.eval(F::zero());
    builder.if_eq(zero, zero).then(|builder| {
        builder.if_eq(one, one).then(|builder| {
            builder.if_eq(two, two).then(|builder| {
                builder.if_eq(three, three).then(|builder| {
                    builder
                        .if_eq(four, four)
                        .then(|builder| builder.assign(c, F::one()))
                })
            })
        })
    });

    builder.halt();

    let program = builder.compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}

#[test]
fn test_compiler_conditionals_const() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let zero: Usize<_> = builder.eval(0);
    let one: Usize<_> = builder.eval(1);
    let two: Usize<_> = builder.eval(2);
    let three: Usize<_> = builder.eval(3);
    let four: Usize<_> = builder.eval(4);

    // 1 instruction to evaluate the variable.
    let c: Var<_> = builder.eval(F::zero());
    builder.if_ne(zero, one).then(|builder| {
        builder.if_eq(zero, zero).then(|builder| {
            builder.if_eq(one, one).then(|builder| {
                builder.if_eq(two, two).then(|builder| {
                    builder.if_eq(three, three).then(|builder| {
                        builder
                            .if_eq(four, four)
                            // 1 instruction to assign the variable.
                            .then(|builder| builder.assign(c, F::one()))
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
