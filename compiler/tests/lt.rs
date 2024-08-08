use afs_compiler::{asm::AsmBuilder, prelude::*, util::execute_program};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

const WORD_SIZE: usize = 1;

#[test]
fn test_compiler_less_than() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let a: Var<_> = builder.constant(F::from_canonical_u32(10));
    let b: Var<_> = builder.constant(F::from_canonical_u32(20));
    let c = builder.lt(a, b);
    builder.assert_var_eq(c, F::one());

    let a: Var<_> = builder.constant(F::from_canonical_u32(20));
    let b: Var<_> = builder.constant(F::from_canonical_u32(10));
    let c = builder.lt(a, b);
    builder.assert_var_eq(c, F::zero());

    builder.halt();

    let program = builder.clone().compile_isa::<WORD_SIZE>();
    execute_program::<WORD_SIZE>(program, vec![]);
}
