use afs_compiler::{asm::AsmBuilder, prelude::*};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

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

    // let program = builder.compile_program();

    // let config = SC::default();
    // let mut runtime = Runtime::<F, EF, _>::new(&program, config.perm.clone());
    // runtime.run();
}
