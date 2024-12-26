use openvm_native_circuit::execute_program;
use openvm_native_compiler::{
    asm::AsmBuilder,
    ir::{Ext, Felt},
};
use openvm_stark_backend::p3_field::{
    extension::BinomialExtensionField, FieldAlgebra, FieldExtensionAlgebra,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{thread_rng, Rng};
#[test]
fn test_ext2felt() {
    const D: usize = 4;
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, D>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let mut rng = thread_rng();
    let val = rng.gen::<EF>();

    let ext: Ext<F, EF> = builder.constant(val);
    let felts = builder.ext2felt(ext);

    for (i, &fe) in val.as_base_slice().iter().enumerate() {
        let lhs = builder.get(&felts, i);
        let rhs: Felt<F> = builder.constant(fe);
        builder.assert_felt_eq(lhs, rhs);
    }
    builder.halt();

    let program = builder.compile_isa();
    println!("{}", program);
    execute_program(program, vec![]);
}

#[test]
fn test_ext_from_base_slice() {
    const D: usize = 4;
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, D>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let base_slice = &[
        F::from_canonical_usize(123),
        F::from_canonical_usize(234),
        F::from_canonical_usize(345),
        F::from_canonical_usize(456),
    ];

    let val = EF::from_base_slice(base_slice);
    let expected: Ext<_, _> = builder.constant(val);

    let felts = base_slice.map(|e| builder.constant::<Felt<_>>(e));
    let actual = builder.ext_from_base_slice(&felts);
    builder.assert_ext_eq(actual, expected);

    builder.halt();

    let program = builder.compile_isa();
    println!("{}", program);
    execute_program(program, vec![]);
}
