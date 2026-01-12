use openvm_native_circuit::execute_program;
use openvm_native_compiler::{
    asm::AsmBuilder,
    ir::{Ext, Felt},
};
use openvm_stark_backend::p3_field::{
    extension::BinomialExtensionField, BasedVectorSpace, PrimeCharacteristicRing,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;
#[test]
fn test_ext2felt() {
    const D: usize = 4;
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, D>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let mut rng = create_seeded_rng();
    let val = rng.random::<EF>();

    let ext: Ext<F, EF> = builder.constant(val);
    let felts = builder.ext2felt(ext);

    for (i, &fe) in val.as_basis_coefficients_slice().iter().enumerate() {
        let lhs = builder.get(&felts, i);
        let rhs: Felt<F> = builder.constant(fe);
        builder.assert_felt_eq(lhs, rhs);
    }
    builder.halt();

    let program = builder.compile_isa();
    println!("{program}");
    execute_program(program, vec![]);
}

#[test]
fn test_ext_from_slice() {
    const D: usize = 4;
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, D>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let base_slice = &[
        F::from_usize(123),
        F::from_usize(234),
        F::from_usize(345),
        F::from_usize(456),
    ];

    let val = EF::from_basis_coefficients_slice(base_slice).unwrap();
    let expected: Ext<_, _> = builder.constant(val);

    let felts = base_slice.map(|e| builder.constant::<Felt<_>>(e));
    let actual = builder.ext_from_base_slice(&felts);
    builder.assert_ext_eq(actual, expected);

    builder.halt();

    let program = builder.compile_isa();
    println!("{program}");
    execute_program(program, vec![]);
}
