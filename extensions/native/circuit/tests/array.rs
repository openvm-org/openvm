use openvm_native_circuit::execute_program;
use openvm_native_compiler::{
    asm::{AsmBuilder, AsmConfig},
    ir::{Array, Config, Ext, Felt, RVar, Usize, Var},
    prelude::{Builder, MemIndex, MemVariable, Ptr, Variable},
};
use openvm_native_compiler_derive::DslVariable;
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

#[derive(DslVariable, Clone, Debug)]
pub struct Point<C: Config> {
    x: Var<C::N>,
    y: Felt<C::F>,
    z: Ext<C::F, C::EF>,
}

#[test]
fn test_fixed_array_const() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    // Sum all the values of an array.
    let len: usize = 1000;
    let fixed_array = builder.vec(vec![Usize::from(1); len]);

    // Put values statically
    builder
        .range(0, fixed_array.len())
        .for_each(|idx_vec, builder| {
            builder.set_value(&fixed_array, idx_vec[0], Usize::from(2));
        });
    // Assert values set.
    builder
        .range(0, fixed_array.len())
        .for_each(|idx_vec, builder| {
            let value = builder.get(&fixed_array, idx_vec[0]);
            builder.assert_usize_eq(value, Usize::from(2));
        });
    let fixed_2d = builder.uninit_fixed_array(1);
    builder.set_value(&fixed_2d, RVar::zero(), fixed_array);

    assert_eq!(
        builder.operations.vec.len(),
        0,
        "No operations should be generated"
    );
}

#[test]
fn test_fixed_array_var() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    // Sum all the values of an array.
    let len: usize = 1000;
    let fixed_array = builder.uninit_fixed_array(len);

    // Put values statically
    builder
        .range(0, fixed_array.len())
        .for_each(|i_vec, builder| {
            let one: Var<_> = builder.eval(F::ONE);
            // `len` instructions
            builder.set(&fixed_array, i_vec[0], Usize::Var(one));
        });
    // Assert values set.
    builder
        .range(0, fixed_array.len())
        .for_each(|i_vec, builder| {
            let value: Usize<_> = builder.get(&fixed_array, i_vec[0]);
            // `len` instructions to initialize variables.
            // `len` instructions of `assert_eq`
            builder.assert_eq::<Var<_>>(value, RVar::from(2));
        });
    let fixed_2d = builder.uninit_fixed_array(1);
    builder.set_value(&fixed_2d, RVar::zero(), fixed_array);

    assert_eq!(
        builder.operations.vec.len(),
        len * 3,
        "No operations should be generated"
    );
}

#[test]
fn test_array_eq() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();
    let arr1: Array<_, Var<_>> = builder.dyn_array(2);
    builder.set(&arr1, 0, F::ONE);
    builder.set(&arr1, 1, F::TWO);
    let arr2: Array<_, Var<_>> = builder.dyn_array(2);
    builder.set(&arr2, 0, F::ONE);
    builder.set(&arr2, 1, F::TWO);
    builder.assert_var_array_eq(&arr1, &arr2);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[should_panic]
#[test]
fn test_array_eq_neg() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();
    let arr1: Array<_, Var<_>> = builder.dyn_array(2);
    builder.set(&arr1, 0, F::ONE);
    builder.set(&arr1, 1, F::TWO);
    let arr2: Array<_, Var<_>> = builder.dyn_array(2);
    builder.set(&arr2, 0, F::ONE);
    builder.set(&arr2, 1, F::ONE);
    builder.assert_var_array_eq(&arr1, &arr2);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_slice_variable_impl_happy_path() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type C = AsmConfig<F, EF>;

    const N: usize = 3;

    let mut builder = AsmBuilder::<F, EF>::default();
    let slice1: [Felt<F>; N] = builder.uninit();
    for (i, f) in slice1.iter().enumerate() {
        builder.assign(f, F::from_canonical_u32(i as u32));
    }
    let slice2: [Felt<F>; N] = builder.uninit();
    slice2.assign(slice1, &mut builder);
    builder.assert_eq::<[_; N]>(slice1, slice2);
    for (i, f) in slice2.iter().enumerate() {
        builder.assign(f, F::from_canonical_u32(i as u32));
    }
    let ptr = builder.alloc(1, <[Felt<F>; N] as MemVariable<C>>::size_of());
    let mem_index = MemIndex {
        index: RVar::zero(),
        offset: 0,
        size: N,
    };
    slice1.store(ptr, mem_index, &mut builder);
    let slice3: [Felt<F>; N] = builder.uninit();
    slice3.load(ptr, mem_index, &mut builder);
    builder.assert_eq::<[_; N]>(slice1, slice3);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
#[should_panic]
fn test_slice_assert_eq_neg() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    const N: usize = 3;

    let mut builder = AsmBuilder::<F, EF>::default();
    let slice1: [Felt<F>; N] = builder.uninit();
    for (i, f) in slice1.iter().enumerate() {
        builder.assign(f, F::from_canonical_u32(i as u32));
    }
    let slice2: [Felt<F>; N] = [builder.eval(F::ZERO); N];
    // Should panic because slice1 != slice2
    builder.assert_eq::<[_; N]>(slice1, slice2);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}
