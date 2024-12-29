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

// #[test]
// #[ignore = "test too slow"]
// fn test_compiler_array() {
//     type F = BabyBear;
//     type EF = BinomialExtensionField<BabyBear, 4>;
//
//     let mut builder = AsmBuilder::<F, EF>::default();
//
//     // Sum all the values of an array.
//     let len: usize = 1000;
//     let mut rng = thread_rng();
//
//     let static_array = builder.array::<Var<_>>(len);
//
//     // Put values statically
//     for i in 0..len {
//         builder.set(&static_array, i, F::ONE);
//     }
//     // Assert values set.
//     for i in 0..len {
//         let value = builder.get(&static_array, i);
//         builder.assert_var_eq(value, F::ONE);
//     }
//
//     let dyn_len: Var<_> = builder.eval(F::from_canonical_usize(len));
//     let var_array = builder.array::<Var<_>>(dyn_len);
//     let felt_array = builder.array::<Felt<_>>(dyn_len);
//     let ext_array = builder.array::<Ext<_, _>>(dyn_len);
//     // Put values statically
//     let var_vals = (0..len).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
//     let felt_vals = (0..len).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
//     let ext_vals = (0..len).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();
//     for i in 0..len {
//
//         builder.set(&var_array, i, var_vals[i]);
//         builder.set(&felt_array, i, felt_vals[i]);
//         builder.set(&ext_array, i, ext_vals[i].cons());
//     }
//     // Assert values set.
//     for i in 0..len {
//         let var_value = builder.get(&var_array, i);
//         builder.assert_var_eq(var_value, var_vals[i]);
//         let felt_value = builder.get(&felt_array, i);
//         builder.assert_felt_eq(felt_value, felt_vals[i]);
//         let ext_value = builder.get(&ext_array, i);
//         builder.assert_ext_eq(ext_value, ext_vals[i].cons());
//     }
//
//     // Put values dynamically
//     builder.range(0, dyn_len).for_each(|i, builder| {
//         builder.set(&var_array, i, i * 2);
//         builder.set(&felt_array, i, F::from_canonical_u32(3));
//         builder.set(&ext_array, i, EF::from_canonical_u32(4).cons());
//     });
//
//     // Assert values set.
//     builder.range(0, dyn_len).for_each(|i, builder| {
//         let var_value = builder.get(&var_array, i);
//         builder.assert_var_eq(var_value, i * 2);
//         let felt_value = builder.get(&felt_array, i);
//         builder.assert_felt_eq(felt_value, F::from_canonical_u32(3));
//         let ext_value = builder.get(&ext_array, i);
//         builder.assert_ext_eq(ext_value, EF::from_canonical_u32(4).cons());
//     });
//
//     // Test the derived macro and mixed size allocations.
//     let point_array = builder.dyn_array::<Point<_>>(len);
//
//     builder.range(0, dyn_len).for_each(|i, builder| {
//         let x: Var<_> = builder.eval(F::TWO);
//         let y: Felt<_> = builder.eval(F::ONE);
//         let z: Ext<_, _> = builder.eval(EF::ONE.cons());
//         let point = Point { x, y, z };
//         builder.set(&point_array, i, point);
//     });
//
//     builder.range(0, dyn_len).for_each(|i, builder| {
//         let point = builder.get(&point_array, i);
//         builder.assert_var_eq(point.x, F::TWO);
//         builder.assert_felt_eq(point.y, F::ONE);
//         builder.assert_ext_eq(point.z, EF::ONE.cons());
//     });
//
//     let array = builder.dyn_array::<Array<_, Var<_>>>(len);
//
//     builder.range(0, array.len()).for_each(|i, builder| {
//         builder.set(&array, i, var_array.clone());
//     });
//
//     // TODO: this part of the test is extremely slow.
//     builder.range(0, array.len()).for_each(|i, builder| {
//         let point_array_back = builder.get(&array, i);
//         builder.assert_eq::<Array<_, _>>(point_array_back, var_array.clone());
//     });
//
//     builder.halt();
//
//     let program = builder.compile_isa();
//     execute_program(program, vec![]);
// }

#[test]
fn test_fixed_array_const() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    // Sum all the values of an array.
    let len: usize = 1000;
    let fixed_array = builder.vec(vec![Usize::from(1); len]);

    // Put values statically
    builder.range(0, fixed_array.len()).for_each(|i, builder| {
        builder.set_value(&fixed_array, i, Usize::from(2));
    });
    // Assert values set.
    builder.range(0, fixed_array.len()).for_each(|i, builder| {
        let value = builder.get(&fixed_array, i);
        builder.assert_eq::<Usize<_>>(value, Usize::from(2));
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
    builder.range(0, fixed_array.len()).for_each(|i, builder| {
        let one: Var<_> = builder.eval(F::ONE);
        // `len` instructions
        builder.set(&fixed_array, i, Usize::Var(one));
    });
    // Assert values set.
    builder.range(0, fixed_array.len()).for_each(|i, builder| {
        let value: Usize<_> = builder.get(&fixed_array, i);
        // `len` instructions to initialize variables. FIXME: this is not optimal.
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
