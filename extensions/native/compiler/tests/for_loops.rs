use openvm_native_circuit::execute_program;
use openvm_native_compiler::{
    asm::{AsmBuilder, AsmConfig},
    ir::{Array, Var},
    prelude::ArrayLike,
};
use openvm_native_compiler_derive::iter_zip;
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_compiler_for_loops() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let n_val = BabyBear::from_canonical_u32(10);
    let m_val = BabyBear::from_canonical_u32(5);

    let zero: Var<_> = builder.eval(F::ZERO);
    let n: Var<_> = builder.eval(n_val);
    let m: Var<_> = builder.eval(m_val);

    let i_counter: Var<_> = builder.eval(F::ZERO);
    let total_counter: Var<_> = builder.eval(F::ZERO);
    builder.range(zero, n).for_each(|_, builder| {
        builder.assign(&i_counter, i_counter + F::ONE);

        let j_counter: Var<_> = builder.eval(F::ZERO);
        builder.range(zero, m).for_each(|_, builder| {
            builder.assign(&total_counter, total_counter + F::ONE);
            builder.assign(&j_counter, j_counter + F::ONE);
        });
        // Assert that the inner loop ran m times, in two different ways.
        builder.assert_var_eq(j_counter, m_val);
        builder.assert_var_eq(j_counter, m);
    });
    // Assert that the outer loop ran n times, in two different ways.
    builder.assert_var_eq(i_counter, n_val);
    builder.assert_var_eq(i_counter, n);
    // Assert that the total counter is equal to n * m, in two ways.
    builder.assert_var_eq(total_counter, n_val * m_val);
    builder.assert_var_eq(total_counter, n * m);

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_zip_fixed() {
    let mut builder = AsmBuilder::<F, EF>::default();
    let zero: Var<_> = builder.eval(F::ZERO);
    let one: Var<_> = builder.eval(F::ONE);
    let three: Var<_> = builder.eval(F::TWO + F::ONE);
    let four: Var<_> = builder.eval(F::TWO + F::TWO);
    let five: Var<_> = builder.eval(F::TWO + F::TWO + F::ONE);
    let arr1 = builder.vec(vec![zero, one]);
    let arr2 = builder.vec(vec![three, four, five]);

    let x1: Var<_> = builder.eval(F::ZERO);
    let x2: Var<_> = builder.eval(F::ZERO);
    let count: Var<_> = builder.eval(F::ZERO);
    let ptr1_cache: Var<_> = builder.eval(F::ZERO);
    let ptr2_cache: Var<_> = builder.eval(F::ZERO);

    iter_zip!(builder, arr1, arr2).for_each(|ptr_vec, builder| {
        let val1 = builder.iter_ptr_get(&arr1, ptr_vec[0]);
        let val2 = builder.iter_ptr_get(&arr2, ptr_vec[1]);
        builder.assign(&x1, x1 + val1);
        builder.assign(&x2, x2 + val2);
        builder.assign(&count, count + F::ONE);
        builder.assign(&ptr1_cache, ptr_vec[0]);
        builder.assign(&ptr2_cache, ptr_vec[1]);
    });
    builder.assert_var_eq(count, F::from_canonical_usize(2));
    builder.assert_var_eq(x1, F::from_canonical_usize(1));
    builder.assert_var_eq(x2, F::from_canonical_usize(7));
    builder.assert_var_eq(ptr1_cache, F::from_canonical_usize(1));
    builder.assert_var_eq(ptr2_cache, F::from_canonical_usize(1));
    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_zip_dyn() {
    let mut builder = AsmBuilder::<F, EF>::default();
    let zero: Var<_> = builder.eval(F::ZERO);
    let one: Var<_> = builder.eval(F::ONE);
    let three: Var<_> = builder.eval(F::TWO + F::ONE);
    let four: Var<_> = builder.eval(F::TWO + F::TWO);
    let five: Var<_> = builder.eval(F::TWO + F::TWO + F::ONE);
    let arr1 = builder.dyn_array(2);
    let arr2 = builder.dyn_array(3);
    builder.set(&arr1, 0, zero);
    builder.set(&arr1, 1, one);
    builder.set(&arr2, 0, three);
    builder.set(&arr2, 1, four);
    builder.set(&arr2, 2, five);

    let x1: Var<_> = builder.eval(F::ZERO);
    let x2: Var<_> = builder.eval(F::ZERO);
    let count: Var<_> = builder.eval(F::ZERO);
    let ptr1_cache: Var<_> = builder.eval(F::ZERO);
    let ptr2_cache: Var<_> = builder.eval(F::ZERO);

    iter_zip!(builder, arr1, arr2).for_each(|ptr_vec, builder| {
        let val1: Var<_> = builder.iter_ptr_get(&arr1, ptr_vec[0]);
        let val2: Var<_> = builder.iter_ptr_get(&arr2, ptr_vec[1]);
        builder.assign(&x1, x1 + val1);
        builder.assign(&x2, x2 + val2);
        builder.assign(&count, count + F::ONE);
        builder.assign(&ptr1_cache, ptr_vec[0]);
        builder.assign(&ptr2_cache, ptr_vec[1]);
    });
    builder.assert_var_eq(count, F::from_canonical_usize(2));
    builder.assert_var_eq(x1, F::from_canonical_usize(1));
    builder.assert_var_eq(x2, F::from_canonical_usize(7));
    builder.assert_var_eq(ptr1_cache, arr1.ptr().address + F::from_canonical_usize(1));
    builder.assert_var_eq(ptr2_cache, arr2.ptr().address + F::from_canonical_usize(1));
    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_nested_array_loop() {
    let mut builder = AsmBuilder::<F, EF>::default();
    type C = AsmConfig<F, EF>;

    let outer_len = 100;
    let inner_len = 10;

    let array: Array<C, Array<C, Var<_>>> = builder.dyn_array(outer_len);

    builder.range(0, array.len()).for_each(|i_vec, builder| {
        let inner_array = builder.dyn_array::<Var<_>>(inner_len);
        builder
            .range(0, inner_array.len())
            .for_each(|j_vec, builder| {
                builder.set(&inner_array, j_vec[0], i_vec[0] + j_vec[0]);
            });
        builder.set(&array, i_vec[0], inner_array);
    });

    // Test that the array is correctly initialized.
    builder.range(0, array.len()).for_each(|i_vec, builder| {
        let inner_array = builder.get(&array, i_vec[0]);
        builder
            .range(0, inner_array.len())
            .for_each(|j_vec, builder| {
                let val = builder.get(&inner_array, j_vec[0]);
                builder.assert_var_eq(val, i_vec[0] + j_vec[0]);
            });
    });

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_bneinc() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let n_val = BabyBear::from_canonical_u32(20);

    let zero: Var<_> = builder.eval(F::ZERO);
    let n: Var<_> = builder.eval(n_val);

    let i_counter: Var<_> = builder.eval(F::ZERO);
    builder.range(zero, n).for_each(|_, builder| {
        builder.assign(&i_counter, i_counter + F::ONE);
    });

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}
