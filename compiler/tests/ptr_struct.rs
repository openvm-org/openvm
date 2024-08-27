use afs_compiler::{
    asm::AsmBuilder,
    ir::{Array, Config, Ext, ExtConst, Felt, RVar, Ref, Var},
    prelude::{Builder, MemIndex, MemVariable, Ptr, Variable},
    util::execute_program,
};
use afs_derive::DslVariable;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use rand::{thread_rng, Rng};

#[derive(DslVariable, Clone, Debug)]
pub struct Point<C: Config> {
    x: Ref<C, Var<C::N>>,
    y: Ref<C, Felt<C::F>>,
    z: Ref<C, Ext<C::F, C::EF>>,
}

#[test]
#[ignore = "test too slow"]
fn test_compiler_array() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    // Sum all the values of an array.
    let len: usize = 3;
    let mut rng = thread_rng();

    let mut static_array = builder.array::<Var<_>>(len);

    // Put values statically
    for i in 0..len {
        builder.set(&mut static_array, i, F::one());
    }
    // Assert values set.
    for i in 0..len {
        let value = builder.get(&static_array, i);
        builder.assert_var_eq(value, F::one());
    }

    let dyn_len: Var<_> = builder.eval(F::from_canonical_usize(len));
    let mut var_array = builder.array::<Var<_>>(dyn_len);
    let mut felt_array = builder.array::<Felt<_>>(dyn_len);
    let mut ext_array = builder.array::<Ext<_, _>>(dyn_len);
    // Put values statically
    let var_vals = (0..len).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
    let felt_vals = (0..len).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
    let ext_vals = (0..len).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();
    for i in 0..len {
        builder.set(&mut var_array, i, var_vals[i]);
        builder.set(&mut felt_array, i, felt_vals[i]);
        builder.set(&mut ext_array, i, ext_vals[i].cons());
    }
    // Assert values set.
    for i in 0..len {
        let var_value = builder.get(&var_array, i);
        builder.assert_var_eq(var_value, var_vals[i]);
        let felt_value = builder.get(&felt_array, i);
        builder.assert_felt_eq(felt_value, felt_vals[i]);
        let ext_value = builder.get(&ext_array, i);
        builder.assert_ext_eq(ext_value, ext_vals[i].cons());
    }

    // Put values dynamically
    builder.range(0, dyn_len).for_each(|i, builder| {
        builder.set(
            &mut var_array,
            i,
            i * RVar::from_field(F::from_canonical_u32(2)),
        );
        builder.set(&mut felt_array, i, F::from_canonical_u32(3));
        builder.set(&mut ext_array, i, EF::from_canonical_u32(4).cons());
    });

    // Assert values set.
    builder.range(0, dyn_len).for_each(|i, builder| {
        let var_value = builder.get(&var_array, i);
        builder.assert_var_eq(var_value, i * RVar::from_field(F::from_canonical_u32(2)));
        let felt_value = builder.get(&felt_array, i);
        builder.assert_felt_eq(felt_value, F::from_canonical_u32(3));
        let ext_value = builder.get(&ext_array, i);
        builder.assert_ext_eq(ext_value, EF::from_canonical_u32(4).cons());
    });

    // Test the derived macro and mixed size allocations.
    let mut point_array = builder.dyn_array::<Point<_>>(len);

    builder.range(0, dyn_len).for_each(|i, builder| {
        let x: Var<_> = builder.eval(F::two());
        let mut x_ptr: Ref<_, Var<_>> = builder.uninit();
        builder.set_to_value(&mut x_ptr, x);
        let y: Felt<_> = builder.eval(F::one());
        let mut y_ptr: Ref<_, Felt<_>> = builder.uninit();
        builder.set_to_value(&mut y_ptr, y);
        let z: Ext<_, _> = builder.eval(EF::one().cons());
        let mut z_ptr: Ref<_, Ext<_, _>> = builder.uninit();
        builder.set_to_value(&mut z_ptr, z);
        let point = Point {
            x: x_ptr,
            y: y_ptr,
            z: z_ptr,
        };
        builder.set(&mut point_array, i, point);
    });

    builder.range(0, dyn_len).for_each(|i, builder| {
        let point = builder.get(&point_array, i);
        let x = builder.deref(&point.x);
        let y = builder.deref(&point.y);
        let z = builder.deref(&point.z);
        builder.assert_var_eq(x, F::two());
        builder.assert_felt_eq(y, F::one());
        builder.assert_ext_eq(z, EF::one().cons());
    });

    let mut array = builder.dyn_array::<Array<_, Var<_>>>(len);

    builder.range(0, array.len()).for_each(|i, builder| {
        builder.set(&mut array, i, var_array.clone());
    });

    // TODO: this part of the test is extremely slow.
    builder.range(0, array.len()).for_each(|i, builder| {
        let point_array_back = builder.get(&array, i);
        builder.assert_eq::<Array<_, _>>(point_array_back, var_array.clone());
    });

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}
