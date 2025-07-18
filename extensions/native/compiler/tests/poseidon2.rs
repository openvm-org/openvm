use openvm_native_circuit::execute_program;
use openvm_native_compiler::{
    asm::AsmBuilder,
    ir::{Array, Var, PERMUTATION_WIDTH},
    prelude::RVar,
};
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::{config::baby_bear_poseidon2::default_perm, p3_baby_bear::BabyBear};
use p3_symmetric::Permutation;
use rand::{thread_rng, Rng};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_compiler_poseidon2_permute() {
    let mut rng = thread_rng();

    let mut builder = AsmBuilder::<F, EF>::default();

    let random_state_vals: [F; PERMUTATION_WIDTH] = rng.gen();
    // Execute the reference permutation
    let perm = default_perm();
    let expected_result = perm.permute(random_state_vals);

    // Execute the permutation in the VM
    // Initialize an array and populate it with the entries.
    let var_width: Var<F> = builder.eval(F::from_canonical_usize(PERMUTATION_WIDTH));
    let random_state = builder.array(var_width);
    for (i, val) in random_state_vals.iter().enumerate() {
        builder.set(&random_state, i, *val);
    }

    // Assert that the values are set correctly.
    for (i, val) in random_state_vals.iter().enumerate() {
        let res = builder.get(&random_state, i);
        builder.assert_felt_eq(res, *val);
    }

    let result = builder.poseidon2_permute(&random_state);

    assert!(matches!(result, Array::Dyn(_, _)));

    // Assert that the result is equal to the expected result.
    for (i, val) in expected_result.iter().enumerate() {
        let res = builder.get(&result, i);
        builder.assert_felt_eq(res, *val);
    }
    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_poseidon2_hash_1() {
    let mut rng = thread_rng();

    let mut builder = AsmBuilder::<F, EF>::default();

    let random_state_vals: [F; 42] = rng.gen();
    println!("{:?}", random_state_vals);
    let rlen = random_state_vals.len();
    let random_state_v2 = builder.dyn_array(rlen);
    for (i, val) in random_state_vals.iter().enumerate() {
        builder.set(&random_state_v2, i, *val);
    }
    let nested_random_state = builder.dyn_array(RVar::one());
    builder.set(&nested_random_state, RVar::zero(), random_state_v2);

    let result_x = builder.poseidon2_hash_x(&nested_random_state);

    builder.range(0, result_x.len()).for_each(|i_vec, builder| {
        let ei = builder.eval(i_vec[0]);
        builder.print_v(ei);
        let el_x = builder.get(&result_x, i_vec[0]);
        builder.print_f(el_x);
    });

    builder.halt();

    let program = builder.compile_isa();
    execute_program(program, vec![]);
}
