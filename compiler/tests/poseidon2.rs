use afs_compiler::{asm::AsmBuilder, prelude::RVar, util::end_to_end_test};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use rand::{thread_rng, Rng};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

const WORD_SIZE: usize = 1;

// #[test]
// fn test_compiler_poseidon2_permute() {
//     let mut rng = thread_rng();
//
//     let mut builder = AsmBuilder::<F, EF>::default();
//
//     let random_state_vals: [F; PERMUTATION_WIDTH] = rng.gen();
//     // Execute the reference permutation
//     let expected_result = perm.permute(random_state_vals);
//
//     // Execture the permutation in the VM
//     // Initialize an array and populate it with the entries.
//     let var_width: Var<F> = builder.eval(F::from_canonical_usize(PERMUTATION_WIDTH));
//     let mut random_state = builder.array(var_width);
//     for (i, val) in random_state_vals.iter().enumerate() {
//         builder.set(&mut random_state, i, *val);
//     }
//
//     // Assert that the values are set correctly.
//     for (i, val) in random_state_vals.iter().enumerate() {
//         let res = builder.get(&random_state, i);
//         builder.assert_felt_eq(res, *val);
//     }
//
//     let result = builder.poseidon2_permute(&random_state);
//
//     assert!(matches!(result, Array::Dyn(_, _)));
//
//     // Assert that the result is equal to the expected result.
//     for (i, val) in expected_result.iter().enumerate() {
//         let res = builder.get(&result, i);
//         builder.assert_felt_eq(res, *val);
//     }
//
//     // let program = builder.compile_program();
//
//     // let mut runtime = Runtime::<F, EF, _>::new(&program, config.perm.clone());
//     // runtime.run();
//     // println!(
//     //     "The program executed successfully, number of cycles: {}",
//     //     runtime.clk.as_canonical_u32() / 4
//     // );
// }

#[test]
fn test_compiler_poseidon2_hash_1() {
    let mut rng = thread_rng();

    let mut builder = AsmBuilder::<F, EF>::default();

    let random_state_vals: [F; 42] = rng.gen();
    println!("{:?}", random_state_vals);
    let rlen = random_state_vals.len();
    let mut random_state_v1 = builder.dyn_array(rlen);
    for (i, val) in random_state_vals.iter().enumerate() {
        builder.set(&mut random_state_v1, i, *val);
    }
    let mut random_state_v2 = builder.dyn_array(rlen);
    for (i, val) in random_state_vals.iter().enumerate() {
        builder.set(&mut random_state_v2, i, *val);
    }
    let mut nested_random_state = builder.dyn_array(RVar::one());
    builder.set(&mut nested_random_state, RVar::zero(), random_state_v2);

    let result = builder.poseidon2_hash(&random_state_v1);
    let result_x = builder.poseidon2_hash_x(&nested_random_state);

    builder.range(0, result.len()).for_each(|i, builder| {
        let ei = builder.eval(i);
        builder.print_v(ei);
        let el = builder.get(&result, i);
        builder.print_f(el);
        let el_x = builder.get(&result_x, i);
        builder.print_f(el_x);
        builder.assert_felt_eq(el, el_x);
    });

    builder.halt();

    end_to_end_test::<WORD_SIZE, _>(builder, vec![]);
}
