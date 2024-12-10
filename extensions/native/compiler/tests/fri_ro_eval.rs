use ax_stark_backend::p3_field::{extension::BinomialExtensionField, AbstractField};
use ax_stark_sdk::p3_baby_bear::BabyBear;
use axvm_native_circuit::execute_program;
use axvm_native_compiler::{
    asm::{AsmBuilder, AsmCompiler},
    conversion::{convert_program, CompilerOptions},
    ir::{Array, Ext, Felt},
};
use rand::{thread_rng, Rng};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_single_reduced_opening_eval() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let mut rng = thread_rng();
    let n = 3;

    let alpha_value = rng.gen::<EF>();
    let initial_alpha_pow_value = rng.gen::<EF>();
    let x_value = rng.gen::<EF>();
    let z_value = rng.gen::<EF>();

    let ps_at_z: Array<_, Ext<_, _>> = builder.dyn_array(n);
    let mat_opening: Array<_, Felt<_>> = builder.dyn_array(n);

    for i in 0..n {
        let a_value = rng.gen::<F>();
        let b_value = rng.gen::<EF>();
        let val = builder.constant::<Felt<_>>(a_value);
        builder.set(&mat_opening, i, val);
        let val = builder.constant::<Ext<_, _>>(b_value);
        builder.set(&ps_at_z, i, val);
    }

    let alpha: Ext<_, _> = builder.constant(alpha_value);
    let initial_alpha_pow: Ext<_, _> = builder.constant(initial_alpha_pow_value);
    let x: Ext<_, _> = builder.constant(x_value);
    let z: Ext<_, _> = builder.constant(z_value);

    let cur_ro: Ext<_, _> = builder.constant(EF::ZERO);
    let cur_alpha_pow: Ext<_, _> = builder.uninit();
    builder.assign(&cur_alpha_pow, initial_alpha_pow);
    builder.range(0, ps_at_z.len()).for_each(|t, builder| {
        let p_at_x = builder.get(&mat_opening, t);
        let p_at_z = builder.get(&ps_at_z, t);
        let quotient = (p_at_z - p_at_x) / (z - x);

        builder.assign(&cur_ro, cur_ro + cur_alpha_pow * quotient);
        builder.assign(&cur_alpha_pow, cur_alpha_pow * alpha);
    });
    let expected_result = cur_ro;
    let expected_final_alpha_pow = cur_alpha_pow;

    // prints don't work?
    /*builder.print_e(expected_result);
    builder.print_e(expected_final_alpha_pow);

    let two = builder.constant(F::TWO);
    builder.print_f(two);
    let ext_1210 = builder.constant(EF::from_base_slice(&[F::ONE, F::TWO, F::ONE, F::ZERO]));
    builder.print_e(ext_1210);*/

    let cur_alpha_pow: Ext<_, _> = builder.uninit();
    builder.assign(&cur_alpha_pow, initial_alpha_pow);
    let single_ro_eval_res =
        builder.fri_single_reduced_opening_eval(alpha, cur_alpha_pow, &mat_opening, &ps_at_z);
    let actual_final_alpha_pow = cur_alpha_pow;
    let actual_result: Ext<_, _> = builder.uninit();
    builder.assign(&actual_result, single_ro_eval_res / (z - x));

    //builder.print_e(actual_result);
    //builder.print_e(actual_final_alpha_pow);

    builder.assert_ext_eq(expected_result, actual_result);
    builder.assert_ext_eq(expected_final_alpha_pow, actual_final_alpha_pow);

    builder.halt();

    let mut compiler = AsmCompiler::new(1);
    compiler.build(builder.operations);
    let asm_code = compiler.code();
    // println!("{}", asm_code);

    let program = convert_program::<F, EF>(asm_code, CompilerOptions::default());
    execute_program(program, vec![]);
}
