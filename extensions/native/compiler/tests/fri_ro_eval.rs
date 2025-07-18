use openvm_native_circuit::execute_program;
use openvm_native_compiler::{
    asm::{AsmBuilder, AsmCompiler},
    conversion::{convert_program, CompilerOptions},
    ir::{Array, Ext, Felt},
};
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
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
    let mut mat_opening = Vec::with_capacity(n);
    let expected_mat_opening: Array<_, Felt<_>> = builder.dyn_array(n);

    for i in 0..n {
        let a_value = rng.gen::<F>();
        let b_value = rng.gen::<EF>();

        mat_opening.push(a_value);

        let val = builder.constant::<Felt<_>>(a_value);
        builder.set(&expected_mat_opening, i, val);
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
    builder.range(0, ps_at_z.len()).for_each(|t_vec, builder| {
        let t = t_vec[0];
        let p_at_x = builder.get(&expected_mat_opening, t);
        let p_at_z = builder.get(&ps_at_z, t);
        let quotient = (p_at_z - p_at_x) / (z - x);

        builder.assign(&cur_ro, cur_ro + cur_alpha_pow * quotient);
        builder.assign(&cur_alpha_pow, cur_alpha_pow * alpha);
    });
    let expected_result = cur_ro;

    builder.assign(&cur_alpha_pow, initial_alpha_pow);

    let hint_id = builder.hint_load();

    let ps_at_x = builder.dyn_array(n);
    let is_init = builder.constant(F::ZERO);
    let single_ro_eval_res =
        builder.fri_single_reduced_opening_eval(alpha, hint_id, is_init, &ps_at_x, &ps_at_z);

    let actual_result: Ext<_, _> = builder.uninit();
    builder.assign(&actual_result, single_ro_eval_res * cur_alpha_pow / (z - x));

    builder.assert_ext_eq(expected_result, actual_result);

    // Check that `ps_at_x` were filled by chip.
    builder.assert_var_eq(expected_mat_opening.len(), ps_at_x.len());
    builder
        .range(0, ps_at_x.len())
        .for_each(|idx_vec, builder| {
            let l = builder.get(&expected_mat_opening, idx_vec[0]);
            let r = builder.get(&ps_at_x, idx_vec[0]);
            builder.assert_felt_eq(l, r);
        });

    builder.halt();

    let mut compiler = AsmCompiler::new(1);
    compiler.build(builder.operations);
    let asm_code = compiler.code();

    let program = convert_program::<F, EF>(asm_code, CompilerOptions::default());
    execute_program(program, vec![mat_opening]);
}
