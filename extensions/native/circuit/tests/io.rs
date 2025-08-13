use openvm_native_circuit::execute_program;
use openvm_native_compiler::{
    asm::{AsmBuilder, AsmCompiler},
    conversion::{convert_program, CompilerOptions},
};
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

const WORD_SIZE: usize = 1;

#[test]
fn test_io() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let vars = builder.hint_vars();
    builder.range(0, vars.len()).for_each(|i_vec, builder| {
        let el = builder.get(&vars, i_vec[0]);
        builder.print_v(el);
    });

    let felts = builder.hint_felts();
    builder.range(0, felts.len()).for_each(|i_vec, builder| {
        let el = builder.get(&felts, i_vec[0]);
        builder.print_f(el);
    });

    let exts = builder.hint_exts();
    builder.range(0, exts.len()).for_each(|i_vec, builder| {
        let el = builder.get(&exts, i_vec[0]);
        builder.print_e(el);
    });

    builder.halt();

    let witness_stream: Vec<Vec<F>> = vec![
        vec![F::ZERO, F::ZERO, F::ONE],
        vec![F::ZERO, F::ZERO, F::TWO],
        vec![F::from_canonical_usize(3)],
        vec![
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE, // 1
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE, // 1
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::TWO, // 2
        ],
    ];

    let mut compiler = AsmCompiler::new(WORD_SIZE);
    compiler.build(builder.operations);
    let asm_code = compiler.code();
    println!("{}", asm_code);

    let program = convert_program::<F, EF>(asm_code, CompilerOptions::default());
    execute_program(program, witness_stream);
}
