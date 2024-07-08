use afs_compiler::util::{display_program, execute_program};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractField;

use afs_compiler::asm::AsmBuilder;
use stark_vm::cpu::WORD_SIZE;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_io() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let arr = builder.hint_vars();
    builder.range(0, arr.len()).for_each(|i, builder| {
        let el = builder.get(&arr, i);
        builder.print_v(el);
    });

    let arr = builder.hint_felts();
    builder.range(0, arr.len()).for_each(|i, builder| {
        let el = builder.get(&arr, i);
        builder.print_f(el);
    });

    // let arr = builder.hint_exts();
    // builder.range(0, arr.len()).for_each(|i, builder| {
    //     let el = builder.get(&arr, i);
    //     builder.print_e(el);
    // });

    let program = builder.compile_isa();

    let witness_stream: Vec<Vec<F>> = vec![
        vec![F::zero(), F::zero(), F::one()],
        vec![F::zero(), F::zero(), F::two()],
        vec![F::one(), F::one(), F::two()],
    ];

    display_program(&program);
    execute_program::<WORD_SIZE, _>(program, witness_stream);

    // let config = SC::default();
    // let mut runtime = Runtime::<F, EF, _>::new(&program, config.perm.clone());
    // runtime.witness_stream = vec![
    //     vec![F::zero().into(), F::zero().into(), F::one().into()],
    //     vec![F::zero().into(), F::zero().into(), F::two().into()],
    //     vec![F::one().into(), F::one().into(), F::two().into()],
    // ]
    // .into();
    // runtime.run();
}
