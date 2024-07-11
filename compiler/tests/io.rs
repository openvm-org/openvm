use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractField;

use afs_compiler::asm::AsmBuilder;
use afs_compiler::util::{display_program, execute_program};
use stark_vm::cpu::WORD_SIZE;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_io() {
    let mut builder = AsmBuilder::<F, EF>::default();

    let vars = builder.hint();
    builder.range(0, vars.len()).for_each(|i, builder| {
        let el = builder.get(&vars, i);
        builder.print_v(el);
    });

    let felts = builder.hint();
    builder.range(0, felts.len()).for_each(|i, builder| {
        let el = builder.get(&felts, i);
        builder.print_f(el);
    });

    // TODO[INT-1727]: support AsmInstruction::LoadE
    // let exts = builder.hint();
    // builder.range(0, exts.len()).for_each(|i, builder| {
    //     let el = builder.get(&exts, i);
    //     builder.print_e(el);
    // });

    builder.halt();

    let program = builder.compile_isa::<WORD_SIZE>();

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
