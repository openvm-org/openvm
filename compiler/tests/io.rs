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

    let vars = builder.hint_vars();
    builder.range(0, vars.len()).for_each(|i, builder| {
        let el = builder.get(&vars, i);
        builder.print_v(el);
    });

    let felts = builder.hint_felts();
    builder.range(0, felts.len()).for_each(|i, builder| {
        let el = builder.get(&felts, i);
        builder.print_f(el);
    });

    let exts = builder.hint_exts();
    builder.range(0, exts.len()).for_each(|i, builder| {
        let el = builder.get(&exts, i);
        builder.print_e(el);
    });

    builder.halt();

    let witness_stream: Vec<Vec<F>> = vec![
        vec![F::zero(), F::zero(), F::one()],
        vec![F::zero(), F::zero(), F::two()],
        vec![F::from_canonical_usize(3)],
        vec![
            F::zero(),
            F::zero(),
            F::zero(),
            F::one(), // 1
            F::zero(),
            F::zero(),
            F::zero(),
            F::one(), // 1
            F::zero(),
            F::zero(),
            F::zero(),
            F::two(), // 2
        ],
    ];

    let program = builder.compile_isa::<1>();
    display_program(&program);
    execute_program::<WORD_SIZE>(program, witness_stream);
}
