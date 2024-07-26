use afs_compiler::util::{display_program, execute_program};
use afs_compiler::{asm::AsmBuilder, ir::Var};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractField;
use stark_vm::cpu::WORD_SIZE;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_cycle_tracker() {
    let mut builder = AsmBuilder::<F, EF>::default();

    builder.cycle_tracker_start("test");

    let n_val = F::from_canonical_u32(10);
    let m_val = F::from_canonical_u32(20);

    let n: Var<_> = builder.eval(n_val);
    let m: Var<_> = builder.eval(m_val);

    let total: Var<_> = builder.eval(F::zero());

    builder.cycle_tracker_start("loop");

    for _ in 0..3 {
        let n_plus_m: Var<_> = builder.eval(n + m);
        builder.assign(total, total + n_plus_m);
    }

    builder.cycle_tracker_end("loop");

    builder.halt();

    builder.cycle_tracker_end("test");

    let program = builder.compile_isa::<WORD_SIZE>();
    display_program(&program);
    execute_program::<WORD_SIZE, _>(program, vec![]);
}
