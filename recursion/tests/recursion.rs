use afs_recursion::stark::get_rec_raps;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractField;
use std::ops::Deref;

use afs_compiler::asm::AsmBuilder;
use afs_compiler::ir::Var;
use stark_vm::cpu::trace::Instruction;
use stark_vm::vm::config::VmConfig;
use stark_vm::vm::{ExecutionResult, VirtualMachine};

mod common;

fn fibonacci_program(a: u32, b: u32, n: u32) -> Vec<Instruction<BabyBear>> {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let prev: Var<_> = builder.constant(F::from_canonical_u32(a));
    let next: Var<_> = builder.constant(F::from_canonical_u32(b));

    for _ in 0..n {
        let tmp: Var<_> = builder.uninit();
        builder.assign(tmp, next);
        builder.assign(next, prev + next);
        builder.assign(prev, tmp);
    }

    builder.halt();

    builder.compile_isa::<1>()
}

#[test]
fn test_fibonacci_program_verify() {
    let fib_program = fibonacci_program(0, 1, 32);

    let vm_config = VmConfig {
        max_segment_len: 2000000,
        ..Default::default()
    };

    let dummy_vm = VirtualMachine::<1, _>::new(vm_config, fib_program.clone(), vec![]);
    let rec_raps = get_rec_raps(&dummy_vm.segments[0]);

    let vm = VirtualMachine::<1, _>::new(vm_config, fib_program, vec![]);
    let ExecutionResult {
        nonempty_traces: traces,
        nonempty_chips: chips,
        nonempty_pis: pvs,
        ..
    } = vm.execute().unwrap();

    let chips = chips.iter().map(|x| x.deref()).collect();

    common::run_recursive_test(chips, rec_raps, traces, pvs);
}
