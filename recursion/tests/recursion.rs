use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, PrimeField32};
use std::ops::Deref;

use afs_compiler::asm::AsmBuilder;
use afs_compiler::ir::{Config, Var};
use afs_recursion::stark::DynRapForRecursion;
use stark_vm::cpu::trace::Instruction;
use stark_vm::vm::config::VmConfig;
use stark_vm::vm::{ExecutionResult, ExecutionSegment, VirtualMachine};

use crate::common::sort_chips;

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
    let (chips, rec_raps, traces, pvs) = sort_chips(chips, rec_raps, traces, pvs);

    let vparams = common::make_verification_params(&chips, traces, &pvs);

    let (fib_verification_program, input_stream) =
        common::build_verification_program(rec_raps, pvs, vparams);

    let vm = VirtualMachine::<1, _>::new(vm_config, fib_verification_program, input_stream);
    vm.execute().unwrap();
}

pub fn get_rec_raps<const WORD_SIZE: usize, C: Config>(
    vm: &ExecutionSegment<WORD_SIZE, C::F>,
) -> Vec<&dyn DynRapForRecursion<C>>
where
    C::F: PrimeField32,
{
    let mut result: Vec<&dyn DynRapForRecursion<C>> = vec![
        &vm.cpu_chip.air,
        &vm.program_chip.air,
        &vm.memory_chip.air,
        &vm.range_checker.air,
    ];
    if vm.options().field_arithmetic_enabled {
        result.push(&vm.field_arithmetic_chip.air);
    }
    if vm.options().field_extension_enabled {
        result.push(&vm.field_extension_chip.air);
    }
    if vm.options().poseidon2_enabled() {
        result.push(&vm.poseidon2_chip.air);
    }
    result
}
