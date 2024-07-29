use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, PrimeField32};

use afs_compiler::asm::AsmBuilder;
use afs_compiler::ir::{Config, Var};
use afs_recursion::stark::DynRapForRecursion;
use stark_vm::cpu::trace::Instruction;
use stark_vm::vm::config::VmConfig;
use stark_vm::vm::{get_chips, VirtualMachine};

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

    let mut vm = VirtualMachine::<1, _>::new(VmConfig::default(), fib_program, vec![]);
    let traces = vm.traces().unwrap();

    let chips = get_chips(&vm);
    let num_chips = chips.len();
    let pvs = vec![vec![]; num_chips];
    let rec_raps = get_rec_raps(&vm);

    let (chips, rec_raps, traces, pvs) = sort_chips(chips, rec_raps, traces, pvs);

    let vparams = common::make_verification_params(&chips, traces, &pvs);

    let (fib_verification_program, input_stream) =
        common::build_verification_program(rec_raps, pvs, vparams);

    let mut vm =
        VirtualMachine::<1, _>::new(VmConfig::default(), fib_verification_program, input_stream);
    vm.traces().unwrap();
}

pub fn get_rec_raps<const WORD_SIZE: usize, C: Config>(
    vm: &VirtualMachine<WORD_SIZE, C::F>,
) -> Vec<&dyn DynRapForRecursion<C>>
where
    C::F: PrimeField32,
{
    let mut result: Vec<&dyn DynRapForRecursion<C>> = vec![
        &vm.cpu_air,
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
