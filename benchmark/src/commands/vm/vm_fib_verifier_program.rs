use std::ops::Deref;

use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};
use afs_recursion::{
    stark::{get_rec_raps, sort_chips},
    types::InnerConfig,
};
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use stark_vm::vm::{config::VmConfig, ExecutionAndTraceGenerationResult, VirtualMachine};

use super::benchmark_helpers::run_recursive_test_benchmark;

pub fn benchmark_fib_verifier_program(n: usize) -> Result<()> {
    const NUM_WORDS: usize = 8;
    const WORD_SIZE: usize = 1;

    println!(
        "Running verifier program of VM STARK benchmark with n = {}",
        n
    );

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();
    let a: Felt<_> = builder.eval(F::zero());
    let b: Felt<_> = builder.eval(F::one());
    let n_ext: Var<_> = builder.eval(F::from_canonical_usize(n));

    let start: Var<_> = builder.eval(F::zero());
    let end = n_ext;

    builder.range(start, end).for_each(|_, builder| {
        let temp: Felt<_> = builder.uninit();
        builder.assign(&temp, b);
        builder.assign(&b, a + b);
        builder.assign(&a, temp);
    });

    builder.halt();

    let fib_program = builder.compile_isa::<1>();

    let vm_config = VmConfig {
        max_segment_len: 2000000,
        ..Default::default()
    };

    let vm = VirtualMachine::<8, 1, _>::new(vm_config, fib_program.clone(), vec![]);

    let ExecutionAndTraceGenerationResult {
        max_log_degree: _,
        nonempty_chips: chips,
        nonempty_traces: traces,
        nonempty_pis: pis,
        ..
    } = vm.execute_and_generate_traces().unwrap();
    let chips = VirtualMachine::<NUM_WORDS, WORD_SIZE, _>::get_chips(&chips);

    let dummy_vm = VirtualMachine::<NUM_WORDS, WORD_SIZE, _>::new(vm_config, fib_program, vec![]);
    let rec_raps = get_rec_raps::<NUM_WORDS, WORD_SIZE, InnerConfig>(&dummy_vm.segments[0]);
    let rec_raps: Vec<_> = rec_raps.iter().map(|x| x.deref()).collect();

    assert_eq!(chips.len(), rec_raps.len());

    let pvs = pis;
    let (chips, rec_raps, traces, pvs) = sort_chips(chips, rec_raps, traces, pvs);

    run_recursive_test_benchmark(
        chips,
        rec_raps,
        traces,
        pvs,
        "VM Verifier of VM Fibonacci Program",
    )
}
