use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};
use color_eyre::eyre::Result;

use super::benchmark_helpers::run_recursive_test_benchmark;
use afs_recursion::{
    stark::{get_rec_raps, sort_chips},
    types::InnerConfig,
};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use stark_vm::vm::{config::VmConfig, ExecutionResult, VirtualMachine};

pub fn benchmark_fib_verifier_program(n: usize) -> Result<()> {
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
        builder.assign(temp, b);
        builder.assign(b, a + b);
        builder.assign(a, temp);
    });

    builder.halt();

    let fib_program = builder.compile_isa::<1>();

    let vm_config = VmConfig {
        max_segment_len: 2000000,
        ..Default::default()
    };

    let vm = VirtualMachine::<1, _>::new(vm_config, fib_program.clone(), vec![]);

    let ExecutionResult {
        max_log_degree: _,
        nonempty_chips: chips,
        nonempty_traces: traces,
        nonempty_pis: pis,
        ..
    } = vm.execute().unwrap();
    let chips = VirtualMachine::<1, _>::get_chips(&chips);

    let dummy_vm = VirtualMachine::<1, _>::new(vm_config, fib_program.clone(), vec![]);
    let rec_raps = get_rec_raps::<1, InnerConfig>(&dummy_vm.segments[0]);

    assert!(chips.len() == rec_raps.len());

    let pvs = pis;
    let (chips, rec_raps, traces, pvs) = sort_chips(chips, rec_raps, traces, pvs);

    run_recursive_test_benchmark(chips, rec_raps, traces, pvs)
}
