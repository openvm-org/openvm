use std::ops::Deref;

use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};
use afs_recursion::stark::sort_chips;
use color_eyre::eyre::Result;
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use stark_vm::vm::{config::VmConfig, segment::SegmentResult, VirtualMachine};

use crate::commands::vm::benchmark_helpers::run_recursive_test_benchmark;

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
        builder.assign(&temp, b);
        builder.assign(&b, a + b);
        builder.assign(&a, temp);
    });

    builder.halt();

    let fib_program = builder.compile_isa();

    let vm_config = VmConfig {
        max_segment_len: 2000000,
        ..Default::default()
    };

    let vm = VirtualMachine::new(vm_config, fib_program.clone(), vec![]);

    let result = vm.execute_and_generate()?;

    assert_eq!(
        result.segment_results.len(),
        1,
        "continuations not yet supported"
    );
    let result = result.segment_results.into_iter().next().unwrap();

    let SegmentResult {
        airs,
        traces,
        public_values,
        ..
    } = result;

    let airs = airs.iter().map(Box::deref).collect_vec();

    let (chips, traces, pvs) = sort_chips(airs, traces, public_values);

    run_recursive_test_benchmark(chips, traces, pvs, "VM Verifier of VM Fibonacci Program")
}
