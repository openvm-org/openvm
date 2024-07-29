use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};

use afs_recursion::common::sort_chips;
use afs_test_utils::config::{baby_bear_poseidon2::BabyBearPoseidon2Config, setup_tracing};
use benchmark_helpers::{
    get_rec_raps, run_recursive_test_benchmark, vm_benchmark_execute_and_prove,
};
use clap::Parser;
use fibonacci::{generate_trace_rows, FibonacciAir};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_matrix::Matrix;
use p3_uni_stark::Val;
use stark_vm::vm::{config::VmConfig, get_chips, VirtualMachine};

use super::CommonCommands;

mod benchmark_helpers;
mod fibonacci;

pub struct VmBenchmarkConfig {
    pub n: usize,
}

#[derive(Debug, Parser)]
pub struct VmCommand {
    #[arg(
        long = "vm-benchmark-program",
        short = 't',
        help = "The benchmark to run: (1) fibonacci program, (2) verify fibair, (3) verifier program for fibonacci program",
        default_value = "1",
        required = true
    )]
    pub t: usize,

    #[arg(
        long = "n-value",
        short = 'n',
        help = "The value of n such that we are computing the n-th Fibonacci number",
        default_value = "2",
        required = true
    )]
    /// The value of n such that we are computing the n-th Fibonacci number
    pub n: usize,

    #[command(flatten)]
    pub common: CommonCommands,
}

pub fn benchmark_fibonacci_program(n: usize) {
    println!("Running Fibonacci program benchmark with n = {}", n);

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

    vm_benchmark_execute_and_prove::<1>(fib_program.clone(), vec![]);
}

pub fn benchmark_verify_fibair(n: usize) {
    println!("Running Verify Fibonacci Air benchmark with n = {}", n);

    type SC = BabyBearPoseidon2Config;
    type F = Val<SC>;

    setup_tracing();

    let fib_air = FibonacciAir {};
    let trace = generate_trace_rows(n);
    let pvs = vec![vec![
        F::from_canonical_u32(0),
        F::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ]];

    run_recursive_test_benchmark(vec![&fib_air], vec![&fib_air], vec![trace], pvs)
}

pub fn benchmark_fibonacci_verifier_program(n: usize) {
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

    let mut vm = VirtualMachine::<1, _>::new(
        VmConfig {
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
            limb_bits: 28,
            decomp: 4,
            compress_poseidon2_enabled: true,
            perm_poseidon2_enabled: true,
            num_public_values: 0,
        },
        fib_program.clone(),
        vec![],
    );

    let traces = vm.traces().unwrap();
    let chips = get_chips(&vm);
    let rec_raps = get_rec_raps(&vm);

    assert!(chips.len() == rec_raps.len());
    let len = chips.len();

    let pvs = vec![vec![]; len];
    let (chips, rec_raps, traces, pvs) = sort_chips(chips, rec_raps, traces, pvs);

    run_recursive_test_benchmark(chips, rec_raps, traces, pvs);
}
