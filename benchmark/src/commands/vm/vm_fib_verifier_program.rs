use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};

use super::benchmark_helpers::{get_rec_raps, run_recursive_test_benchmark};
use afs_recursion::common::sort_chips;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use stark_vm::vm::{config::VmConfig, get_chips, VirtualMachine};

pub fn benchmark_fib_verifier_program(n: usize) {
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
