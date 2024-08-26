use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_vm::vm::{config::VmConfig, VirtualMachine, VirtualMachineResult};

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

    let result: VirtualMachineResult<BabyBearPoseidon2Config> = vm.execute_and_generate()?;

    assert_eq!(result.segment_results.len(), 1, "continuations not yet supported");

    // FIXME[zach]: restore after removal of RecursiveVerifierConstraintFolder
    panic!("fib_verifier_program disabled");

    // let (chips, rec_raps, traces, pvs) = sort_chips(chips, rec_raps, traces, pvs);
    //
    // run_recursive_test_benchmark(
    //     chips,
    //     rec_raps,
    //     traces,
    //     pvs,
    //     "VM Verifier of VM Fibonacci Program",
    // )
}
