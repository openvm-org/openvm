use std::ops::Deref;

use afs_compiler::{asm::AsmBuilder, ir::Felt};
use ax_sdk::config::{
    fri_params::{fri_params_fast_testing, fri_params_with_80_bits_of_security},
    setup_tracing,
};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use stark_vm::{
    program::Program,
    vm::{config::VmConfig, segment::SegmentResult, VirtualMachine},
};

mod common;

fn fibonacci_program(a: u32, b: u32, n: u32) -> Program<BabyBear> {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let prev: Felt<_> = builder.constant(F::from_canonical_u32(a));
    let next: Felt<_> = builder.constant(F::from_canonical_u32(b));

    builder.commit_public_value(prev);
    builder.commit_public_value(next);

    for _ in 2..n {
        let tmp: Felt<_> = builder.uninit();
        builder.assign(&tmp, next);
        builder.assign(&next, prev + next);
        builder.assign(&prev, tmp);
    }

    builder.commit_public_value(next);

    builder.halt();

    builder.compile_isa()
}

#[test]
fn test_fibonacci_program_verify() {
    setup_tracing();

    let fib_program = fibonacci_program(0, 1, 32);

    let vm_config = VmConfig {
        num_public_values: 3,
        ..Default::default()
    };

    let vm = VirtualMachine::new(vm_config, fib_program, vec![]);
    vm.segments[0].cpu_chip.borrow_mut().public_values = vec![
        Some(BabyBear::zero()),
        Some(BabyBear::one()),
        Some(BabyBear::from_canonical_u32(1346269)),
    ];

    let result = vm.execute_and_generate().unwrap();
    assert_eq!(result.segment_results.len(), 1, "unexpected continuation");
    let SegmentResult {
        airs,
        traces,
        public_values,
        ..
    } = result.segment_results.into_iter().next().unwrap();

    let airs = airs.iter().map(Box::deref).collect_vec();

    // blowup factor = 3
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };
    common::run_recursive_test(airs, traces, public_values, fri_params);
}
