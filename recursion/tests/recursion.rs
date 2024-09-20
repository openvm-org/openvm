use std::cmp::Reverse;

use afs_compiler::{asm::AsmBuilder, ir::Felt};
use afs_recursion::testing_utils::{inner::run_recursive_test, StarkForTest};
use ax_sdk::config::setup_tracing;
use itertools::{izip, multiunzip, Itertools};
use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_matrix::Matrix;
use p3_uni_stark::{Domain, StarkGenericConfig};
use stark_vm::{
    program::Program,
    vm::{config::VmConfig, segment::SegmentResult, VirtualMachine},
};

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

pub(crate) fn fibonacci_program_stark_for_test<SC: StarkGenericConfig>(
    a: u32,
    b: u32,
    n: u32,
) -> StarkForTest<SC>
where
    Domain<SC>: PolynomialSpace<Val = BabyBear>,
{
    let fib_program = fibonacci_program(a, b, n);

    let vm_config = VmConfig {
        num_public_values: 3,
        ..Default::default()
    };

    let vm = VirtualMachine::new(vm_config, fib_program, vec![]);
    vm.segments[0].core_chip.borrow_mut().public_values = vec![
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

    let mut groups = izip!(airs, traces, public_values).collect_vec();
    groups.sort_by_key(|(_, trace, _)| Reverse(trace.height()));
    let (airs, traces, pvs): (Vec<_>, _, _) = multiunzip(groups);
    let airs = airs.into_iter().map(|x| x.into()).collect_vec();
    StarkForTest {
        any_raps: airs,
        traces,
        pvs,
    }
}

#[test]
fn test_fibonacci_program_verify() {
    setup_tracing();

    let fib_program_stark = fibonacci_program_stark_for_test(0, 1, 32);
    run_recursive_test(fib_program_stark);
}

#[cfg(feature = "static-verifier")]
#[test]
fn test_fibonacci_program_halo2_verify() {
    use afs_recursion::halo2::testing_utils::run_static_verifier_test;
    use ax_sdk::config::fri_params::default_fri_params;
    setup_tracing();

    let fib_program_stark = fibonacci_program_stark_for_test(0, 1, 32);
    run_static_verifier_test(&fib_program_stark, default_fri_params());
}
