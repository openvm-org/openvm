use std::{cmp::Reverse, time::Instant};

use afs_compiler::{asm::AsmBuilder, ir::Felt};
use afs_recursion::{
    halo2::testing_utils::run_evm_verifier_e2e_test,
    testing_utils::{
        gen_vm_program_stark_for_test, inner::build_verification_program, StarkForTest,
    },
};
use ax_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::fri_params_with_80_bits_of_security,
    },
    engine::StarkFriEngine,
};
use itertools::{izip, multiunzip, Itertools};
use metrics::gauge;
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::{debugging::DebuggingRecorder, layers::Layer};
use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_matrix::Matrix;
use p3_uni_stark::{Domain, StarkGenericConfig};
use stark_vm::{
    program::Program,
    vm::{config::VmConfig, segment::SegmentResult, VirtualMachine},
};
use tracing::{info_span, Level};
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

fn calc_fibonacci(a: u32, b: u32, n: u32) -> u32 {
    let mut prev = a;
    let mut next = b;

    for _ in 2..n {
        let tmp = next;
        next += prev;
        prev = tmp;
    }

    next
}

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

    let start = Instant::now();
    let mut vm_config = VmConfig::core();
    vm_config.field_arithmetic_enabled = true;
    vm_config.num_public_values = 3;

    let vm = VirtualMachine::new(vm_config, fib_program, vec![]);
    vm.segments[0].cpu_chip.borrow_mut().public_values = vec![
        Some(BabyBear::zero()),
        Some(BabyBear::one()),
        Some(BabyBear::from_canonical_u32(calc_fibonacci(a, b, n))),
    ];

    let result = vm.execute_and_generate().unwrap();
    assert_eq!(result.segment_results.len(), 1, "unexpected continuation");
    let SegmentResult {
        airs,
        traces,
        public_values,
        ..
    } = result.segment_results.into_iter().next().unwrap();
    gauge!("trace_generation_time_ms", "program" => "fibonacci_program")
        .set(start.elapsed().as_millis() as f64);

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

fn main() {
    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();
    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new());
    // Prepare tracing.
    tracing::subscriber::set_global_default(subscriber).unwrap();

    // Prepare metrics.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let recorder = TracingContextLayer::all().layer(recorder);
    // Install the registry as the global recorder
    metrics::set_global_recorder(recorder).unwrap();

    let span = info_span!("Fibonacci Program Inner", group = "fibonacci_program_inner").entered();
    let fib_program_stark = fibonacci_program_stark_for_test(0, 1, 32);
    let StarkForTest {
        any_raps,
        traces,
        pvs,
    } = fib_program_stark;
    let any_raps: Vec<_> = any_raps.iter().map(|x| x.as_ref()).collect();
    let vdata =
        BabyBearPoseidon2Engine::run_simple_test_with_default_engine(&any_raps, traces, &pvs)
            .unwrap();
    span.exit();

    let span = info_span!("Recursive Verify e2e", group = "recursive_verify_e2e").entered();
    let (program, witness_stream) = build_verification_program(pvs, vdata);
    let inner_verifier_sft = gen_vm_program_stark_for_test(
        program,
        witness_stream,
        VmConfig {
            num_public_values: 4,
            ..Default::default()
        },
    );
    run_evm_verifier_e2e_test(
        &inner_verifier_sft,
        // log_blowup = 3 because of poseidon2 chip.
        Some(fri_params_with_80_bits_of_security()[1]),
    );
    span.exit();

    let snapshot = snapshotter.snapshot();
    println!("{:?}", snapshot.into_hashmap())
}
