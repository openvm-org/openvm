use divan::Bencher;
use openvm_benchmarks_execute::{
    executor, load_program_executable, metered_cost_setup, metering_setup, setup_internal_verifier,
    setup_leaf_verifier, transmute_interpreter_lifetime,
};
use openvm_native_circuit::NATIVE_MAX_TRACE_HEIGHTS;

const APP_PROGRAMS: &[&str] = &[
    "fibonacci_recursive",
    "fibonacci_iterative",
    "quicksort",
    "bubblesort",
    "factorial_iterative_u256",
    "revm_snailtracer",
    "keccak256",
    "keccak256_iter",
    "sha256",
    "sha256_iter",
    "revm_transfer",
    "pairing",
];
const LEAF_VERIFIER_PROGRAMS: &[&str] = &["kitchen-sink"];
const INTERNAL_VERIFIER_PROGRAMS: &[&str] = &["fibonacci"];

fn main() {
    divan::main();
}

#[divan::bench(args = APP_PROGRAMS, sample_count=10)]
fn benchmark_execute(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let interpreter = executor().instance(&exe).unwrap();
            (interpreter, vec![])
        })
        .bench_values(|(interpreter, input)| {
            interpreter
                .execute(input, None)
                .expect("Failed to execute program in interpreted mode");
        });
}

#[divan::bench(args = APP_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let (ctx, executor_idx_to_air_idx) = metering_setup();
            let interpreter = executor()
                .metered_instance(&exe, executor_idx_to_air_idx)
                .unwrap();
            (interpreter, vec![], ctx.clone())
        })
        .bench_values(|(interpreter, input, ctx)| {
            interpreter
                .execute_metered(input, ctx)
                .expect("Failed to execute program");
        });
}

#[divan::bench(ignore = true, args = APP_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered_cost(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let (ctx, executor_idx_to_air_idx) = metered_cost_setup();
            let interpreter = executor()
                .metered_cost_instance(&exe, executor_idx_to_air_idx)
                .unwrap();
            (interpreter, vec![], ctx.clone())
        })
        .bench_values(|(interpreter, input, ctx)| {
            interpreter
                .execute_metered_cost(input, ctx)
                .expect("Failed to execute program with metered cost");
        });
}

#[divan::bench(args = LEAF_VERIFIER_PROGRAMS, sample_count = 5)]
fn benchmark_leaf_verifier_execute(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let (vm, leaf_exe, input_stream) = setup_leaf_verifier(program);
            let interpreter = vm.executor().instance(&leaf_exe).unwrap();
            let interpreter = transmute_interpreter_lifetime(interpreter);

            (vm, interpreter, input_stream)
        })
        .bench_values(|(_vm, interpreter, input_stream)| {
            interpreter
                .execute(input_stream, None)
                .expect("Failed to execute program in interpreted mode");
        });
}

#[divan::bench(args = LEAF_VERIFIER_PROGRAMS, sample_count = 5)]
fn benchmark_leaf_verifier_execute_metered(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let (vm, leaf_exe, input_stream) = setup_leaf_verifier(program);
            let ctx = vm.build_metered_ctx();
            let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
            let interpreter = vm
                .executor()
                .metered_instance(&leaf_exe, &executor_idx_to_air_idx)
                .unwrap();
            let interpreter = transmute_interpreter_lifetime(interpreter);

            (vm, interpreter, input_stream, ctx)
        })
        .bench_values(|(_vm, interpreter, input_stream, ctx)| {
            interpreter
                .execute_metered(input_stream, ctx)
                .expect("Failed to execute program");
        });
}

#[divan::bench(args = LEAF_VERIFIER_PROGRAMS, sample_count = 5)]
fn benchmark_leaf_verifier_execute_preflight(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let (vm, leaf_exe, input_stream) = setup_leaf_verifier(program);
            let state = vm.create_initial_state(&leaf_exe, input_stream);
            let interpreter = vm.preflight_interpreter(&leaf_exe).unwrap();

            (vm, state, interpreter)
        })
        .bench_values(|(vm, state, mut interpreter)| {
            let _out = vm
                .execute_preflight(&mut interpreter, state, None, NATIVE_MAX_TRACE_HEIGHTS)
                .expect("Failed to execute preflight");
        });
}

#[divan::bench(args = INTERNAL_VERIFIER_PROGRAMS, sample_count = 5)]
fn benchmark_internal_verifier_execute(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let (vm, internal_exe, input_stream) = setup_internal_verifier(program);
            let interpreter = vm.executor().instance(&internal_exe).unwrap();
            let interpreter = transmute_interpreter_lifetime(interpreter);

            (vm, interpreter, input_stream)
        })
        .bench_values(|(_vm, interpreter, input_stream)| {
            interpreter
                .execute(input_stream, None)
                .expect("Failed to execute program in interpreted mode");
        });
}

#[divan::bench(args = INTERNAL_VERIFIER_PROGRAMS, sample_count = 5)]
fn benchmark_internal_verifier_execute_metered(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let (vm, internal_exe, input_stream) = setup_internal_verifier(program);
            let ctx = vm.build_metered_ctx();
            let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
            let interpreter = vm
                .executor()
                .metered_instance(&internal_exe, &executor_idx_to_air_idx)
                .unwrap();
            let interpreter = transmute_interpreter_lifetime(interpreter);

            (vm, interpreter, input_stream, ctx)
        })
        .bench_values(|(_vm, interpreter, input_stream, ctx)| {
            interpreter
                .execute_metered(input_stream, ctx)
                .expect("Failed to execute program");
        });
}

#[divan::bench(args = INTERNAL_VERIFIER_PROGRAMS, sample_count = 5)]
fn benchmark_internal_verifier_execute_preflight(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let (vm, internal_exe, input_stream) = setup_internal_verifier(program);
            let state = vm.create_initial_state(&internal_exe, input_stream);
            let interpreter = vm.preflight_interpreter(&internal_exe).unwrap();

            (vm, state, interpreter)
        })
        .bench_values(|(vm, state, mut interpreter)| {
            let _out = vm
                .execute_preflight(&mut interpreter, state, None, NATIVE_MAX_TRACE_HEIGHTS)
                .expect("Failed to execute preflight");
        });
}
