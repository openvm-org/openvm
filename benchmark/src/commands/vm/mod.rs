use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
};

use afs_recursion::{
    hints::{Hintable, InnerVal},
    stark::{DynRapForRecursion, VerifierProgram},
    types::{new_from_multi_vk, InnerConfig, VerifierProgramInput},
};
use afs_stark_backend::{
    prover::trace::TraceCommitmentBuilder, rap::AnyRap, verifier::MultiTraceStarkVerifier,
};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{default_perm, engine_from_perm, BabyBearPoseidon2Config},
        fri_params::{fri_params_fast_testing, fri_params_with_80_bits_of_security},
        setup_tracing,
    },
    engine::StarkEngine,
};
use clap::Parser;
use fibonacci::{generate_trace_rows, FibonacciAir};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::Val;
use p3_util::log2_strict_usize;
use stark_vm::{
    cpu::trace::Instruction,
    vm::{config::VmConfig, get_chips, VirtualMachine},
};
use tracing::info_span;

use super::CommonCommands;

mod fibonacci;

pub struct VmBenchmarkConfig {
    pub n: usize,
}

#[derive(Debug, Parser)]
pub struct VmCommand {
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

fn calc_fibonacci(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        let mut a = 0;
        let mut b = 1;
        for _ in 0..n {
            let temp = b;
            b += a;
            a = temp;
        }
        a
    }
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

pub fn get_rec_raps<const WORD_SIZE: usize>(
    vm: &VirtualMachine<WORD_SIZE, InnerVal>,
) -> Vec<&dyn DynRapForRecursion<InnerConfig>> {
    let mut result: Vec<&dyn DynRapForRecursion<InnerConfig>> = vec![
        &vm.cpu_air,
        &vm.program_chip.air,
        &vm.memory_chip.air,
        &vm.range_checker.air,
    ];
    if vm.options().field_arithmetic_enabled {
        result.push(&vm.field_arithmetic_chip.air);
    }
    if vm.options().field_extension_enabled {
        result.push(&vm.field_extension_chip.air);
    }
    if vm.options().poseidon2_enabled() {
        result.push(&vm.poseidon2_chip.air);
    }
    result
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

    let expected_value = F::from_canonical_usize(calc_fibonacci(n));
    builder.assert_felt_eq(a, expected_value);

    //builder.print_f(a);
    builder.halt();

    let program = builder.compile_isa::<1>();

    // vm_benchmark_execute_and_prove::<1>(program.clone(), vec![]);

    let mut vm = VirtualMachine::<1, _>::new(
        VmConfig {
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
            limb_bits: 28,
            decomp: 4,
            compress_poseidon2_enabled: true,
            perm_poseidon2_enabled: true,
        },
        program.clone(),
        vec![],
    );

    let traces = vm.traces().unwrap();
    let chips = get_chips(&vm);

    let rec_raps = get_rec_raps(&vm);

    assert!(chips.len() == rec_raps.len());
    let len = chips.len();

    run_recursive_test_benchmark(chips, rec_raps, traces, vec![vec![]; len]);

    // run_recursive_test(chips, rec_raps, traces, vec![vec![]; len]);
}

fn run_recursive_test_benchmark(
    // TODO: find way to not duplicate parameters
    any_raps: Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    rec_raps: Vec<&dyn DynRapForRecursion<InnerConfig>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: Vec<Vec<BabyBear>>,
) {
    let num_pvs: Vec<usize> = pvs.iter().map(|pv| pv.len()).collect();

    let trace_heights: Vec<usize> = traces.iter().map(|t| t.height()).collect();

    let log_degree = log2_strict_usize(trace_heights.clone().into_iter().max().unwrap());

    let fri_params = fri_params_fast_testing()[0];
    let perm = default_perm();
    let engine = engine_from_perm(perm, log_degree, fri_params);

    let mut keygen_builder = engine.keygen_builder();
    for (&rap, &num_pv) in any_raps.iter().zip(num_pvs.iter()) {
        keygen_builder.add_air(rap, num_pv);
    }

    // keygen span
    let keygen_span = info_span!("Benchmark keygen").entered();
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();
    keygen_span.exit();

    let prover = engine.prover();

    // span for starting trace geneartion to proof finishes outside of eDSL
    let trace_and_prove_span = info_span!("Benchmark trace and prove outside of eDSL").entered();

    // span for trace generation
    let trace_span = info_span!("Benchmark trace generation").entered();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces.clone() {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();
    trace_span.exit();

    let main_trace_data = trace_builder.view(&vk, any_raps.clone());

    let mut challenger = engine.new_challenger();

    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pvs);

    // Make sure proof verifies outside eDSL...
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(
            &mut engine.new_challenger(),
            &vk,
            any_raps.clone(),
            &proof,
            &pvs,
        )
        .expect("afs proof should verify");
    trace_and_prove_span.exit();

    let log_degree_per_air = proof
        .degrees
        .iter()
        .map(|degree| log2_strict_usize(*degree))
        .collect();

    // Build verification program in eDSL.
    let advice = new_from_multi_vk(&vk);

    let program = VerifierProgram::build(rec_raps, advice, &engine.fri_params);

    let input = VerifierProgramInput {
        proof,
        log_degree_per_air,
        public_values: pvs.clone(),
    };

    let mut witness_stream = Vec::new();
    witness_stream.extend(input.write());

    vm_benchmark_execute_and_prove::<1>(program, witness_stream);
}

pub fn vm_benchmark_execute_and_prove<const WORD_SIZE: usize>(
    program: Vec<Instruction<BabyBear>>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
            limb_bits: 28,
            decomp: 4,
            compress_poseidon2_enabled: true,
            perm_poseidon2_enabled: true,
        },
        program,
        input_stream,
    );
    let max_log_degree = vm.max_log_degree().unwrap();

    let perm = default_perm();
    // blowup factor 8 for poseidon2 chip
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    let trace_span = info_span!("Benchmark trace generation").entered();
    let traces = vm.traces().unwrap();

    let chips = get_chips(&vm);

    assert_eq!(chips.len(), traces.len());

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    for trace in traces {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();
    trace_span.exit();

    let num_chips = chips.len();

    setup_tracing();

    let public_values = vec![vec![]; num_chips];

    let keygen_span = info_span!("Benchmark keygen").entered();
    let mut keygen_builder = engine.keygen_builder();

    for i in 0..chips.len() {
        keygen_builder.add_air(chips[i], public_values[i].len());
    }

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();
    keygen_span.exit();

    let main_trace_data = trace_builder.view(&vk, chips.to_vec());

    let mut challenger = engine.new_challenger();

    let prove_span = info_span!("Benchmark prove").entered();
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &public_values);
    prove_span.exit();

    let mut challenger = engine.new_challenger();

    let verify_span = info_span!("Benchmark verify").entered();
    let verifier = engine.verifier();
    verifier
        .verify(&mut challenger, &vk, chips, &proof, &public_values)
        .expect("Verification failed");
    verify_span.exit();
}
