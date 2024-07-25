use afs_compiler::{
    asm::AsmBuilder,
    ir::{Felt, Var},
    util::{display_program, execute_and_prove_program},
};
use afs_recursion::{
    hints::Hintable,
    stark::{DynRapForRecursion, VerifierProgram},
    types::{new_from_multi_vk, InnerConfig, VerifierProgramInput},
};
use afs_stark_backend::{
    prover::trace::TraceCommitmentBuilder, rap::AnyRap, verifier::MultiTraceStarkVerifier,
};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config},
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

    run_recursive_test(vec![&fib_air], vec![&fib_air], vec![trace], pvs)
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
    display_program(&program);
    execute_and_prove_program::<1>(program, vec![]);
}

fn run_recursive_test(
    // TODO: find way to not duplicate parameters
    any_raps: Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    rec_raps: Vec<&dyn DynRapForRecursion<InnerConfig>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: Vec<Vec<BabyBear>>,
) {
    let num_pvs: Vec<usize> = pvs.iter().map(|pv| pv.len()).collect();

    let trace_heights: Vec<usize> = traces.iter().map(|t| t.height()).collect();
    let log_degree = log2_strict_usize(trace_heights.clone().into_iter().max().unwrap());

    let engine = default_engine(log_degree);

    let mut keygen_builder = engine.keygen_builder();
    for (&rap, &num_pv) in any_raps.iter().zip(num_pvs.iter()) {
        keygen_builder.add_air(rap, num_pv);
    }

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    // put into span; span for keygen, trace generation, time from starting trace gen to proof finishes
    for trace in traces.clone() {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, any_raps.clone());

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pvs);
    let log_degree_per_air = proof
        .degrees
        .iter()
        .map(|degree| log2_strict_usize(*degree))
        .collect();
    // Make sure proof verifies outside eDSL...
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(&mut engine.new_challenger(), &vk, any_raps, &proof, &pvs)
        .expect("afs proof should verify");

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

    execute_and_prove_program::<1>(program, witness_stream);
}
