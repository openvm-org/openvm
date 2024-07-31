use afs_recursion::{
    hints::Hintable,
    stark::{DynRapForRecursion, VerifierProgram},
    types::{new_from_multi_vk, InnerConfig, VerifierInput},
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
use p3_baby_bear::BabyBear;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use stark_vm::{
    cpu::trace::Instruction,
    vm::{config::VmConfig, ExecutionResult, VirtualMachine},
};
use tracing::info_span;

pub fn run_recursive_test_benchmark(
    // TODO: find way to not duplicate parameters
    any_raps: Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    rec_raps: Vec<&dyn DynRapForRecursion<InnerConfig>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: Vec<Vec<BabyBear>>,
) {
    let num_pvs: Vec<usize> = pvs.iter().map(|pv| pv.len()).collect();

    let trace_heights: Vec<usize> = traces.iter().map(|t| t.height()).collect();

    let log_degree = log2_strict_usize(trace_heights.clone().into_iter().max().unwrap());

    // FRI params to prove `any_raps` with
    // log_blowup_factor = 1
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[2]
    } else {
        fri_params_with_80_bits_of_security()[2]
    };
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
    let trace_and_prove_span =
        info_span!("Benchmark trace commitment and prove before recursion").entered();

    // span for trace generation
    let trace_commitment_span = info_span!("Benchmark trace commitment").entered();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces.clone() {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();
    trace_commitment_span.exit();

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

    let input = VerifierInput {
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
    let vm_config = VmConfig {
        max_segment_len: 1 << 25, // turn off segmentation
        ..Default::default()
    };

    let vm = VirtualMachine::<WORD_SIZE, _>::new(vm_config, program, input_stream);

    let vm_execute_span = info_span!("Benchmark vm execute").entered();
    let ExecutionResult {
        max_log_degree,
        nonempty_chips: chips,
        nonempty_traces: traces,
        nonempty_pis: public_values,
        ..
    } = vm.execute().unwrap();
    vm_execute_span.exit();

    let chips = VirtualMachine::<WORD_SIZE, _>::get_chips(&chips);

    let perm = default_perm();
    // blowup factor 8 for poseidon2 chip
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    setup_tracing();

    assert_eq!(chips.len(), traces.len());

    let keygen_span = info_span!("Benchmark keygen").entered();
    let mut keygen_builder = engine.keygen_builder();

    for i in 0..chips.len() {
        keygen_builder.add_air(chips[i], public_values[i].len());
    }

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();
    keygen_span.exit();

    let prover = engine.prover();

    let trace_commitment_span = info_span!("Benchmark trace commitment").entered();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();
    trace_commitment_span.exit();

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
