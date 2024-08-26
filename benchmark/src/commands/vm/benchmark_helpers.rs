use std::{collections::HashMap, fs::File, io::Write as _};

use afs_recursion::{
    hints::Hintable,
    stark::VerifierProgram,
    types::{new_from_inner_multi_vk, VerifierInput},
};
use afs_stark_backend::{
    prover::{metrics::trace_metrics, trace::TraceCommitmentBuilder},
    rap::AnyRap,
    verifier::MultiTraceStarkVerifier,
};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{default_perm, engine_from_perm, BabyBearPoseidon2Config},
        fri_params::{fri_params_fast_testing, fri_params_with_80_bits_of_security},
    },
    engine::StarkEngine,
};
use color_eyre::eyre;
use p3_baby_bear::BabyBear;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use stark_vm::{
    program::Program,
    vm::{config::VmConfig, ExecutionAndTraceGenerationResult, VirtualMachine},
};
use tracing::info_span;

use crate::{
    config::benchmark_data::{BenchmarkSetup, BACKEND_TIMING_FILTERS, BACKEND_TIMING_HEADERS},
    utils::tracing::{clear_tracing_log, extract_timing_data_from_log, setup_benchmark_tracing},
    workflow::metrics::BenchmarkMetrics,
    TMP_RESULT_MD, TMP_TRACING_LOG,
};

pub fn run_recursive_test_benchmark(
    any_raps: Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: Vec<Vec<BabyBear>>,
    benchmark_name: &str,
) -> eyre::Result<()> {
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
        .verify(&mut engine.new_challenger(), &vk, &proof, &pvs)
        .expect("proof should verify");
    trace_and_prove_span.exit();

    let log_degree_per_air = proof
        .degrees
        .iter()
        .map(|degree| log2_strict_usize(*degree))
        .collect();

    // Build verification program in eDSL.
    let advice = new_from_inner_multi_vk(&vk);

    let program = VerifierProgram::build(advice, &engine.fri_params);

    let input = VerifierInput {
        proof,
        log_degree_per_air,
        public_values: pvs.clone(),
    };

    let mut witness_stream = Vec::new();
    witness_stream.extend(input.write());

    vm_benchmark_execute_and_prove::<8, 1>(program, witness_stream, benchmark_name)
}

pub fn vm_benchmark_execute_and_prove<const NUM_WORDS: usize, const WORD_SIZE: usize>(
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
    benchmark_name: &str,
) -> eyre::Result<()> {
    clear_tracing_log(TMP_TRACING_LOG.as_str())?;
    setup_benchmark_tracing();
    let vm_config = VmConfig {
        max_segment_len: 1 << 25, // turn off segmentation
        ..Default::default()
    };

    let mut vm = VirtualMachine::<NUM_WORDS, WORD_SIZE, _>::new(vm_config, program, input_stream);
    vm.enable_metrics_collection();

    let vm_execute_span = info_span!("Benchmark vm execute").entered();
    let ExecutionAndTraceGenerationResult {
        max_log_degree,
        nonempty_chips: chips,
        nonempty_traces: traces,
        nonempty_pis: public_values,
        metrics: mut vm_metrics,
        ..
    } = vm.execute_and_generate_traces().unwrap();
    vm_execute_span.exit();

    let chips = VirtualMachine::<NUM_WORDS, WORD_SIZE, _>::get_chips(&chips);

    let perm = default_perm();
    // blowup factor 8 for poseidon2 chip
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

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
    let prove_span = info_span!("Benchmark prove").entered(); // prove includes trace generation
    let trace_commitment_span = info_span!("Benchmark trace commitment").entered();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();
    trace_commitment_span.exit();

    let main_trace_data = trace_builder.view(&vk, chips.to_vec());

    let mut challenger = engine.new_challenger();

    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &public_values);
    prove_span.exit();

    let mut challenger = engine.new_challenger();

    let verify_span = info_span!("Benchmark verify").entered();
    let verifier = engine.verifier();
    verifier
        .verify(&mut challenger, &vk, &proof, &public_values)
        .expect("Verification failed");
    verify_span.exit();

    let setup = benchmark_setup_vm();
    let timing_data =
        extract_timing_data_from_log(TMP_TRACING_LOG.as_str(), setup.timing_filters.clone())?;
    let timing_data: HashMap<String, f64> = timing_data
        .into_iter()
        .map(|(k, v)| (k, v.parse::<f64>().unwrap()))
        .collect();
    let trace_metrics = trace_metrics(&pk.per_air, &proof.degrees);
    let main_trace_gen_ms = timing_data["Benchmark vm execute: benchmark"];
    let perm_trace_gen_ms = timing_data[BACKEND_TIMING_FILTERS[0]];
    let calc_quotient_values_ms = timing_data[BACKEND_TIMING_FILTERS[2]];
    let total_prove_ms = timing_data["Benchmark prove: benchmark"];
    let vm_metrics = vm_metrics.pop().unwrap(); // only 1 segment

    let metrics = BenchmarkMetrics {
        name: benchmark_name.to_string(),
        total_prove_ms,
        main_trace_gen_ms,
        perm_trace_gen_ms,
        calc_quotient_values_ms,
        trace: trace_metrics,
        custom: vm_metrics,
    };

    write!(File::create(TMP_RESULT_MD.as_str())?, "{}", metrics)?;
    Ok(())
}

pub fn benchmark_setup_vm() -> BenchmarkSetup {
    BenchmarkSetup {
        event_section: "air width".to_string(),
        event_headers: ["preprocessed", "main", "challenge"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        event_filters: [
            "Total air width: preprocessed=",
            "Total air width: partitioned_main=",
            "Total air width: after_challenge=",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        timing_section: "timing (ms)".to_string(),
        timing_headers: ["Keygen time", "Main trace generation", "Main trace commit"]
            .iter()
            .chain(BACKEND_TIMING_HEADERS)
            .chain(&["Prove time (total)", "Verify time"])
            .map(|s| s.to_string())
            .collect(),
        timing_filters: [
            "Benchmark keygen: benchmark",
            "Benchmark vm execute: benchmark",
            "Benchmark trace commitment: benchmark",
        ]
        .iter()
        .chain(BACKEND_TIMING_FILTERS)
        .chain(&["Benchmark prove: benchmark", "Benchmark verify: benchmark"])
        .map(|s| s.to_string())
        .collect(),
    }
}
