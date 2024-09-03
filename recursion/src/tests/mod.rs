use std::{rc::Rc, sync::Arc};

use afs_compiler::util::execute_and_prove_program;
use afs_primitives::{range::bus::RangeCheckBus, range_gate::RangeCheckerGateChip, sum::SumChip};
use afs_stark_backend::{
    prover::trace::TraceCommitmentBuilder, rap::AnyRap, verifier::MultiTraceStarkVerifier,
};
use ax_sdk::{
    config::{
        baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config},
        setup_tracing,
    },
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::{generate_fib_trace_rows, to_field_vec, FibonacciAir},
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_strict_usize;
use stark_vm::vm::config::VmConfig;

use crate::{
    hints::Hintable,
    stark::VerifierProgram,
    types::{new_from_inner_multi_vk, VerifierInput},
};

pub(crate) struct StarkForTest<SC: StarkGenericConfig> {
    pub any_raps: Vec<Rc<dyn AnyRap<SC>>>,
    pub traces: Vec<RowMajorMatrix<BabyBear>>,
    pub pvs: Vec<Vec<BabyBear>>,
}

pub(crate) fn fibonacci_stark_for_test<SC: StarkGenericConfig>() -> StarkForTest<SC> {
    setup_tracing();

    let fib_air = Rc::new(FibonacciAir {});
    let n = 16;
    let trace = generate_fib_trace_rows::<BabyBear>(n);
    let pvs = vec![vec![
        BabyBear::from_canonical_u32(0),
        BabyBear::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ]];
    StarkForTest {
        any_raps: vec![fib_air.clone()],
        traces: vec![trace],
        pvs,
    }
}

pub(crate) fn interaction_stark_for_test<SC: StarkGenericConfig>() -> StarkForTest<SC> {
    const INPUT_BUS: usize = 0;
    const OUTPUT_BUS: usize = 1;
    const RANGE_BUS: usize = 2;
    const RANGE_MAX: u32 = 16;

    let range_bus = RangeCheckBus::new(RANGE_BUS, RANGE_MAX);
    let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus));
    let sum_chip = SumChip::new(INPUT_BUS, OUTPUT_BUS, 4, 4, range_checker);

    let mut sum_trace_u32 = Vec::<(u32, u32, u32, u32)>::new();
    let n = 16;
    for i in 0..n {
        sum_trace_u32.push((0, 1, i + 1, (i == n - 1) as u32));
    }

    let kv: &[(u32, u32)] = &sum_trace_u32
        .iter()
        .map(|&(key, value, _, _)| (key, value))
        .collect::<Vec<_>>();
    let sum_trace = sum_chip.generate_trace(kv);
    let sender_air = DummyInteractionAir::new(2, true, INPUT_BUS);
    let sender_trace = RowMajorMatrix::new(
        to_field_vec(
            sum_trace_u32
                .iter()
                .flat_map(|&(key, val, _, _)| [1, key, val])
                .collect(),
        ),
        sender_air.field_width() + 1,
    );
    let receiver_air = DummyInteractionAir::new(2, false, OUTPUT_BUS);
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(
            sum_trace_u32
                .iter()
                .flat_map(|&(key, _, sum, is_final)| [is_final, key, sum])
                .collect(),
        ),
        receiver_air.field_width() + 1,
    );
    let range_checker_trace = sum_chip.range_checker.generate_trace();
    let sum_air = Rc::new(sum_chip.air);
    let sender_air = Rc::new(sender_air);
    let receiver_air = Rc::new(receiver_air);
    let range_checker_air = Rc::new(sum_chip.range_checker.air);

    let any_raps: Vec<Rc<dyn AnyRap<SC>>> =
        vec![sum_air, sender_air, receiver_air, range_checker_air];
    let traces = vec![sum_trace, sender_trace, receiver_trace, range_checker_trace];
    let pvs = vec![vec![], vec![], vec![], vec![]];

    StarkForTest {
        any_raps,
        traces,
        pvs,
    }
}

#[test]
fn test_fibonacci() {
    setup_tracing();

    run_recursive_test(&fibonacci_stark_for_test::<BabyBearPoseidon2Config>())
}

#[test]
fn test_interactions() {
    setup_tracing();

    run_recursive_test(&interaction_stark_for_test::<BabyBearPoseidon2Config>())
}

fn run_recursive_test(stark_for_test: &StarkForTest<BabyBearPoseidon2Config>) {
    let StarkForTest {
        any_raps,
        traces,
        pvs,
    } = stark_for_test;
    let any_raps: Vec<_> = any_raps.iter().map(|x| x.as_ref()).collect();

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
    for trace in traces.clone() {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, any_raps.clone());

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, pvs);
    let log_degree_per_air = proof
        .degrees
        .iter()
        .map(|degree| log2_strict_usize(*degree))
        .collect();
    // Make sure proof verifies outside eDSL...
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(&mut engine.new_challenger(), &vk, &proof, pvs)
        .expect("afs proof should verify");

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

    execute_and_prove_program(program, witness_stream, VmConfig::default());
}
