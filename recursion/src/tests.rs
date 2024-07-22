use std::sync::Arc;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::Val;
use p3_util::log2_strict_usize;

use afs_chips::range_gate::RangeCheckerGateChip;
use afs_chips::sum::SumChip;
use afs_compiler::util::execute_program;
use afs_stark_backend::interaction::AirBridge;
use afs_stark_backend::prover::trace::TraceCommitmentBuilder;
use afs_stark_backend::rap::AnyRap;
use afs_stark_backend::verifier::MultiTraceStarkVerifier;
use afs_test_utils::config::baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config};
use afs_test_utils::config::setup_tracing;
use afs_test_utils::engine::StarkEngine;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::utils::to_field_vec;

use crate::hints::Hintable;
use crate::stark::{AxiomVerifier, DynRapForRecursion};
use crate::types::{AxiomMemoryLayout, InnerConfig, MultiStarkVerificationAdvice};

pub struct FibonacciAir;

impl<F: Field> AirBridge<F> for FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        2
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let mut when_first_row = builder.when_first_row();
        when_first_row.assert_eq(local[0], a);
        when_first_row.assert_eq(local[1], b);

        let mut when_transition = builder.when_transition();
        when_transition.assert_eq(next[0], local[1]);
        when_transition.assert_eq(next[1], local[0] + local[1]);

        builder.when_last_row().assert_eq(local[1], x);
    }
}

pub fn generate_trace_rows<F: PrimeField32>(n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut rows = vec![vec![F::zero(), F::one()]];

    for i in 1..n {
        rows.push(vec![rows[i - 1][1], rows[i - 1][0] + rows[i - 1][1]]);
    }

    RowMajorMatrix::new(rows.concat(), 2)
}

#[test]
fn test_fibonacci() {
    type SC = BabyBearPoseidon2Config;
    type F = Val<SC>;

    setup_tracing();

    let fib_air = FibonacciAir {};
    let n = 16;
    let trace = generate_trace_rows(n);
    let pvs = vec![vec![
        F::from_canonical_u32(0),
        F::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ]];

    run_recursive_test(vec![&fib_air], vec![&fib_air], vec![trace], pvs)
}

#[test]
fn test_interactions() {
    type SC = BabyBearPoseidon2Config;

    const INPUT_BUS: usize = 0;
    const OUTPUT_BUS: usize = 1;
    const RANGE_BUS: usize = 2;
    const RANGE_MAX: u32 = 16;

    setup_tracing();

    let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_BUS, RANGE_MAX));
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

    let any_raps: Vec<&dyn AnyRap<SC>> = vec![
        &sum_chip.air,
        &sender_air,
        &receiver_air,
        &sum_chip.range_checker.air,
    ];
    let rec_raps: Vec<&dyn DynRapForRecursion<InnerConfig>> = vec![
        &sum_chip.air,
        &sender_air,
        &receiver_air,
        &sum_chip.range_checker.air,
    ];
    let traces = vec![sum_trace, sender_trace, receiver_trace, range_checker_trace];
    let pvs = vec![vec![], vec![], vec![], vec![]];

    run_recursive_test(any_raps, rec_raps, traces, pvs)
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

    let partial_pk = keygen_builder.generate_partial_pk();
    let partial_vk = partial_pk.partial_vk();

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces.clone() {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&partial_vk, any_raps.clone());

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &partial_pk, main_trace_data, &pvs);
    let log_degree_per_air = proof
        .degrees
        .iter()
        .map(|degree| log2_strict_usize(*degree))
        .collect();
    // Make sure proof verifies outside eDSL...
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(
            &mut engine.new_challenger(),
            &partial_vk,
            any_raps,
            &proof,
            &pvs,
        )
        .expect("afs proof should verify");

    // Build verification program in eDSL.
    let advice = MultiStarkVerificationAdvice::new_from_multi_vk(&partial_vk);

    let program = AxiomVerifier::build(rec_raps, advice, &engine.fri_params);

    let input = AxiomMemoryLayout {
        proof,
        log_degree_per_air,
        public_values: pvs.clone(),
    };

    let mut witness_stream = Vec::new();
    witness_stream.extend(input.write());

    execute_program::<1, _>(program, witness_stream);
}
