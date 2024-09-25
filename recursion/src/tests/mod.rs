use std::{rc::Rc, sync::Arc};

use afs_primitives::{
    sum::SumChip,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::rap::AnyRap;
use ax_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config, fri_params::default_fri_params,
        setup_tracing, FriParameters,
    },
    engine::StarkForTest,
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::{generate_fib_trace_rows, to_field_vec, FibonacciAir},
};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::testing_utils::inner::run_recursive_test;

pub fn fibonacci_stark_for_test<SC: StarkGenericConfig>(n: usize) -> StarkForTest<SC>
where
    Val<SC>: PrimeField32,
{
    setup_tracing();

    let fib_air = Rc::new(FibonacciAir {});
    let trace = generate_fib_trace_rows::<Val<SC>>(n);
    let pvs = vec![vec![
        Val::<SC>::from_canonical_u32(0),
        Val::<SC>::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ]];
    StarkForTest {
        any_raps: vec![fib_air.clone()],
        traces: vec![trace],
        pvs,
    }
}

pub fn interaction_stark_for_test<SC: StarkGenericConfig>() -> StarkForTest<SC>
where
    Val<SC>: PrimeField32,
{
    const INPUT_BUS: usize = 0;
    const OUTPUT_BUS: usize = 1;
    const RANGE_BUS: usize = 2;
    const RANGE_MAX_BITS: usize = 4;

    let range_bus = VariableRangeCheckerBus::new(RANGE_BUS, RANGE_MAX_BITS);
    let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
    let sum_chip = SumChip::new(INPUT_BUS, OUTPUT_BUS, 4, range_checker);

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
        vec![range_checker_air, sum_air, sender_air, receiver_air];
    let traces = vec![range_checker_trace, sum_trace, sender_trace, receiver_trace];
    let pvs = vec![vec![], vec![], vec![], vec![]];

    StarkForTest {
        any_raps,
        traces,
        pvs,
    }
}

pub fn unordered_stark_for_test<SC: StarkGenericConfig>() -> StarkForTest<SC>
where
    Val<SC>: PrimeField32,
{
    const BUS: usize = 0;
    const SENDER_HEIGHT: usize = 2;
    const RECEIVER_HEIGHT: usize = 4;
    let sender_air = DummyInteractionAir::new(1, true, BUS);
    let sender_trace = RowMajorMatrix::new(
        to_field_vec([[2, 1]; SENDER_HEIGHT].into_iter().flatten().collect()),
        sender_air.field_width() + 1,
    );
    let receiver_air = DummyInteractionAir::new(1, false, BUS);
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec([[1, 1]; RECEIVER_HEIGHT].into_iter().flatten().collect()),
        receiver_air.field_width() + 1,
    );
    let sender_air = Rc::new(sender_air);
    let receiver_air = Rc::new(receiver_air);

    let any_raps: Vec<Rc<dyn AnyRap<SC>>> = vec![sender_air, receiver_air];
    let traces = vec![sender_trace, receiver_trace];
    let pvs = vec![vec![]; 2];

    StarkForTest {
        any_raps,
        traces,
        pvs,
    }
}

#[test]
fn test_fibonacci_small() {
    setup_tracing();

    run_recursive_test(
        fibonacci_stark_for_test::<BabyBearPoseidon2Config>(1 << 5),
        default_fri_params(),
    )
}

#[test]
fn test_fibonacci_small() {
    setup_tracing();

    run_recursive_test(
        fibonacci_stark_for_test::<BabyBearPoseidon2Config>(1 << 5),
        default_fri_params(),
    )
}

#[test]
fn test_fibonacci() {
    setup_tracing();

    // test lde = 27
    run_recursive_test(
        fibonacci_stark_for_test::<BabyBearPoseidon2Config>(1 << 24),
        FriParameters {
            log_blowup: 3,
            num_queries: 2,
            proof_of_work_bits: 0,
        },
    )
}

#[test]
fn test_interactions() {
    setup_tracing();

    run_recursive_test(
        interaction_stark_for_test::<BabyBearPoseidon2Config>(),
        default_fri_params(),
    )
}

#[test]
fn test_unordered() {
    setup_tracing();

    run_recursive_test(
        unordered_stark_for_test::<BabyBearPoseidon2Config>(),
        default_fri_params(),
    )
}
