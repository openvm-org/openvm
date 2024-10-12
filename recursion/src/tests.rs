use std::{panic::catch_unwind, sync::Arc};

use afs_primitives::{
    sum::SumChip,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::utils::AirInfo;
use ax_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        setup_tracing, FriParameters,
    },
    dummy_airs::{
        fib_air::chip::FibonacciChip,
        interaction::dummy_interaction_air::{
            DummyInteractionAir, DummyInteractionChip, DummyInteractionData,
        },
    },
    engine::{StarkForTest, StarkFriEngine},
    utils::{generate_fib_trace_rows, to_field_vec, FibonacciAir},
};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use stark_vm::{sdk::gen_vm_program_stark_for_test, vm::config::VmConfig};

use crate::{
    hints::Hintable, stark::VerifierProgram, testing_utils::inner::run_recursive_test,
    types::new_from_inner_multi_vk,
};

pub fn fibonacci_stark_for_test<SC: StarkGenericConfig>(n: usize) -> StarkForTest<SC>
where
    Val<SC>: PrimeField32,
{
    setup_tracing();

    let fib_air = Box::new(FibonacciAir {});
    let trace = generate_fib_trace_rows::<Val<SC>>(n);
    let pvs = vec![
        Val::<SC>::from_canonical_u32(0),
        Val::<SC>::from_canonical_u32(1),
        trace.get(n - 1, 1),
    ];
    StarkForTest {
        air_infos: vec![AirInfo::simple(fib_air, trace, pvs)],
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
    let sum_air = Box::new(sum_chip.air);
    let sender_air = Box::new(sender_air);
    let receiver_air = Box::new(receiver_air);
    let range_checker_air = Box::new(sum_chip.range_checker.air);

    let range_checker_air_info = AirInfo::simple_no_pis(range_checker_air, range_checker_trace);
    let sum_air_info = AirInfo::simple_no_pis(sum_air, sum_trace);
    let sender_air_info = AirInfo::simple_no_pis(sender_air, sender_trace);
    let receiver_air_info = AirInfo::simple_no_pis(receiver_air, receiver_trace);

    StarkForTest {
        air_infos: vec![
            range_checker_air_info,
            sum_air_info,
            sender_air_info,
            receiver_air_info,
        ],
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

    let sender_air_info = AirInfo::simple_no_pis(Box::new(sender_air), sender_trace);
    let receiver_air_info = AirInfo::simple_no_pis(Box::new(receiver_air), receiver_trace);

    StarkForTest {
        air_infos: vec![sender_air_info, receiver_air_info],
    }
}

#[test]
fn test_fibonacci_small() {
    setup_tracing();

    run_recursive_test(
        fibonacci_stark_for_test::<BabyBearPoseidon2Config>(1 << 5),
        standard_fri_params_with_100_bits_conjectured_security(3),
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
        standard_fri_params_with_100_bits_conjectured_security(3),
    )
}

#[test]
fn test_unordered() {
    setup_tracing();

    run_recursive_test(
        unordered_stark_for_test::<BabyBearPoseidon2Config>(),
        standard_fri_params_with_100_bits_conjectured_security(3),
    )
}

#[test]
fn test_optional_air() {
    use afs_stark_backend::{
        engine::StarkEngine,
        prover::types::{Chip, ProofInput},
    };
    setup_tracing();

    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3);
    let engine = BabyBearPoseidon2Engine::new(fri_params);
    let fib_chip = FibonacciChip::new(0, 1, 8);
    let mut send_chip1 = DummyInteractionChip::new_without_partition(1, true, 0);
    let mut send_chip2 =
        DummyInteractionChip::new_with_partition(engine.config().pcs(), 1, true, 0);
    let mut recv_chip1 = DummyInteractionChip::new_without_partition(1, false, 0);
    let mut keygen_builder = engine.keygen_builder();
    let fib_chip_id = keygen_builder.add_air(fib_chip.air());
    let send_chip1_id = keygen_builder.add_air(send_chip1.air());
    let send_chip2_id = keygen_builder.add_air(send_chip2.air());
    let recv_chip1_id = keygen_builder.add_air(recv_chip1.air());
    let pk = keygen_builder.generate_pk();
    let prover = engine.prover();
    let verifier = engine.verifier();

    let m_advice = new_from_inner_multi_vk(&pk.get_vk());
    let vm_config = VmConfig::aggregation(7);
    let program = VerifierProgram::build(m_advice, &fri_params);

    // Case 1: All AIRs are present.
    {
        let mut challenger = engine.new_challenger();
        send_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        send_chip2.load_data(DummyInteractionData {
            count: vec![1, 2, 8],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        recv_chip1.load_data(DummyInteractionData {
            count: vec![2, 4, 12],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let proof = prover.prove(
            &mut challenger,
            &pk,
            ProofInput {
                per_air: vec![
                    fib_chip.generate_air_proof_input_with_id(fib_chip_id),
                    send_chip1.generate_air_proof_input_with_id(send_chip1_id),
                    send_chip2.generate_air_proof_input_with_id(send_chip2_id),
                    recv_chip1.generate_air_proof_input_with_id(recv_chip1_id),
                ],
            },
        );
        let mut challenger = engine.new_challenger();
        verifier
            .verify(&mut challenger, &pk.get_vk(), &proof)
            .expect("Verification failed");
        // The VM program will panic when the program cannot verify the proof.
        gen_vm_program_stark_for_test::<BabyBearPoseidon2Config>(
            program.clone(),
            proof.write(),
            vm_config.clone(),
        );
    }
    // Case 2: The second AIR is not presented.
    {
        let mut challenger = engine.new_challenger();
        send_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        recv_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let proof = prover.prove(
            &mut challenger,
            &pk,
            ProofInput {
                per_air: vec![
                    send_chip1.generate_air_proof_input_with_id(send_chip1_id),
                    recv_chip1.generate_air_proof_input_with_id(recv_chip1_id),
                ],
            },
        );
        let mut challenger = engine.new_challenger();
        verifier
            .verify(&mut challenger, &pk.get_vk(), &proof)
            .expect("Verification failed");
        // The VM program will panic when the program cannot verify the proof.
        gen_vm_program_stark_for_test::<BabyBearPoseidon2Config>(
            program.clone(),
            proof.write(),
            vm_config.clone(),
        );
    }
    // Case 3: Negative - unbalanced interactions.
    {
        let mut challenger = engine.new_challenger();
        recv_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let proof = prover.prove(
            &mut challenger,
            &pk,
            ProofInput {
                per_air: vec![recv_chip1.generate_air_proof_input_with_id(recv_chip1_id)],
            },
        );
        let mut challenger = engine.new_challenger();
        assert!(verifier
            .verify(&mut challenger, &pk.get_vk(), &proof)
            .is_err());
        // The VM program should panic when the proof cannot be verified.
        let unwind_res = catch_unwind(|| {
            gen_vm_program_stark_for_test::<BabyBearPoseidon2Config>(
                program.clone(),
                proof.write(),
                vm_config.clone(),
            )
        });
        assert!(unwind_res.is_err());
    }
}
