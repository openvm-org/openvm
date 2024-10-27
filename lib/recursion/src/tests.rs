use std::{panic::catch_unwind, sync::Arc};

use ax_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    dummy_airs::{
        fib_air::chip::FibonacciChip,
        interaction::dummy_interaction_air::{
            DummyInteractionAir, DummyInteractionChip, DummyInteractionData,
        },
    },
    engine::{ProofInputForTest, StarkFriEngine},
    utils::to_field_vec,
};
use ax_stark_backend::{prover::types::AirProofInput, utils::disable_debug_builder, Chip};
use axvm_circuit::{sdk::gen_vm_program_test_proof_input, system::vm::config::VmConfig};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::{
    hints::Hintable, stark::VerifierProgram, testing_utils::inner::run_recursive_test,
    types::new_from_inner_multi_vk,
};

pub fn fibonacci_test_proof_input<SC: StarkGenericConfig>(n: usize) -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
{
    let fib_chip = FibonacciChip::new(0, 1, n);
    ProofInputForTest {
        per_air: vec![fib_chip.generate_air_proof_input()],
    }
}

pub fn interaction_test_proof_input<SC: StarkGenericConfig>() -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
{
    const BUS: usize = 0;
    let mut send_chip1 = DummyInteractionChip::new_without_partition(2, true, BUS);
    let mut send_chip2 = DummyInteractionChip::new_without_partition(2, true, BUS);
    let mut recv_chip = DummyInteractionChip::new_without_partition(2, false, BUS);
    send_chip1.load_data(DummyInteractionData {
        count: vec![1, 2, 4, 0],
        fields: vec![vec![1, 1], vec![1, 2], vec![3, 4], vec![888, 999]],
    });
    send_chip2.load_data(DummyInteractionData {
        count: vec![4, 0],
        fields: vec![vec![3, 4], vec![0, 0]],
    });
    recv_chip.load_data(DummyInteractionData {
        count: vec![1, 2, 8, 0],
        fields: vec![vec![1, 1], vec![1, 2], vec![3, 4], vec![9999, 0]],
    });

    ProofInputForTest {
        per_air: vec![
            send_chip1.generate_air_proof_input(),
            send_chip2.generate_air_proof_input(),
            recv_chip.generate_air_proof_input(),
        ],
    }
}

pub fn unordered_test_proof_input<SC: StarkGenericConfig>() -> ProofInputForTest<SC>
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

    let sender_air_proof_input = AirProofInput::simple_no_pis(Arc::new(sender_air), sender_trace);
    let receiver_air_proof_input =
        AirProofInput::simple_no_pis(Arc::new(receiver_air), receiver_trace);

    ProofInputForTest {
        per_air: vec![sender_air_proof_input, receiver_air_proof_input],
    }
}

#[test]
fn test_fibonacci_small() {
    run_recursive_test(
        fibonacci_test_proof_input::<BabyBearPoseidon2Config>(1 << 5),
        standard_fri_params_with_100_bits_conjectured_security(3),
    )
}

#[test]
fn test_fibonacci() {
    // test lde = 27
    run_recursive_test(
        fibonacci_test_proof_input::<BabyBearPoseidon2Config>(1 << 24),
        FriParameters {
            log_blowup: 3,
            num_queries: 2,
            proof_of_work_bits: 0,
        },
    )
}

#[test]
fn test_interactions() {
    run_recursive_test(
        interaction_test_proof_input::<BabyBearPoseidon2Config>(),
        standard_fri_params_with_100_bits_conjectured_security(3),
    )
}

#[test]
fn test_unordered() {
    run_recursive_test(
        unordered_test_proof_input::<BabyBearPoseidon2Config>(),
        standard_fri_params_with_100_bits_conjectured_security(3),
    )
}

#[test]
fn test_optional_air() {
    use ax_stark_backend::{engine::StarkEngine, prover::types::ProofInput, Chip};
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3);
    let engine = BabyBearPoseidon2Engine::new(fri_params);
    let fib_chip = FibonacciChip::new(0, 1, 8);
    let send_chip1 = DummyInteractionChip::new_without_partition(1, true, 0);
    let send_chip2 = DummyInteractionChip::new_with_partition(engine.config().pcs(), 1, true, 0);
    let recv_chip1 = DummyInteractionChip::new_without_partition(1, false, 0);
    let mut keygen_builder = engine.keygen_builder();
    let fib_chip_id = keygen_builder.add_air(fib_chip.air());
    let send_chip1_id = keygen_builder.add_air(send_chip1.air());
    let send_chip2_id = keygen_builder.add_air(send_chip2.air());
    let recv_chip1_id = keygen_builder.add_air(recv_chip1.air());
    let pk = keygen_builder.generate_pk();
    let prover = engine.prover();
    let verifier = engine.verifier();

    let m_advice = new_from_inner_multi_vk(&pk.get_vk());
    let vm_config = VmConfig::aggregation(4, 7);
    let program = VerifierProgram::build(m_advice, &fri_params);

    // Case 1: All AIRs are present.
    {
        let fib_chip = fib_chip.clone();
        let mut send_chip1 = send_chip1.clone();
        let mut send_chip2 = send_chip2.clone();
        let mut recv_chip1 = recv_chip1.clone();
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
        gen_vm_program_test_proof_input::<BabyBearPoseidon2Config>(
            program.clone(),
            proof.write(),
            vm_config.clone(),
        );
    }
    // Case 2: The second AIR is not presented.
    {
        let mut send_chip1 = send_chip1.clone();
        let mut recv_chip1 = recv_chip1.clone();
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
        gen_vm_program_test_proof_input::<BabyBearPoseidon2Config>(
            program.clone(),
            proof.write(),
            vm_config.clone(),
        );
    }
    // Case 3: Negative - unbalanced interactions.
    {
        disable_debug_builder();
        let mut recv_chip1 = recv_chip1.clone();
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
            gen_vm_program_test_proof_input::<BabyBearPoseidon2Config>(
                program.clone(),
                proof.write(),
                vm_config.clone(),
            )
        });
        assert!(unwind_res.is_err());
    }
}
