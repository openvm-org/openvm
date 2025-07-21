use std::sync::Arc;

use openvm_native_circuit::{execute_program_with_config, test_native_config, NativeCpuBuilder};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::BusIndex,
    p3_field::PrimeField32,
    p3_matrix::dense::RowMajorMatrix,
    prover::{
        hal::DeviceDataTransporter,
        types::{AirProvingContext, ProvingContext},
    },
    utils::disable_debug_builder,
    Chip,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        FriParameters,
    },
    dummy_airs::{
        fib_air::chip::FibonacciChip,
        interaction::dummy_interaction_air::{
            DummyInteractionAir, DummyInteractionChip, DummyInteractionData,
        },
    },
    engine::StarkFriEngine,
    utils::{to_field_vec, ProofInputForTest},
};

use crate::{
    hints::Hintable, stark::VerifierProgram, testing_utils::inner::run_recursive_test,
    types::new_from_inner_multi_vk,
};

pub fn fibonacci_test_proof_input<SC: StarkGenericConfig>(n: usize) -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
{
    let fib_chip = FibonacciChip::new(0, 1, n);
    let airs = vec![fib_chip.air()];
    let air_ctx = fib_chip.generate_proving_ctx(());
    ProofInputForTest {
        airs,
        per_air: vec![air_ctx],
    }
}

pub fn interaction_test_proof_input<SC: StarkGenericConfig>() -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
{
    const BUS: BusIndex = 0;
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

    let airs = vec![send_chip1.air(), send_chip2.air(), recv_chip.air()];
    let per_air = vec![
        send_chip1.generate_proving_ctx(()),
        send_chip2.generate_proving_ctx(()),
        recv_chip.generate_proving_ctx(()),
    ];
    ProofInputForTest { airs, per_air }
}

pub fn unordered_test_proof_input<SC: StarkGenericConfig>() -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
{
    const BUS: BusIndex = 0;
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

    let sender_ctx = AirProvingContext::simple_no_pis(Arc::new(sender_trace));
    let receiver_ctx = AirProvingContext::simple_no_pis(Arc::new(receiver_trace));

    ProofInputForTest {
        airs: vec![Arc::new(sender_air), Arc::new(receiver_air)],
        per_air: vec![sender_ctx, receiver_ctx],
    }
}

#[test]
fn test_fibonacci_small() {
    run_recursive_test(
        fibonacci_test_proof_input::<BabyBearPoseidon2Config>(1 << 5),
        FriParameters::new_for_testing(3),
    )
}

#[test]
fn test_fibonacci() {
    // test lde = 27
    run_recursive_test(
        fibonacci_test_proof_input::<BabyBearPoseidon2Config>(1 << 24),
        FriParameters {
            log_blowup: 3,
            log_final_poly_len: 0,
            num_queries: 2,
            proof_of_work_bits: 0,
        },
    )
}

#[test]
fn test_interactions() {
    run_recursive_test(
        interaction_test_proof_input::<BabyBearPoseidon2Config>(),
        FriParameters::new_for_testing(3),
    )
}

#[test]
fn test_unordered() {
    run_recursive_test(
        unordered_test_proof_input::<BabyBearPoseidon2Config>(),
        FriParameters::new_for_testing(3),
    )
}

#[test]
fn test_optional_air() {
    use openvm_stark_backend::engine::StarkEngine;
    let fri_params = FriParameters::new_for_testing(3);
    let engine = BabyBearPoseidon2Engine::new(fri_params);
    let fib_chip = FibonacciChip::new(0, 1, 8);
    let send_chip1 = DummyInteractionChip::new_without_partition(1, true, 0);
    let send_chip2 = DummyInteractionChip::new_with_partition(engine.device().clone(), 1, true, 0);
    let recv_chip1 = DummyInteractionChip::new_without_partition(1, false, 0);
    let mut keygen_builder = engine.keygen_builder();
    let fib_chip_id = keygen_builder.add_air(fib_chip.air());
    let send_chip1_id = keygen_builder.add_air(send_chip1.air());
    let send_chip2_id = keygen_builder.add_air(send_chip2.air());
    let recv_chip1_id = keygen_builder.add_air(recv_chip1.air());
    let pk = keygen_builder.generate_pk();

    let m_advice = new_from_inner_multi_vk(&pk.get_vk());
    let config = test_native_config();
    let program = VerifierProgram::build(m_advice, &fri_params);

    // Case 1: All AIRs are present.
    {
        let fib_chip = fib_chip.clone();
        let mut send_chip1 = send_chip1.clone();
        let mut send_chip2 = send_chip2.clone();
        let mut recv_chip1 = recv_chip1.clone();
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
        let proof = engine
            .prove_then_verify(
                &pk,
                ProvingContext {
                    per_air: vec![
                        (fib_chip_id, fib_chip.generate_proving_ctx(())),
                        (send_chip1_id, send_chip1.generate_proving_ctx(())),
                        (send_chip2_id, send_chip2.generate_proving_ctx(())),
                        (recv_chip1_id, recv_chip1.generate_proving_ctx(())),
                    ],
                },
            )
            .unwrap();
        // The VM program will panic when the program cannot verify the proof.
        assert!(execute_program_with_config::<BabyBearPoseidon2Engine, _>(
            program.clone(),
            proof.write(),
            NativeCpuBuilder,
            config.clone()
        )
        .is_ok());
    }
    // Case 2: The second AIR is not presented.
    {
        let mut send_chip1 = send_chip1.clone();
        let mut recv_chip1 = recv_chip1.clone();
        send_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        recv_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let proof = engine
            .prove_then_verify(
                &pk,
                ProvingContext {
                    per_air: vec![
                        (send_chip1_id, send_chip1.generate_proving_ctx(())),
                        (recv_chip1_id, recv_chip1.generate_proving_ctx(())),
                    ],
                },
            )
            .unwrap();
        // The VM program will panic when the program cannot verify the proof.
        assert!(execute_program_with_config::<BabyBearPoseidon2Engine, _>(
            program.clone(),
            proof.write(),
            NativeCpuBuilder,
            config.clone()
        )
        .is_ok());
    }
    // Case 3: Negative - unbalanced interactions.
    {
        disable_debug_builder();
        let mut recv_chip1 = recv_chip1.clone();
        recv_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let proof = engine.prove(
            &d_pk,
            ProvingContext {
                per_air: vec![(recv_chip1_id, recv_chip1.generate_proving_ctx(()))],
            },
        );
        assert!(engine.verify(&pk.get_vk(), &proof).is_err());
        // The VM program should panic when the proof cannot be verified.
        assert!(execute_program_with_config::<BabyBearPoseidon2Engine, _>(
            program.clone(),
            proof.write(),
            NativeCpuBuilder,
            config.clone()
        )
        .is_err());
    }
}
