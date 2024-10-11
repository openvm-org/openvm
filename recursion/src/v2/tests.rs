use std::panic::catch_unwind;

use ax_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        setup_tracing,
    },
    dummy_airs::{
        fib_air::chip::FibonacciChip,
        interaction::dummy_interaction_air::{DummyInteractionChip, DummyInteractionData},
    },
    engine::StarkFriEngine,
};
use p3_uni_stark::StarkGenericConfig;
use stark_vm::{sdk::gen_vm_program_stark_for_test, vm::config::VmConfig};

use crate::{
    hints::Hintable,
    v2::{stark::VerifierProgramV2, types::new_from_inner_multi_vkv2},
};

#[test]
fn test_optional_air() {
    use afs_stark_backend::{
        engine::StarkEngine,
        prover::v2::types::{Chip, ProofInput},
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

    let m_advice = new_from_inner_multi_vkv2(&pk.get_vk());
    let vm_config = VmConfig::aggregation(7);
    let program = VerifierProgramV2::build(m_advice, &fri_params);

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
