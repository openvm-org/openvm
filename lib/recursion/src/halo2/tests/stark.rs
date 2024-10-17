use afs_compiler::ir::Witness;
use ax_sdk::{
    config::baby_bear_poseidon2_outer::{
        BabyBearPoseidon2OuterConfig, BabyBearPoseidon2OuterEngine,
    },
    engine::{ProofInputForTest, StarkFriEngine},
};

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::Halo2Prover,
    stark::outer::build_circuit_verify_operations,
    tests::{fibonacci_test_proof_input, interaction_test_proof_input},
    witness::Witnessable,
};

#[test]
fn test_fibonacci() {
    run_recursive_test(fibonacci_test_proof_input::<BabyBearPoseidon2OuterConfig>(
        16,
    ))
}

#[test]
fn test_interactions() {
    // Please make sure kzg trusted params are downloaded before running the test.
    run_recursive_test(interaction_test_proof_input::<BabyBearPoseidon2OuterConfig>())
}

fn run_recursive_test(test_proof_input: ProofInputForTest<BabyBearPoseidon2OuterConfig>) {
    let vparams =
        <BabyBearPoseidon2OuterEngine as StarkFriEngine<BabyBearPoseidon2OuterConfig>>::run_test_fast(
            test_proof_input.per_air,
        )
        .unwrap();
    let advice = new_from_outer_multi_vk(&vparams.data.vk);
    let proof = vparams.data.proof;

    let mut witness = Witness::default();
    proof.write(&mut witness);
    let operations = build_circuit_verify_operations(advice, &vparams.fri_params, &proof);
    Halo2Prover::mock(20, operations, witness);
}
