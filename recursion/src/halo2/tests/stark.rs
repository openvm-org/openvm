use afs_compiler::ir::Witness;
use ax_sdk::{
    config::{
        baby_bear_poseidon2_outer::{BabyBearPoseidon2OuterConfig, BabyBearPoseidon2OuterEngine},
        setup_tracing,
    },
    engine::{StarkForTest, StarkFriEngine},
};

use crate::{
    config::outer::new_from_outer_multi_vkv2,
    halo2::Halo2Prover,
    stark::outer::build_circuit_verify_operations,
    tests::{fibonacci_stark_for_test, interaction_stark_for_test},
    witness::Witnessable,
};

#[test]
fn test_fibonacci() {
    setup_tracing();
    run_recursive_test(fibonacci_stark_for_test::<BabyBearPoseidon2OuterConfig>(16))
}

#[test]
fn test_interactions() {
    // Please make sure kzg trusted params are downloaded before running the test.
    setup_tracing();
    run_recursive_test(interaction_stark_for_test::<BabyBearPoseidon2OuterConfig>())
}

fn run_recursive_test(stark_for_test: StarkForTest<BabyBearPoseidon2OuterConfig>) {
    let vparams =
        <BabyBearPoseidon2OuterEngine as StarkFriEngine<BabyBearPoseidon2OuterConfig>>::run_test_fast(
            stark_for_test.air_infos,
        )
        .unwrap();
    let advice = new_from_outer_multi_vkv2(&vparams.data.vk);
    let proof = vparams.data.proof;

    let mut witness = Witness::default();
    proof.write(&mut witness);
    let operations = build_circuit_verify_operations(advice, &vparams.fri_params, &proof);
    Halo2Prover::mock(20, operations, witness);
}
