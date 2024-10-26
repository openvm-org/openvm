use afs_compiler::ir::{Builder, DslIr, TracedVec};
use afs_stark_backend::prover::types::Proof;
use ax_sdk::config::{baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig, FriParameters};

use crate::{
    challenger::multi_field32::MultiField32ChallengerVariable, config::outer::OuterConfig,
    fri::TwoAdicFriPcsVariable, stark::StarkVerifier, types::MultiStarkVerificationAdvice,
    utils::const_fri_config, witness::Witnessable,
};

pub fn build_circuit_verify_operations(
    advice: MultiStarkVerificationAdvice<OuterConfig>,
    fri_params: &FriParameters,
    proof: &Proof<BabyBearPoseidon2OuterConfig>,
) -> TracedVec<DslIr<OuterConfig>> {
    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;

    builder.cycle_tracker_start("VerifierProgram");
    let input = proof.read(&mut builder);

    let pcs = TwoAdicFriPcsVariable {
        config: const_fri_config(&mut builder, fri_params),
    };
    StarkVerifier::verify::<MultiField32ChallengerVariable<_>>(&mut builder, &pcs, &advice, &input);

    builder.cycle_tracker_end("VerifierProgram");
    builder.operations
}
