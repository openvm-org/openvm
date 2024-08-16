use afs_compiler::{conversion::CompilerOptions, ir::Builder};
use afs_test_utils::config::{baby_bear_poseidon2::BabyBearPoseidon2Config, FriParameters};
use p3_baby_bear::BabyBear;
use stark_vm::program::Program;

use crate::{
    config::outer::OuterConfig,
    fri::TwoAdicFriPcsVariable,
    stark::{DynRapForRecursion, StarkVerifier, VerifierProgram},
    types::{InnerConfig, MultiStarkVerificationAdvice, VerifierInput, VerifierInputVariable},
    utils::const_fri_config,
};

pub fn build_circuit_verify_operations(
    builder: &mut Builder<OuterConfig>,
    raps: Vec<&dyn DynRapForRecursion<OuterConfig>>,
    constants: MultiStarkVerificationAdvice<OuterConfig>,
    fri_params: &FriParameters,
    input: VerifierInputVariable<OuterConfig>,
) {
    let mut builder = Builder::<OuterConfig>::default();

    builder.cycle_tracker_start("VerifierProgram");
    let input: VerifierInputVariable<_> = builder.uninit();
    // VerifierInput::<BabyBearPoseidon2Config>::witness(&input, &mut builder);

    let pcs = TwoAdicFriPcsVariable {
        config: const_fri_config(&mut builder, fri_params),
    };
    StarkVerifier::verify(&mut builder, &pcs, raps, constants, &input);

    builder.cycle_tracker_end("VerifierProgram");
    builder.halt();
}
