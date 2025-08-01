use openvm_circuit::{arch::instructions::program::Program, utils::air_test_impl};
use openvm_stark_backend::engine::VerificationData;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
    utils::ProofInputForTest,
};

use crate::hints::InnerVal;

type InnerSC = BabyBearPoseidon2Config;

pub mod inner {
    use openvm_native_circuit::{test_native_config, NativeCpuBuilder};
    use openvm_native_compiler::conversion::CompilerOptions;
    use openvm_stark_sdk::{
        config::{
            baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
            FriParameters,
        },
        engine::{StarkFriEngine, VerificationDataWithFriParams},
    };

    use super::*;
    use crate::{hints::Hintable, stark::VerifierProgram, types::new_from_inner_multi_vk};

    pub fn build_verification_program(
        vparams: VerificationDataWithFriParams<InnerSC>,
        compiler_options: CompilerOptions,
    ) -> (Program<BabyBear>, Vec<Vec<InnerVal>>) {
        let VerificationDataWithFriParams { data, fri_params } = vparams;
        let VerificationData { proof, vk } = data;

        let advice = new_from_inner_multi_vk(&vk);
        cfg_if::cfg_if! {
            if #[cfg(feature = "metrics")] {
                let start = std::time::Instant::now();
            }
        }
        let program = VerifierProgram::build_with_options(advice, &fri_params, compiler_options);
        #[cfg(feature = "metrics")]
        metrics::gauge!("verify_program_compile_ms").set(start.elapsed().as_millis() as f64);

        let mut input_stream = Vec::new();
        input_stream.extend(proof.write());

        (program, input_stream)
    }

    /// Steps of recursive tests:
    /// 1. Generate a stark proof, P.
    /// 2. build a verifier program which can verify P.
    /// 3. Execute the verifier program and generate a proof.
    ///
    /// This is a convenience function with default configs for testing purposes only.
    pub fn run_recursive_test(
        test_proof_input: ProofInputForTest<BabyBearPoseidon2Config>,
        fri_params: FriParameters,
    ) {
        let vparams = test_proof_input
            .run_test(&BabyBearPoseidon2Engine::new(
                FriParameters::new_for_testing(1),
            ))
            .unwrap();

        let compiler_options = CompilerOptions::default();
        let (program, witness_stream) = build_verification_program(vparams, compiler_options);
        air_test_impl::<BabyBearPoseidon2Engine, _>(
            fri_params,
            NativeCpuBuilder,
            test_native_config(),
            program,
            witness_stream,
            1,
            true,
        )
        .unwrap();
    }
}
