use ax_stark_backend::{
    config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig},
    engine::VerificationData,
    verifier::VerificationError,
};
use ax_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    engine::{ProofInputForTest, StarkFriEngine, VerificationDataWithFriParams},
};
use axvm_circuit::{
    arch::{instructions::program::Program, VmConfig},
    utils::execute_and_prove_program,
};
use axvm_native_compiler::conversion::CompilerOptions;
use inner::build_verification_program;
use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;

use crate::hints::InnerVal;

type InnerSC = BabyBearPoseidon2Config;

pub mod inner {
    use ax_stark_sdk::{
        config::{
            baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
            FriParameters,
        },
        engine::{StarkFriEngine, VerificationDataWithFriParams},
    };
    use axvm_circuit::arch::VmConfig;
    use axvm_native_compiler::conversion::CompilerOptions;

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
            if #[cfg(feature = "bench-metrics")] {
                let start = std::time::Instant::now();
            }
        }
        let program = VerifierProgram::build_with_options(advice, &fri_params, compiler_options);
        #[cfg(feature = "bench-metrics")]
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
        let ProofInputForTest {
            per_air: air_proof_inputs,
        } = test_proof_input;
        let vparams =
            <BabyBearPoseidon2Engine as StarkFriEngine<BabyBearPoseidon2Config>>::run_test_fast(
                air_proof_inputs,
            )
            .unwrap();

        recursive_stark_test(
            vparams,
            CompilerOptions::default(),
            VmConfig::aggregation(4, 7),
            &BabyBearPoseidon2Engine::new(fri_params),
        )
        .unwrap();
    }
}

/// 1. Builds the recursive verification program to verify `vparams`
/// 2. Execute and proves the program in VM with `AggSC` config using `engine`.
///
/// The `vparams` must be from the BabyBearPoseidon2 stark config for the recursion
/// program to work at the moment.
#[allow(clippy::type_complexity)]
pub fn recursive_stark_test<AggSC: StarkGenericConfig, E: StarkFriEngine<AggSC>>(
    vparams: VerificationDataWithFriParams<InnerSC>,
    compiler_options: CompilerOptions,
    vm_config: VmConfig,
    engine: &E,
) -> Result<VerificationDataWithFriParams<AggSC>, VerificationError>
where
    Domain<AggSC>: PolynomialSpace<Val = BabyBear>,
    Domain<AggSC>: Send + Sync,
    PcsProverData<AggSC>: Send + Sync,
    Com<AggSC>: Send + Sync,
    PcsProof<AggSC>: Send + Sync,
{
    let (program, witness_stream) = build_verification_program(vparams, compiler_options);

    execute_and_prove_program(program, witness_stream, vm_config, engine)
}
