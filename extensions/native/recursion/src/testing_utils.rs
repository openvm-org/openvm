use inner::build_verification_program;
use openvm_circuit::{
    arch::{
        execution_mode::e1::E1Ctx, instructions::program::Program, interpreter::InterpretedInstance,
    },
    utils::execute_and_prove_program,
};
use openvm_native_circuit::{test_native_config, NativeConfig};
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_stark_backend::{
    config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig},
    engine::VerificationData,
    p3_commit::PolynomialSpace,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    engine::{StarkFriEngine, VerificationDataWithFriParams},
    p3_baby_bear::BabyBear,
    utils::ProofInputForTest,
};

use crate::hints::InnerVal;

type InnerSC = BabyBearPoseidon2Config;

pub mod inner {
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
        let vparams = test_proof_input
            .run_test(&BabyBearPoseidon2Engine::new(
                FriParameters::new_for_testing(1),
            ))
            .unwrap();

        recursive_stark_test(
            vparams,
            CompilerOptions::default(),
            test_native_config(),
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
    vm_config: NativeConfig,
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

    let interpreter = InterpretedInstance::new(vm_config.clone(), program.clone());
    interpreter
        .execute(E1Ctx::new(None), witness_stream.clone())
        .unwrap();
    execute_and_prove_program(program, witness_stream, vm_config, engine)
}
