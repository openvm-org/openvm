use afs_compiler::{conversion::CompilerOptions, util::execute_and_prove_program};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    engine::VerificationData,
    verifier::VerificationError,
};
use ax_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    engine::{StarkForTest, StarkFriEngine, VerificationDataWithFriParams},
};
use inner::build_verification_program;
use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use stark_vm::{program::Program, vm::config::VmConfig};

use crate::hints::InnerVal;

type InnerSC = BabyBearPoseidon2Config;

pub mod inner {
    use afs_compiler::conversion::CompilerOptions;
    use ax_sdk::{
        config::{
            baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
            FriParameters,
        },
        engine::{StarkFriEngine, VerificationDataWithFriParams},
    };
    use stark_vm::vm::config::VmConfig;

    use super::*;
    use crate::{
        hints::Hintable,
        v2::{stark::VerifierProgramV2, types::new_from_inner_multi_vkv2},
    };

    pub fn build_verification_program(
        vparams: VerificationDataWithFriParams<InnerSC>,
        compiler_options: CompilerOptions,
    ) -> (Program<BabyBear>, Vec<Vec<InnerVal>>) {
        let VerificationDataWithFriParams { data, fri_params } = vparams;
        let VerificationData { proof, vk } = data;

        let advice = new_from_inner_multi_vkv2(&vk);
        cfg_if::cfg_if! {
            if #[cfg(feature = "bench-metrics")] {
                let start = std::time::Instant::now();
            }
        }
        let program = VerifierProgramV2::build_with_options(advice, &fri_params, compiler_options);
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
        stark_for_test: StarkForTest<BabyBearPoseidon2Config>,
        fri_params: FriParameters,
    ) {
        let StarkForTest { air_infos } = stark_for_test;
        let vparams =
            <BabyBearPoseidon2Engine as StarkFriEngine<BabyBearPoseidon2Config>>::run_test_fast(
                air_infos,
            )
            .unwrap();

        recursive_stark_test(
            vparams,
            CompilerOptions::default(),
            VmConfig::aggregation(7),
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
) -> Result<(VerificationDataWithFriParams<AggSC>, Vec<Vec<Val<AggSC>>>), VerificationError>
where
    Domain<AggSC>: PolynomialSpace<Val = BabyBear>,
    AggSC::Pcs: Sync,
    Domain<AggSC>: Send + Sync,
    PcsProverData<AggSC>: Send + Sync,
    Com<AggSC>: Send + Sync,
    AggSC::Challenge: Send + Sync,
    PcsProof<AggSC>: Send + Sync,
{
    let (program, witness_stream) = build_verification_program(vparams, compiler_options);

    execute_and_prove_program(program, witness_stream, vm_config, engine)
}
