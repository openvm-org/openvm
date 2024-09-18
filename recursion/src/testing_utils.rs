use std::rc::Rc;

use afs_compiler::util::execute_and_prove_program;
use afs_stark_backend::{engine::VerificationData, rap::AnyRap};
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::StarkGenericConfig;
use stark_vm::program::Program;

use crate::{
    hints::InnerVal,
    stark::{sort_chips, VerifierProgram},
    types::{new_from_inner_multi_vk, VerifierInput},
};

/// A struct that contains all the necessary data to build a verifier for a Stark.
pub struct StarkForTest<SC: StarkGenericConfig> {
    pub any_raps: Vec<Rc<dyn AnyRap<SC>>>,
    pub traces: Vec<RowMajorMatrix<BabyBear>>,
    pub pvs: Vec<Vec<BabyBear>>,
}

pub mod inner {
    use ax_sdk::{
        config::baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        engine::{StarkFriEngine, VerificationDataWithFriParams},
    };
    use stark_vm::vm::config::VmConfig;

    use super::*;
    use crate::hints::Hintable;

    pub fn build_verification_program(
        pvs: Vec<Vec<InnerVal>>,
        vparams: VerificationDataWithFriParams<BabyBearPoseidon2Config>,
    ) -> (Program<BabyBear>, Vec<Vec<InnerVal>>) {
        let VerificationDataWithFriParams { data, fri_params } = vparams;
        let VerificationData { proof, vk } = data;

        let advice = new_from_inner_multi_vk(&vk);
        let program = VerifierProgram::build(advice, &fri_params);

        let log_degree_per_air = proof.log_degrees();

        let input = VerifierInput {
            proof,
            log_degree_per_air,
            public_values: pvs.clone(),
        };

        let mut input_stream = Vec::new();
        input_stream.extend(input.write());

        (program, input_stream)
    }

    /// Steps of recursive tests:
    /// 1. Generate a stark proof, P.
    /// 2. build a verifier program which can verify P.
    /// 3. Execute the verifier program and generate a proof.
    pub fn run_recursive_test(stark_for_test: StarkForTest<BabyBearPoseidon2Config>) {
        let StarkForTest {
            any_raps,
            traces,
            pvs,
        } = stark_for_test;
        let any_raps: Vec<_> = any_raps.iter().map(|x| x.as_ref()).collect();
        let (any_raps, traces, pvs) = sort_chips(any_raps, traces, pvs);

        let vparams =
            <BabyBearPoseidon2Engine as StarkFriEngine<BabyBearPoseidon2Config>>::run_simple_test(
                &any_raps, traces, &pvs,
            )
            .unwrap();

        let (program, witness_stream) = build_verification_program(pvs, vparams);
        execute_and_prove_program(
            program,
            witness_stream,
            VmConfig {
                num_public_values: 4,
                ..Default::default()
            },
        );
    }
}
