use std::rc::Rc;

use afs_compiler::util::execute_and_prove_program;
use afs_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    prover::{trace::TraceCommitmentBuilder, types::Proof},
    rap::AnyRap,
    verifier::MultiTraceStarkVerifier,
};
use ax_sdk::{config::FriParameters, engine::StarkEngine};
use p3_baby_bear::BabyBear;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_strict_usize;
use stark_vm::program::Program;

use crate::{
    hints::InnerVal,
    stark::{sort_chips, VerifierProgram},
    types::{new_from_inner_multi_vk, VerifierInput},
};

/// All necessary data to verify a Stark proof.
pub struct VerificationData<SC: StarkGenericConfig> {
    pub vk: MultiStarkVerifyingKey<SC>,
    pub proof: Proof<SC>,
    pub fri_params: FriParameters,
}

/// A struct that contains all the necessary data to build a verifier for a Stark.
pub struct StarkForTest<SC: StarkGenericConfig> {
    pub any_raps: Vec<Rc<dyn AnyRap<SC>>>,
    pub traces: Vec<RowMajorMatrix<BabyBear>>,
    pub pvs: Vec<Vec<BabyBear>>,
}

pub mod outer {
    use ax_sdk::config::baby_bear_poseidon2_outer::{
        engine_from_perm, outer_perm, BabyBearPoseidon2OuterConfig,
    };

    use super::*;

    pub fn make_verification_data(
        raps: &[&dyn AnyRap<BabyBearPoseidon2OuterConfig>],
        traces: Vec<RowMajorMatrix<BabyBear>>,
        pvs: &[Vec<BabyBear>],
        fri_params: FriParameters,
    ) -> VerificationData<BabyBearPoseidon2OuterConfig> {
        let num_pvs: Vec<usize> = pvs.iter().map(|pv| pv.len()).collect();

        let trace_heights: Vec<usize> = traces.iter().map(|t| t.height()).collect();
        let log_degree = log2_strict_usize(trace_heights.into_iter().max().unwrap());

        let engine = engine_from_perm(outer_perm(), log_degree, fri_params);

        let mut keygen_builder = engine.keygen_builder();
        for (&rap, &num_pv) in raps.iter().zip(num_pvs.iter()) {
            keygen_builder.add_air(rap, num_pv);
        }

        let pk = keygen_builder.generate_pk();
        let vk = pk.vk();

        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
        for trace in traces.clone() {
            trace_builder.load_trace(trace);
        }
        trace_builder.commit_current();

        let main_trace_data = trace_builder.view(&vk, raps.to_vec());

        let mut challenger = engine.new_challenger();
        let proof = prover.prove(&mut challenger, &pk, main_trace_data, pvs);

        let verifier = MultiTraceStarkVerifier::new(prover.config);
        verifier
            .verify(&mut engine.new_challenger(), &vk, &proof, pvs)
            .expect("proof should verify");

        VerificationData {
            vk,
            proof,
            fri_params: engine.fri_params,
        }
    }
}

pub mod inner {
    use ax_sdk::config::{
        baby_bear_poseidon2::{default_perm, engine_from_perm, BabyBearPoseidon2Config},
        FriParameters,
    };
    use stark_vm::vm::config::VmConfig;

    use super::*;
    use crate::hints::Hintable;

    pub fn make_verification_data(
        raps: &[&dyn AnyRap<BabyBearPoseidon2Config>],
        traces: Vec<RowMajorMatrix<BabyBear>>,
        pvs: &[Vec<BabyBear>],
        fri_params: FriParameters,
    ) -> VerificationData<BabyBearPoseidon2Config> {
        let num_pvs: Vec<usize> = pvs.iter().map(|pv| pv.len()).collect();

        let trace_heights: Vec<usize> = traces.iter().map(|t| t.height()).collect();
        let log_degree = log2_strict_usize(trace_heights.into_iter().max().unwrap());

        let engine = engine_from_perm(default_perm(), log_degree, fri_params);

        let mut keygen_builder = engine.keygen_builder();
        for (&rap, &num_pv) in raps.iter().zip(num_pvs.iter()) {
            keygen_builder.add_air(rap, num_pv);
        }

        let pk = keygen_builder.generate_pk();
        let vk = pk.vk();

        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
        for trace in traces.clone() {
            trace_builder.load_trace(trace);
        }
        trace_builder.commit_current();

        let main_trace_data = trace_builder.view(&vk, raps.to_vec());

        let mut challenger = engine.new_challenger();
        let proof = prover.prove(&mut challenger, &pk, main_trace_data, pvs);

        let verifier = MultiTraceStarkVerifier::new(prover.config);
        verifier
            .verify(&mut engine.new_challenger(), &vk, &proof, pvs)
            .expect("proof should verify");

        VerificationData {
            vk,
            proof,
            fri_params: engine.fri_params,
        }
    }
    pub fn build_verification_program(
        pvs: Vec<Vec<InnerVal>>,
        vparams: VerificationData<BabyBearPoseidon2Config>,
    ) -> (Program<BabyBear>, Vec<Vec<InnerVal>>) {
        let VerificationData {
            vk,
            proof,
            fri_params,
        } = vparams;

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
    pub fn run_recursive_test(
        stark_for_test: StarkForTest<BabyBearPoseidon2Config>,
        fri_params: FriParameters,
    ) {
        let StarkForTest {
            any_raps,
            traces,
            pvs,
        } = stark_for_test;
        let any_raps: Vec<_> = any_raps.iter().map(|x| x.as_ref()).collect();
        let (any_raps, traces, pvs) = sort_chips(any_raps, traces, pvs);

        let vparams = make_verification_data(&any_raps, traces, &pvs, fri_params);

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
