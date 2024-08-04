use afs_compiler::util::execute_and_prove_program;
use afs_recursion::{
    hints::{Hintable, InnerVal},
    stark::{sort_chips, DynRapForRecursion, VerifierProgram},
    types::{new_from_multi_vk, InnerConfig, VerifierInput},
};
use afs_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    prover::{trace::TraceCommitmentBuilder, types::Proof},
    rap::AnyRap,
    verifier::MultiTraceStarkVerifier,
};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{default_perm, engine_from_perm, BabyBearPoseidon2Config},
        FriParameters,
    },
    engine::StarkEngine,
};
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_strict_usize;
use stark_vm::cpu::trace::Instruction;

pub struct VerificationParams<SC: StarkGenericConfig> {
    pub vk: MultiStarkVerifyingKey<SC>,
    pub proof: Proof<SC>,
    pub fri_params: FriParameters,
}

pub fn make_verification_params(
    raps: &[&dyn AnyRap<BabyBearPoseidon2Config>],
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: &[Vec<BabyBear>],
    fri_params: FriParameters,
) -> VerificationParams<BabyBearPoseidon2Config> {
    let num_pvs: Vec<usize> = pvs.iter().map(|pv| pv.len()).collect();

    let engine = engine_from_perm(default_perm(), fri_params);

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
        .verify(
            &mut engine.new_challenger(),
            &vk,
            raps.to_vec(),
            &proof,
            pvs,
        )
        .expect("proof should verify");

    VerificationParams {
        vk,
        proof,
        fri_params: engine.fri_params,
    }
}

pub fn build_verification_program(
    rec_raps: Vec<&dyn DynRapForRecursion<InnerConfig>>,
    pvs: Vec<Vec<InnerVal>>,
    vparams: VerificationParams<BabyBearPoseidon2Config>,
) -> (Vec<Instruction<BabyBear>>, Vec<Vec<InnerVal>>) {
    let VerificationParams {
        vk,
        proof,
        fri_params,
    } = vparams;

    let advice = new_from_multi_vk(&vk);
    let program = VerifierProgram::build(rec_raps, advice, &fri_params);

    let log_degree_per_air = proof
        .degrees
        .iter()
        .map(|degree| log2_strict_usize(*degree))
        .collect();

    let input = VerifierInput {
        proof,
        log_degree_per_air,
        public_values: pvs.clone(),
    };

    let mut input_stream = Vec::new();
    input_stream.extend(input.write());

    (program, input_stream)
}

#[allow(dead_code)]
pub fn run_recursive_test(
    // TODO: find way to de-duplicate parameters
    any_raps: Vec<&dyn AnyRap<BabyBearPoseidon2Config>>,
    rec_raps: Vec<&dyn DynRapForRecursion<InnerConfig>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: Vec<Vec<BabyBear>>,
    fri_params: FriParameters,
) {
    let (any_raps, rec_raps, traces, pvs) = sort_chips(any_raps, rec_raps, traces, pvs);

    let vparams = make_verification_params(&any_raps, traces, &pvs, fri_params);

    let (program, witness_stream) = build_verification_program(rec_raps, pvs, vparams);
    execute_and_prove_program::<1>(program, witness_stream);
}
