use ax_stark_sdk::{
    ax_stark_backend::{
        config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
        verifier::VerificationError,
    },
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    engine::{ProofInputForTest, StarkFriEngine, VerificationDataWithFriParams},
};
use axvm_instructions::{exe::AxVmExe, program::Program};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;

use crate::arch::{PersistenceType, VirtualMachine, VmConfig, VmExecutor};

pub fn air_test(config: VmConfig, exe: impl Into<AxVmExe<BabyBear>>) {
    air_test_with_min_segments(config, exe, vec![], 1);
}

pub fn air_test_with_min_segments(
    config: VmConfig,
    exe: impl Into<AxVmExe<BabyBear>>,
    input: Vec<Vec<BabyBear>>,
    min_segments: usize,
) {
    setup_tracing();

    let persistence_type = config.memory_config.persistence_type;

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let vm = VirtualMachine::<BabyBear, _, _>::new(engine, config);
    let pk = vm.keygen();
    let result = vm.execute_and_generate(exe, input).unwrap();
    let proofs = vm.prove(&pk, result);

    assert!(proofs.len() >= min_segments);
    match persistence_type {
        PersistenceType::Volatile => {
            assert_eq!(proofs.len(), 1);
            vm.verify_single(&pk.get_vk(), &proofs.into_iter().next().unwrap())
                .expect("segment proof should verify");
        }
        PersistenceType::Persistent => {
            vm.verify(&pk.get_vk(), proofs)
                .expect("segment proofs should verify");
        }
    }
}

// TODO[jpw]: this should be deleted once tests switch to new API
/// Generates the VM STARK circuit, in the form of AIRs and traces, but does not
/// do any proving. Output is the payload of everything the prover needs.
///
/// The output AIRs and traces are sorted by height in descending order.
pub fn gen_vm_program_test_proof_input<SC: StarkGenericConfig>(
    program: Program<Val<SC>>,
    input_stream: Vec<Vec<Val<SC>>>,
    config: VmConfig,
) -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "bench-metrics")] {
            // Run once with metrics collection enabled, which can improve runtime performance
            let mut config = config;
            config.collect_metrics = true;
            {
                let executor = VmExecutor::<Val<SC>>::new(config.clone());
                executor.execute(program.clone(), input_stream.clone()).unwrap();
            }
            // Run again with metrics collection disabled and measure trace generation time
            config.collect_metrics = false;
            let start = std::time::Instant::now();
        }
    }

    let executor = VmExecutor::<Val<SC>>::new(config);

    let mut result = executor
        .execute_and_generate(program, input_stream)
        .unwrap();
    assert_eq!(
        result.per_segment.len(),
        1,
        "only proving one segment for now"
    );

    let result = result.per_segment.pop().unwrap();
    #[cfg(feature = "bench-metrics")]
    {
        metrics::gauge!("trace_gen_time_ms").set(start.elapsed().as_millis() as f64);
    }

    ProofInputForTest {
        per_air: result.into_air_proof_input_vec(),
    }
}

type ExecuteAndProveResult<SC> =
    Result<(VerificationDataWithFriParams<SC>, Vec<Vec<Val<SC>>>), VerificationError>;

/// Executes program and runs simple STARK prover test (keygen, prove, verify).
pub fn execute_and_prove_program<SC: StarkGenericConfig, E: StarkFriEngine<SC>>(
    program: Program<Val<SC>>,
    input_stream: Vec<Vec<Val<SC>>>,
    config: VmConfig,
    engine: &E,
) -> ExecuteAndProveResult<SC>
where
    Val<SC>: PrimeField32,
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    let span = tracing::info_span!("execute_and_prove_program").entered();
    let test_proof_input = gen_vm_program_test_proof_input(program, input_stream, config);
    let pvs = test_proof_input
        .per_air
        .iter()
        .map(|air| air.raw.public_values.clone())
        .collect();
    let vparams = test_proof_input.run_test(engine)?;
    span.exit();
    Ok((vparams, pvs))
}
