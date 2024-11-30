use ax_stark_backend::Chip;
use ax_stark_sdk::{
    ax_stark_backend::{
        config::{StarkGenericConfig, Val},
        verifier::VerificationError,
    },
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        setup_tracing, FriParameters,
    },
    engine::{ProofInputForTest, StarkFriEngine, VerificationDataWithFriParams},
};
use axvm_instructions::{exe::AxVmExe, program::Program};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;

use crate::arch::{
    new_vm::{VirtualMachine as NewVirtualMachine, VmExecutor as NewVmExecutor},
    VirtualMachine, VmConfig, VmGenericConfig, VmMemoryState,
};

pub fn air_test(config: VmConfig, exe: impl Into<AxVmExe<BabyBear>>) {
    air_test_with_min_segments(config, exe, vec![], 1);
}

/// Executes the VM and returns the final memory state.
pub fn air_test_with_min_segments(
    config: VmConfig,
    exe: impl Into<AxVmExe<BabyBear>>,
    input: Vec<Vec<BabyBear>>,
    min_segments: usize,
) -> Option<VmMemoryState<BabyBear>> {
    setup_tracing();
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let vm = VirtualMachine::new(engine, config);
    let pk = vm.keygen();
    let mut result = vm.execute_and_generate(exe, input).unwrap();
    let final_memory = result.final_memory.take();
    let proofs = vm.prove(&pk, result);

    assert!(proofs.len() >= min_segments);
    vm.verify(&pk.get_vk(), proofs)
        .expect("segment proofs should verify");
    final_memory
}

/// Executes the VM and returns the final memory state.
pub fn new_air_test_with_min_segments<VmConfig>(
    config: VmConfig,
    exe: impl Into<AxVmExe<BabyBear>>,
    input: Vec<Vec<BabyBear>>,
    min_segments: usize,
) -> Option<VmMemoryState<BabyBear>>
where
    VmConfig: VmGenericConfig<BabyBear>,
    VmConfig::Executor: Chip<BabyBearPoseidon2Config>,
    VmConfig::Periphery: Chip<BabyBearPoseidon2Config>,
{
    setup_tracing();
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let vm = NewVirtualMachine::new(engine, config);
    let pk = vm.keygen();
    let mut result = vm.execute_and_generate(exe, input).unwrap();
    let final_memory = result.final_memory.take();
    let proofs = vm.prove(&pk, result);

    assert!(proofs.len() >= min_segments);
    vm.verify(&pk.get_vk(), proofs)
        .expect("segment proofs should verify");
    final_memory
}

// TODO[jpw]: this should be deleted once tests switch to new API
/// Generates the VM STARK circuit, in the form of AIRs and traces, but does not
/// do any proving. Output is the payload of everything the prover needs.
///
/// The output AIRs and traces are sorted by height in descending order.
pub fn gen_vm_program_test_proof_input<SC: StarkGenericConfig, VmConfig>(
    program: Program<Val<SC>>,
    input_stream: Vec<Vec<Val<SC>>>,
    #[allow(unused_mut)] mut config: VmConfig,
) -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
    VmConfig: VmGenericConfig<Val<SC>> + Clone,
    VmConfig::Executor: Chip<SC>,
    VmConfig::Periphery: Chip<SC>,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "bench-metrics")] {
            // Run once with metrics collection enabled, which can improve runtime performance
            config.system_mut().collect_metrics = true;
            {
                let executor = NewVmExecutor::<Val<SC>, VmConfig>::new(config.clone());
                executor.execute(program.clone(), input_stream.clone()).unwrap();
            }
            // Run again with metrics collection disabled and measure trace generation time
            config.system_mut().collect_metrics = false;
            let start = std::time::Instant::now();
        }
    }

    let executor = NewVmExecutor::<Val<SC>, VmConfig>::new(config);

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
        metrics::gauge!("execute_and_trace_gen_time_ms").set(start.elapsed().as_millis() as f64);
    }

    ProofInputForTest {
        per_air: result.into_air_proof_input_vec(),
    }
}

type ExecuteAndProveResult<SC> = Result<VerificationDataWithFriParams<SC>, VerificationError>;

/// Executes program and runs simple STARK prover test (keygen, prove, verify).
pub fn execute_and_prove_program<SC: StarkGenericConfig, E: StarkFriEngine<SC>, VmConfig>(
    program: Program<Val<SC>>,
    input_stream: Vec<Vec<Val<SC>>>,
    config: VmConfig,
    engine: &E,
) -> ExecuteAndProveResult<SC>
where
    Val<SC>: PrimeField32,
    VmConfig: VmGenericConfig<Val<SC>> + Clone,
    VmConfig::Executor: Chip<SC>,
    VmConfig::Periphery: Chip<SC>,
{
    let span = tracing::info_span!("execute_and_prove_program").entered();
    let test_proof_input = gen_vm_program_test_proof_input(program, input_stream, config);
    let vparams = test_proof_input.run_test(engine)?;
    span.exit();
    Ok(vparams)
}
