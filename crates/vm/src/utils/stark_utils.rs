use std::borrow::Borrow;

use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    verifier::VerificationError,
    Chip,
};
use ax_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        setup_tracing, FriParameters,
    },
    engine::{ProofInputForTest, StarkFriEngine, VerificationDataWithFriParams},
};
use axvm_instructions::{exe::AxVmExe, program::Program};
use ax_stark_sdk::p3_baby_bear::BabyBear;
use ax_stark_backend::p3_field::{AbstractField, PrimeField32};

use crate::{
    arch::{
        vm::{VirtualMachine, VmExecutor},
        ExitCode, Streams, VmConfig, VmMemoryState, CONNECTOR_AIR_ID,
    },
    system::connector::VmConnectorPvs,
};

pub fn air_test<VC>(config: VC, exe: impl Into<AxVmExe<BabyBear>>)
where
    VC: VmConfig<BabyBear>,
    VC::Executor: Chip<BabyBearPoseidon2Config>,
    VC::Periphery: Chip<BabyBearPoseidon2Config>,
{
    air_test_with_min_segments(config, exe, Streams::default(), 1);
}

/// Executes the VM and returns the final memory state.
pub fn air_test_with_min_segments<VC>(
    config: VC,
    exe: impl Into<AxVmExe<BabyBear>>,
    input: impl Into<Streams<BabyBear>>,
    min_segments: usize,
) -> Option<VmMemoryState<BabyBear>>
where
    VC: VmConfig<BabyBear>,
    VC::Executor: Chip<BabyBearPoseidon2Config>,
    VC::Periphery: Chip<BabyBearPoseidon2Config>,
{
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
pub fn new_air_test_with_min_segments<VC>(
    config: VC,
    exe: impl Into<AxVmExe<BabyBear>>,
    input: impl Into<Streams<BabyBear>>,
    min_segments: usize,
    always_prove: bool,
) -> Option<VmMemoryState<BabyBear>>
where
    VC: VmConfig<BabyBear>,
    VC::Executor: Chip<BabyBearPoseidon2Config>,
    VC::Periphery: Chip<BabyBearPoseidon2Config>,
{
    setup_tracing();
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let vm = VirtualMachine::new(engine, config);
    let pk = vm.keygen();
    let mut result = vm.execute_and_generate(exe, input).unwrap();
    let connector_pvs = &result.per_segment.last().unwrap().per_air[CONNECTOR_AIR_ID]
        .1
        .raw
        .public_values[..];
    let pvs: &VmConnectorPvs<_> = connector_pvs.borrow();
    assert_eq!(
        pvs.exit_code,
        AbstractField::from_canonical_u32(ExitCode::Success as u32),
        "Runtime did not exit successfully"
    );
    let final_memory = result.final_memory.take();
    if std::env::var("RUN_AIR_TEST_PROVING").is_ok() || always_prove {
        let proofs = vm.prove(&pk, result);

        assert!(proofs.len() >= min_segments);
        vm.verify(&pk.get_vk(), proofs)
            .expect("segment proofs should verify");
    }
    final_memory
}

// TODO[jpw]: this should be deleted once tests switch to new API
/// Generates the VM STARK circuit, in the form of AIRs and traces, but does not
/// do any proving. Output is the payload of everything the prover needs.
///
/// The output AIRs and traces are sorted by height in descending order.
pub fn gen_vm_program_test_proof_input<SC: StarkGenericConfig, VC>(
    program: Program<Val<SC>>,
    input_stream: impl Into<Streams<Val<SC>>> + Clone,
    #[allow(unused_mut)] mut config: VC,
) -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
    VC: VmConfig<Val<SC>> + Clone,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "bench-metrics")] {
            // Run once with metrics collection enabled, which can improve runtime performance
            config.system_mut().collect_metrics = true;
            {
                let executor = VmExecutor::<Val<SC>, VC>::new(config.clone());
                executor.execute(program.clone(), input_stream.clone()).unwrap();
            }
            // Run again with metrics collection disabled and measure trace generation time
            config.system_mut().collect_metrics = false;
            let start = std::time::Instant::now();
        }
    }

    let executor = VmExecutor::<Val<SC>, VC>::new(config);

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
pub fn execute_and_prove_program<SC: StarkGenericConfig, E: StarkFriEngine<SC>, VC>(
    program: Program<Val<SC>>,
    input_stream: impl Into<Streams<Val<SC>>> + Clone,
    config: VC,
    engine: &E,
) -> ExecuteAndProveResult<SC>
where
    Val<SC>: PrimeField32,
    VC: VmConfig<Val<SC>> + Clone,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    let span = tracing::info_span!("execute_and_prove_program").entered();
    let test_proof_input = gen_vm_program_test_proof_input(program, input_stream, config);
    let vparams = test_proof_input.run_test(engine)?;
    span.exit();
    Ok(vparams)
}
