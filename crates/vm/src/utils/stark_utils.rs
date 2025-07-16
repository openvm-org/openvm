use itertools::{multiunzip, Itertools};
use openvm_instructions::exe::VmExe;
use openvm_stark_backend::{
    engine::VerificationData,
    prover::{hal::DeviceDataTransporter, types::AirProofRawInput},
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        setup_tracing, FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine, VerificationDataWithFriParams},
    p3_baby_bear::BabyBear,
};

use crate::{
    arch::{
        execution_mode::metered::Segment, vm::VirtualMachine, ExitCode, InsExecutorE1,
        InsExecutorE2, InstructionExecutor, MatrixRecordArena, PreflightExecutionOutput, Streams,
        VmCircuitConfig, VmExecutionConfig, VmProverConfig,
    },
    system::memory::MemoryImage,
};

pub fn air_test<VC>(config: VC, exe: impl Into<VmExe<BabyBear>>)
where
    VC: VmExecutionConfig<BabyBear>
        + VmCircuitConfig<BabyBearPoseidon2Config>
        + VmProverConfig<BabyBearPoseidon2Engine, RecordArena = MatrixRecordArena<BabyBear>>,
    <VC as VmExecutionConfig<BabyBear>>::Executor:
        InsExecutorE1<BabyBear> + InsExecutorE2<BabyBear> + InstructionExecutor<BabyBear>,
{
    air_test_with_min_segments(config, exe, Streams::default(), 1);
}

/// Executes and proves the VM and returns the final memory state.
pub fn air_test_with_min_segments<VC>(
    config: VC,
    exe: impl Into<VmExe<BabyBear>>,
    input: impl Into<Streams<BabyBear>>,
    min_segments: usize,
) -> Option<MemoryImage>
where
    VC: VmExecutionConfig<BabyBear>
        + VmCircuitConfig<BabyBearPoseidon2Config>
        + VmProverConfig<BabyBearPoseidon2Engine, RecordArena = MatrixRecordArena<BabyBear>>,
    <VC as VmExecutionConfig<BabyBear>>::Executor:
        InsExecutorE1<BabyBear> + InsExecutorE2<BabyBear> + InstructionExecutor<BabyBear>,
{
    let mut log_blowup = 1;
    while config.as_ref().max_constraint_degree > (1 << log_blowup) + 1 {
        log_blowup += 1;
    }
    let fri_params = FriParameters::new_for_testing(log_blowup);
    let (final_memory, _) =
        air_test_impl(fri_params, config, exe, input, min_segments, true).unwrap();
    final_memory
}

/// Executes and proves the VM and returns the final memory state.
/// If `debug` is true, runs the debug prover.
//
// Same implementation as VmLocalProver, but we need to do something special to run the debug prover
pub fn air_test_impl<VC>(
    fri_params: FriParameters,
    config: VC,
    exe: impl Into<VmExe<BabyBear>>,
    input: impl Into<Streams<BabyBear>>,
    min_segments: usize,
    debug: bool,
) -> eyre::Result<(
    Option<MemoryImage>,
    Vec<VerificationDataWithFriParams<BabyBearPoseidon2Config>>,
)>
where
    // NOTE: the compiler cannot figure out Val<SC>=BabyBear without the VmExecutionConfig and
    // VmCircuitConfig bounds even though VmProverConfig already includes them
    VC: VmExecutionConfig<BabyBear>
        + VmCircuitConfig<BabyBearPoseidon2Config>
        + VmProverConfig<BabyBearPoseidon2Engine, RecordArena = MatrixRecordArena<BabyBear>>,
    <VC as VmExecutionConfig<BabyBear>>::Executor:
        InsExecutorE1<BabyBear> + InsExecutorE2<BabyBear> + InstructionExecutor<BabyBear>,
{
    setup_tracing();
    let engine = BabyBearPoseidon2Engine::new(fri_params);
    let pk = config.keygen(engine.config())?;
    let vk = pk.get_vk();
    let d_pk = engine.device().transport_pk_to_device(&pk);
    let mut vm = VirtualMachine::<BabyBearPoseidon2Engine, VC>::new(engine, config, d_pk)?;
    let exe = exe.into();
    let input = input.into();
    let metered_ctx = vm.build_metered_ctx();
    let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
    let segments = vm.executor().execute_metered(
        exe.clone(),
        input.clone(),
        &executor_idx_to_air_idx,
        metered_ctx,
    )?;
    let committed_exe = vm.commit_exe(exe);
    let cached_program_trace = vm.transport_committed_exe_to_device(&committed_exe);
    vm.load_program(cached_program_trace);
    let exe = committed_exe.exe;

    let mut state = Some(vm.executor().create_initial_state(&exe, input));
    let global_airs = vm.config().create_airs().unwrap().into_airs().collect_vec();
    let mut proofs = Vec::new();
    let mut exit_code = None;
    for segment in segments {
        let Segment {
            instret_start,
            num_insns,
            trace_heights,
        } = segment;
        assert_eq!(state.as_ref().unwrap().instret, instret_start);
        let from_state = Option::take(&mut state).unwrap();
        vm.transport_init_memory_to_device(&from_state.memory);
        let PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        } = vm.execute_preflight(exe.clone(), from_state, Some(num_insns), &trace_heights)?;
        state = Some(to_state);
        exit_code = system_records.exit_code;

        let ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
        let device = vm.engine.device();
        if debug {
            let (airs, pks, proof_inputs): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(ctx.per_air.iter().map(|(air_id, air_ctx)| {
                    // Unfortunate H2D transfers
                    let cached_mains = air_ctx
                        .cached_mains
                        .iter()
                        .map(|pre| device.transport_matrix_from_device_to_host(&pre.trace))
                        .collect_vec();
                    let common_main = air_ctx
                        .common_main
                        .as_ref()
                        .map(|m| device.transport_matrix_from_device_to_host(m));
                    let public_values = air_ctx.public_values.clone();
                    let raw = AirProofRawInput {
                        cached_mains,
                        common_main,
                        public_values,
                    };
                    (
                        global_airs[*air_id].clone(),
                        pk.per_air[*air_id].clone(),
                        raw,
                    )
                }));
            vm.engine.debug(&airs, &pks, &proof_inputs);
        }
        let proof = vm.engine.prove(vm.pk(), ctx);
        proofs.push(proof);
    }
    assert!(proofs.len() >= min_segments);
    vm.verify(&vk, &proofs)
        .expect("segment proofs should verify");
    let state = state.unwrap();
    let final_memory = (exit_code == Some(ExitCode::Success as u32)).then_some(state.memory.memory);
    let vdata = proofs
        .into_iter()
        .map(|proof| VerificationDataWithFriParams {
            data: VerificationData {
                vk: vk.clone(),
                proof,
            },
            fri_params: vm.engine.fri_params(),
        })
        .collect();

    Ok((final_memory, vdata))
}
