use openvm_instructions::exe::VmExe;
use openvm_stark_backend::{
    config::{Com, Val},
    engine::VerificationData,
    p3_field::PrimeField32,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, setup_tracing, FriParameters},
    engine::{StarkFriEngine, VerificationDataWithFriParams},
    p3_baby_bear::BabyBear,
};

use crate::{
    arch::{
        debug_proving_ctx, execution_mode::Segment, vm::VirtualMachine, Executor, ExitCode,
        MeteredExecutor, PreflightExecutionOutput, PreflightExecutor, Streams, VmBuilder,
        VmCircuitConfig, VmConfig, VmExecutionConfig,
    },
    system::memory::{MemoryImage, CHUNK},
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        pub use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine as TestStarkEngine, chip::cpu_proving_ctx_to_gpu};
        use crate::arch::DenseRecordArena;
        pub type TestRecordArena = DenseRecordArena;
    } else {
        pub use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine as TestStarkEngine;
        use crate::arch::MatrixRecordArena;
        pub type TestRecordArena = MatrixRecordArena<BabyBear>;
    }
}
type RA = TestRecordArena;

// NOTE on trait bounds: the compiler cannot figure out Val<SC>=BabyBear without the
// VmExecutionConfig and VmCircuitConfig bounds even though VmProverBuilder already includes them.
// The compiler also seems to need the extra VC even though VC=VB::VmConfig
pub fn air_test<VB, VC>(builder: VB, config: VC, exe: impl Into<VmExe<BabyBear>>)
where
    VB: VmBuilder<TestStarkEngine, VmConfig = VC, RecordArena = RA>,
    VC: VmExecutionConfig<BabyBear>
        + VmCircuitConfig<BabyBearPoseidon2Config>
        + VmConfig<BabyBearPoseidon2Config>,
    <VC as VmExecutionConfig<BabyBear>>::Executor:
        Executor<BabyBear> + MeteredExecutor<BabyBear> + PreflightExecutor<BabyBear, RA>,
{
    air_test_with_min_segments(builder, config, exe, Streams::default(), 1);
}

/// Executes and proves the VM and returns the final memory state.
pub fn air_test_with_min_segments<VB, VC>(
    builder: VB,
    config: VC,
    exe: impl Into<VmExe<BabyBear>>,
    input: impl Into<Streams<BabyBear>>,
    min_segments: usize,
) -> Option<MemoryImage>
where
    VB: VmBuilder<TestStarkEngine, VmConfig = VC, RecordArena = RA>,
    VC: VmExecutionConfig<BabyBear>
        + VmCircuitConfig<BabyBearPoseidon2Config>
        + VmConfig<BabyBearPoseidon2Config>,
    <VC as VmExecutionConfig<BabyBear>>::Executor:
        Executor<BabyBear> + MeteredExecutor<BabyBear> + PreflightExecutor<BabyBear, RA>,
{
    let mut log_blowup = 1;
    while config.as_ref().max_constraint_degree > (1 << log_blowup) + 1 {
        log_blowup += 1;
    }
    let fri_params = FriParameters::new_for_testing(log_blowup);
    #[cfg(feature = "cuda")]
    let debug = std::env::var("OPENVM_SKIP_DEBUG") != Result::Ok(String::from("1"));
    #[cfg(not(feature = "cuda"))]
    let debug = true;
    let (final_memory, _) = air_test_impl::<TestStarkEngine, VB>(
        fri_params,
        builder,
        config,
        exe,
        input,
        min_segments,
        debug,
    )
    .unwrap();
    final_memory
}

/// Executes and proves the VM and returns the final memory state.
/// If `debug` is true, runs the debug prover.
//
// Same implementation as VmLocalProver, but we need to do something special to run the debug prover
#[allow(clippy::type_complexity)]
pub fn air_test_impl<E, VB>(
    fri_params: FriParameters,
    builder: VB,
    config: VB::VmConfig,
    exe: impl Into<VmExe<Val<E::SC>>>,
    input: impl Into<Streams<Val<E::SC>>>,
    min_segments: usize,
    debug: bool,
) -> eyre::Result<(
    Option<MemoryImage>,
    Vec<VerificationDataWithFriParams<E::SC>>,
)>
where
    E: StarkFriEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    Com<E::SC>: AsRef<[Val<E::SC>; CHUNK]> + From<[Val<E::SC>; CHUNK]>,
{
    setup_tracing();
    let engine = E::new(fri_params);
    let (mut vm, pk) = VirtualMachine::<E, VB>::new_with_keygen(engine, builder, config)?;
    let vk = pk.get_vk();
    let exe = exe.into();
    let input = input.into();
    let metered_ctx = vm.build_metered_ctx();
    let (segments, _) = vm
        .metered_interpreter(&exe)?
        .execute_metered(input.clone(), metered_ctx)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program_trace);
    let mut preflight_interpreter = vm.preflight_interpreter(&exe)?;

    let mut state = Some(vm.create_initial_state(&exe, input));
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
        } = vm.execute_preflight(
            &mut preflight_interpreter,
            from_state,
            Some(num_insns),
            &trace_heights,
        )?;
        state = Some(to_state);
        exit_code = system_records.exit_code;

        let ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
        if debug {
            debug_proving_ctx(&vm, &pk, &ctx);
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
