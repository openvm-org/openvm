use itertools::multiunzip;
use openvm_instructions::{exe::VmExe, program::Program};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::{
        cpu::{CpuBackend, CpuDevice},
        hal::DeviceDataTransporter,
    },
    verifier::VerificationError,
    Chip,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        setup_tracing, FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine, VerificationDataWithFriParams},
    p3_baby_bear::BabyBear,
    utils::ProofInputForTest,
};

#[cfg(feature = "bench-metrics")]
use crate::arch::vm::VmExecutor;
use crate::{
    arch::{
        vm::VirtualMachine, InsExecutorE1, InstructionExecutor, MatrixRecordArena, Streams,
        VmCircuitConfig, VmConfig, VmExecutionConfig, VmProverConfig,
    },
    system::memory::MemoryImage,
};

// pub fn air_test<VC>(config: VC, exe: impl Into<VmExe<BabyBear>>)
// where
//     VC: VmProverConfig<BabyBearPoseidon2Config, CpuBackend<BabyBearPoseidon2Config>>,
//     VC::Executor: InsExecutorE1<BabyBear> + InstructionExecutor<BabyBear, VC::RecordArena>,
// {
//     air_test_with_min_segments(config, exe, Streams::default(), 1);
// }

// /// Executes and proves the VM and returns the final memory state.
// pub fn air_test_with_min_segments<VC>(
//     config: VC,
//     exe: impl Into<VmExe<BabyBear>>,
//     input: impl Into<Streams<BabyBear>>,
//     min_segments: usize,
// ) -> Option<MemoryImage>
// where
//     VC: VmProverConfig<BabyBearPoseidon2Config, CpuBackend<BabyBearPoseidon2Config>>,
//     VC::Executor: InsExecutorE1<BabyBear> + InstructionExecutor<BabyBear, VC::RecordArena>,
// {
//     air_test_impl(config, exe, input, min_segments, true)
// }

/// Executes and proves the VM and returns the final memory state.
/// If `debug` is true, runs the debug prover.
pub fn air_test_impl<VC>(
    config: VC,
    exe: impl Into<VmExe<BabyBear>>,
    input: impl Into<Streams<BabyBear>>,
    min_segments: usize,
    debug: bool,
) -> Option<MemoryImage>
where
    // NOTE: the compiler cannot figure out Val<SC>=BabyBear without the VmExecutionConfig and
    // VmCircuitConfig bounds even though VmProverConfig already includes them
    VC: VmExecutionConfig<BabyBear>
        + VmCircuitConfig<BabyBearPoseidon2Config>
        + VmProverConfig<
            BabyBearPoseidon2Config,
            CpuBackend<BabyBearPoseidon2Config>,
            RecordArena = MatrixRecordArena<BabyBear>,
        >,
    <VC as VmExecutionConfig<BabyBear>>::Executor:
        InsExecutorE1<BabyBear> + InstructionExecutor<BabyBear>,
{
    setup_tracing();
    let mut log_blowup = 1;
    while config.as_ref().max_constraint_degree > (1 << log_blowup) + 1 {
        log_blowup += 1;
    }
    let engine = BabyBearPoseidon2Engine::new(FriParameters::new_for_testing(log_blowup));
    let pk_host = config.keygen(engine.config()).unwrap();
    let vk = pk_host.get_vk();
    let pk_device = engine.device().transport_pk_to_device(&pk_host);
    let mut vm =
        VirtualMachine::<BabyBearPoseidon2Engine, VC>::new(engine, config, pk_device).unwrap();
    let exe = exe.into();
    let input = input.into();
    let segments = vm
        .executor()
        .execute_metered(
            exe.clone(),
            input.clone(),
            &vk.total_widths(),
            &vk.num_interactions(),
        )
        .unwrap();
    vm.set_main_widths(vk.main_widths());
    let mut result = vm.execute_and_generate(exe, input, &segments).unwrap();
    let final_memory = Option::take(&mut result.final_memory);
    let global_airs = vm.config().create_circuit().unwrap().into_airs();
    if debug {
        for proof_input in &result.per_segment {
            let (airs, pks, air_proof_inputs): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(proof_input.per_air.iter().map(|(air_id, air_proof_input)| {
                    (
                        global_airs[*air_id].clone(),
                        pk.per_air[*air_id].clone(),
                        air_proof_input.clone(),
                    )
                }));
            vm.engine.debug(&airs, &pks, &air_proof_inputs);
        }
    }
    let proofs = vm.prove(&pk, result);

    assert!(proofs.len() >= min_segments);
    vm.verify(&pk.get_vk(), proofs)
        .expect("segment proofs should verify");
    final_memory
}

// /// Generates the VM STARK circuit, in the form of AIRs and traces, but does not
// /// do any proving. Output is the payload of everything the prover needs.
// ///
// /// The output AIRs and traces are sorted by height in descending order.
// pub fn gen_vm_program_test_proof_input<VC, E>(
//     program: Program<Val<E::SC>>,
//     input_stream: impl Into<Streams<Val<E::SC>>> + Clone,
//     #[allow(unused_mut)] mut config: VC,
// ) -> ProofInputForTest<E::SC>
// where
//     E: StarkFriEngine,
//     Val<E::SC>: PrimeField32,
//     VC: VmProverConfig<E::SC, E::PB>,
//     VC::Executor: InsExecutorE1<Val<E::SC>>,
// {
//     let program_exe = VmExe::new(program);
//     let input = input_stream.into();

//     let airs = config.create_chip_complex().unwrap().airs();
//     let engine = E::new(FriParameters::new_for_testing(1));
//     let vm = VirtualMachine::new(engine, config.clone());

//     let pk = vm.keygen();
//     let vk = pk.get_vk();
//     let segments = vm
//         .executor
//         .execute_metered(
//             program_exe.clone(),
//             input.clone(),
//             &vk.total_widths(),
//             &vk.num_interactions(),
//         )
//         .unwrap();

//     cfg_if::cfg_if! {
//         if #[cfg(feature = "bench-metrics")] {
//             // Run once with metrics collection enabled, which can improve runtime performance
//             config.as_mut().profiling = true;
//             {
//                 let executor = VmExecutor::<Val<E::SC>, VC>::new(config.clone()).unwrap();
//                 executor.execute(program_exe.clone(), input.clone(), &segments).unwrap();
//             }
//             // Run again with metrics collection disabled and measure trace generation time
//             config.as_mut().profiling = false;
//             let start = std::time::Instant::now();
//         }
//     }
//     let mut result = vm
//         .executor
//         .execute_and_generate(program_exe, input, &segments)
//         .unwrap();

//     assert_eq!(
//         result.per_segment.len(),
//         1,
//         "only proving one segment for now"
//     );

//     let result = result.per_segment.pop().unwrap();
//     #[cfg(feature = "bench-metrics")]
//     metrics::gauge!("execute_and_trace_gen_time_ms").set(start.elapsed().as_millis() as f64);
//     // Filter out unused AIRS (where trace is empty)
//     let (used_airs, per_air) = result
//         .per_air
//         .into_iter()
//         .map(|(air_id, x)| (airs[air_id].clone(), x))
//         .unzip();
//     ProofInputForTest {
//         airs: used_airs,
//         per_air,
//     }
// }

// type ExecuteAndProveResult<SC> = Result<VerificationDataWithFriParams<SC>, VerificationError>;

// /// Executes program and runs simple STARK prover test (keygen, prove, verify).
// pub fn execute_and_prove_program<SC, E, VC>(
//     program: Program<Val<SC>>,
//     input_stream: impl Into<Streams<Val<SC>>> + Clone,
//     config: VC,
//     engine: &E,
// ) -> ExecuteAndProveResult<SC>
// where
//     SC: StarkGenericConfig,
//     E: StarkFriEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
//     Val<E::SC>: PrimeField32,
//     VC: VmProverConfig<SC, E::PB>,
//     VC::Executor: InsExecutorE1<Val<SC>>,
// {
//     let span = tracing::info_span!("execute_and_prove_program").entered();
//     let test_proof_input = gen_vm_program_test_proof_input::<_, E>(program, input_stream,
// config);     let vparams = test_proof_input.run_test(engine)?;
//     span.exit();
//     Ok(vparams)
// }
