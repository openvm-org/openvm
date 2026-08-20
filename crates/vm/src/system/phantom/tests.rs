use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::{
    instruction::Instruction, PhantomDiscriminant, SysPhantom, SystemOpcode, VmOpcode,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_instructions::{program::Program, riscv::MEMORY_AS};
use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;

use super::{generate_trace_from_postflight, NopPhantomExecutor, PhantomExecutor};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{
    arch::{cuda::postflight::GpuPostflightProgram, PreflightMemoryLog},
    system::cuda::phantom::PhantomChipGPU,
};
use crate::{
    arch::{
        instructions::LocalOpcode,
        testing::{TestBuilder, TestChipHarness, TestPreflight, VmChipTestBuilder},
        Executor, MemoryConfig, PhantomSubExecutor, Postflight, PreflightHistory,
        PreflightMemoryEvent, PreflightProgramEvent, Streams,
    },
    system::{memory::online::GuestMemory, phantom::PhantomAir},
};

type F = BabyBear;

struct CountingPhantomExecutor(Arc<AtomicUsize>);

impl PhantomSubExecutor for CountingPhantomExecutor {
    fn phantom_execute(
        &self,
        _memory: &GuestMemory,
        _streams: &mut Streams,
        _rng: &mut StdRng,
        _discriminant: PhantomDiscriminant,
        _a: u32,
        _b: u32,
        _c_upper: u16,
    ) -> eyre::Result<()> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn run_phantom_test<E>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    preflight: &mut TestPreflight,
    phantom_opcode: VmOpcode,
    num_nops: usize,
) where
    E: Executor<F> + Clone,
{
    let nop = Instruction::from_isize(phantom_opcode, 0, 0, 0, 0, 0);
    let mut pc = F::ZERO;

    for _ in 0..num_nops {
        tester.execute_with_pc(executor, preflight, &nop, pc.as_canonical_u32());
        let new_state = tester.execution_final_state();
        assert_eq!(pc + F::from_usize(4), new_state.pc);
        assert_eq!(F::TWO, new_state.timestamp);
        pc = new_state.pc;
    }
}

#[test]
fn test_nops_and_terminate() {
    const NUM_NOPS: usize = 100;
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();

    let mut tester = VmChipTestBuilder::default();
    let mut phantom_executors: FxHashMap<PhantomDiscriminant, Arc<dyn PhantomSubExecutor>> =
        FxHashMap::default();
    phantom_executors.insert(
        PhantomDiscriminant(SysPhantom::Nop as u16),
        Arc::new(NopPhantomExecutor),
    );
    let executor = PhantomExecutor::new(phantom_executors);
    let chip = ();
    let air = PhantomAir {
        execution_bridge: tester.execution_bridge(),
        phantom_opcode,
    };
    let mut harness =
        TestChipHarness::with_capacity(executor, air, chip, NUM_NOPS, |_, postflight| {
            generate_trace_from_postflight(postflight)
        });

    run_phantom_test(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        phantom_opcode,
        NUM_NOPS,
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn postflight_trace_does_not_replay_callbacks() {
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
    let discriminant = PhantomDiscriminant(0x7ffe);
    let callback_count = Arc::new(AtomicUsize::new(0));
    let mut phantom_executors: FxHashMap<PhantomDiscriminant, Arc<dyn PhantomSubExecutor>> =
        FxHashMap::default();
    phantom_executors.insert(
        discriminant,
        Arc::new(CountingPhantomExecutor(callback_count.clone())),
    );

    let mut tester = VmChipTestBuilder::default();
    let executor = PhantomExecutor::new(phantom_executors);
    let chip = ();
    let air = PhantomAir {
        execution_bridge: tester.execution_bridge(),
        phantom_opcode,
    };
    let mut harness: TestChipHarness<F, _, _, _> =
        TestChipHarness::with_capacity(executor, air, chip, 3, |_, postflight| {
            generate_trace_from_postflight(postflight)
        });
    let instructions = [
        Instruction::phantom(discriminant, 0x1234_u16, 0x5678_u16, 0x1234_u16),
        Instruction::phantom(discriminant, 0x8765_u16, 0x4321_u16, 0x4321_u16),
        Instruction::phantom(discriminant, 0_u16, 1_u16, 0_u16),
    ];
    for (index, instruction) in instructions.iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.preflight,
            instruction,
            index as u32 * 4,
        );
    }
    assert_eq!(callback_count.load(Ordering::Relaxed), instructions.len());

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
    assert_eq!(callback_count.load(Ordering::Relaxed), instructions.len());
}

#[test]
fn postflight_trace_rejects_phantom_history_with_memory_events() {
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
    let instruction = Instruction::phantom(
        PhantomDiscriminant(0x7ffe),
        0x1234_u16,
        0x5678_u16,
        0x1234_u16,
    );
    let program = openvm_instructions::program::Program::new_without_debug_infos(
        &[instruction, Instruction::from_usize(phantom_opcode, [0; 3])],
        0,
    );
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 2,
            },
        ],
        memory: crate::arch::PreflightMemoryLog {
            accesses: vec![PreflightMemoryEvent {
                timestamp: 1,
                address_space_and_kind: 1,
                pointer: 0,
                value: [0; 4],
            }],
            ..Default::default()
        },
    };
    let memory_config = MemoryConfig::default();
    let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();
    let error = generate_trace_from_postflight::<F>(&postflight).unwrap_err();

    assert!(error.to_string().contains("left 1 memory events unread"));
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_phantom_tracegen() {
    use crate::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness};

    const NUM_NOPS: usize = 100;
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
    let mut tester = GpuChipTestBuilder::default();

    let mut phantom_executors: FxHashMap<PhantomDiscriminant, Arc<dyn PhantomSubExecutor>> =
        FxHashMap::default();
    phantom_executors.insert(
        PhantomDiscriminant(SysPhantom::Nop as u16),
        Arc::new(NopPhantomExecutor),
    );
    let executor = PhantomExecutor::new(phantom_executors);
    let air = PhantomAir {
        execution_bridge: tester.execution_bridge(),
        phantom_opcode,
    };
    let gpu_chip = PhantomChipGPU::new(tester.range_checker().device_ctx.clone());
    let cpu_chip = ();
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, NUM_NOPS)
            .with_trace_generators(
                |_, postflight| generate_trace_from_postflight(postflight),
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            );

    run_phantom_test(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        phantom_opcode,
        NUM_NOPS,
    );

    tester
        .build()
        .load_gpu_harness(harness)
        .simple_test()
        .expect("Verification failed");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_phantom_preflight_replay() {
    let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
    // This discriminant is deliberately not registered with the RV64 executor.
    // GPU replay must treat it as an execution-bus operand and never invoke a callback.
    let instruction = Instruction::phantom(
        PhantomDiscriminant(0x7ffe),
        0x1234_u16,
        0x5678_u16,
        0xabcd_u16,
    );
    let program = Program::new_without_debug_infos(&[instruction.clone(), instruction.clone()], 0);
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 2,
            },
            PreflightProgramEvent {
                pc: 8,
                timestamp: 3,
            },
        ],
        memory: PreflightMemoryLog::default(),
    };
    let memory_config = MemoryConfig::default();
    let d_program = GpuPostflightProgram::upload(&program, &memory_config, &device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_isolated_history_for_test(&program, &history)
        .unwrap();
    let chip = PhantomChipGPU::new(device_ctx.clone());
    let _replay_ctx = chip
        .generate_proving_ctx_from_postflight(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);

    let corrupt = PreflightHistory {
        program: history.program,
        memory: PreflightMemoryLog {
            accesses: vec![PreflightMemoryEvent {
                timestamp: 1,
                address_space_and_kind: MEMORY_AS,
                pointer: 0,
                value: [0; 4],
            }],
            ..Default::default()
        },
    };
    let (d_corrupt, d_corrupt_plan) = d_program
        .upload_isolated_history_for_test(&program, &corrupt)
        .unwrap();
    chip.generate_proving_ctx_from_postflight(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 855);
}
