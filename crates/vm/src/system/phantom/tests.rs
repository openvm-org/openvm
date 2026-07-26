#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::{instruction::Instruction, SystemOpcode, VmOpcode};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_instructions::{program::Program, riscv::RV64_MEMORY_AS, PhantomDiscriminant};
use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use rvr_state::{PreflightMemoryEvent, PreflightProgramEvent};

use super::PhantomExecutor;
use crate::{
    arch::{
        instructions::LocalOpcode,
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionState, PreflightExecutor,
    },
    system::phantom::{PhantomAir, PhantomChip, PhantomFiller},
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{
    arch::{
        rvr::{cuda::GpuPostflightProgram, PreflightEndpoint, PreflightEventLog},
        MemoryConfig,
    },
    system::cuda::phantom::PhantomChipGPU,
};

type F = BabyBear;

fn run_phantom_test<E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    phantom_opcode: VmOpcode,
    num_nops: usize,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
    let nop = Instruction::from_isize(phantom_opcode, 0, 0, 0, 0, 0);
    let mut state: ExecutionState<F> = ExecutionState::new(F::ZERO, F::ONE);

    for _ in 0..num_nops {
        tester.execute_with_pc(executor, arena, &nop, state.pc.as_canonical_u32());
        let new_state = tester.execution_final_state();
        assert_eq!(state.pc + F::from_usize(4), new_state.pc);
        assert_eq!(state.timestamp + F::ONE, new_state.timestamp);
        state = new_state;
    }
}

#[test]
fn test_nops_and_terminate() {
    const NUM_NOPS: usize = 100;
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();

    let mut tester = VmChipTestBuilder::default();
    let executor = PhantomExecutor::new(Default::default(), phantom_opcode);
    let chip = PhantomChip::new(PhantomFiller, tester.memory_helper());
    let air = PhantomAir {
        execution_bridge: tester.execution_bridge(),
        phantom_opcode,
    };
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, NUM_NOPS);

    run_phantom_test(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        phantom_opcode,
        NUM_NOPS,
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_phantom_tracegen() {
    use crate::{
        arch::{
            testing::{GpuChipTestBuilder, GpuTestChipHarness},
            EmptyMultiRowLayout,
        },
        system::{cuda::phantom::PhantomChipGPU, phantom::PhantomRecord},
    };

    const NUM_NOPS: usize = 100;
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
    let mut tester = GpuChipTestBuilder::default();

    let executor = PhantomExecutor::new(Default::default(), phantom_opcode);
    let air = PhantomAir {
        execution_bridge: tester.execution_bridge(),
        phantom_opcode,
    };
    let gpu_chip = PhantomChipGPU::new(tester.range_checker().device_ctx.clone());
    let cpu_chip = PhantomChip::new(PhantomFiller, tester.dummy_memory_helper());
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, NUM_NOPS);

    run_phantom_test(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        phantom_opcode,
        NUM_NOPS,
    );

    harness
        .dense_arena
        .get_record_seeker::<&mut PhantomRecord, EmptyMultiRowLayout>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

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
        F::from_u32(0x1234),
        F::from_u32(0x5678),
        0xabcd,
    );
    let program = Program::new_without_debug_infos(&[instruction.clone(), instruction.clone()], 0);
    let transcript = PreflightEventLog {
        program_log: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 2,
            },
        ],
        memory_log: vec![],
        initial_write_log: vec![],
    };
    let endpoint = PreflightEndpoint::Suspended;
    let memory_config = MemoryConfig::default();
    let d_program = GpuPostflightProgram::upload(&program, &memory_config, &device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program.upload_transcript(&transcript, endpoint).unwrap();
    let chip = PhantomChipGPU::new(device_ctx.clone());
    let _replay_ctx = chip
        .generate_proving_ctx_from_postflight(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);

    let corrupt = PreflightEventLog {
        program_log: transcript.program_log,
        memory_log: vec![PreflightMemoryEvent {
            timestamp: 1,
            address_space_and_kind: RV64_MEMORY_AS,
            pointer: 0,
            value: [0; 4],
        }],
        initial_write_log: vec![],
    };
    let (d_corrupt, d_corrupt_plan) = d_program.upload_transcript(&corrupt, endpoint).unwrap();
    chip.generate_proving_ctx_from_postflight(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 855);
}
