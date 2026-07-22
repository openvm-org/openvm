use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::{BaseAluImmOpcode, BaseAluWImmOpcode};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::Rv64IConfig,
    openvm_circuit::{
        arch::{
            rvr::{
                cuda::{GpuRvrInputError, GpuRvrProgram},
                RvrPreflightEndpoint, RvrPreflightLimits, RvrPreflightTranscript,
            },
            VmExecutor,
        },
        utils::test_system_config,
    },
    openvm_instructions::{
        exe::VmExe,
        instruction::Instruction,
        program::{Program, DEFAULT_PC_STEP},
        riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
        SystemOpcode,
    },
    openvm_riscv_transpiler::BranchEqualOpcode,
};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::{
            Rv64BaseAluImmU16AdapterRecord, Rv64BaseAluWImmU16AdapterRecord, Rv64BranchAdapterAir,
            Rv64BranchAdapterExecutor, Rv64BranchAdapterFiller, RV64_WORD_U16_LIMBS,
        },
        AddICoreRecord, BranchEqualCoreAir, BranchEqualFiller, Rv64AddIChipGpu, Rv64AddIWChipGpu,
        Rv64BranchEqualAir, Rv64BranchEqualChip, Rv64BranchEqualChipGpu, Rv64BranchEqualExecutor,
    },
    openvm_circuit::arch::{
        testing::{default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout, MatrixRecordArena,
    },
    openvm_circuit_primitives::{var_range::VariableRangeCheckerChip, Chip},
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::{
        data_transporter::assert_eq_host_and_device_matrix_col_maj, prelude::SC,
    },
    openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
    openvm_stark_backend::prover::ColMajorMatrix,
    std::sync::Arc,
};

use super::{AddICoreAir, Rv64AddIChip, Rv64AddIExecutor, Rv64AddIWChip, Rv64AddIWExecutor};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor,
        Rv64BaseAluImmU16AdapterFiller, Rv64BaseAluWImmU16AdapterAir,
        Rv64BaseAluWImmU16AdapterExecutor, Rv64BaseAluWImmU16AdapterFiller,
        RV64_REGISTER_NUM_LIMBS, U16_BITS,
    },
    addi::AddICoreCols,
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
    AddIFiller, Rv64AddIAir, Rv64AddIWAir,
};

const MAX_INS_CAPACITY: usize = 128;
const NONCANONICAL_ZERO: [u32; BLOCK_FE_WIDTH] = [
    1 << U16_BITS,
    (1 << U16_BITS) - 1,
    (1 << U16_BITS) - 1,
    (1 << U16_BITS) - 1,
];
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AddIExecutor, Rv64AddIAir, Rv64AddIChip<F>>;
type WHarness = TestChipHarness<F, Rv64AddIWExecutor, Rv64AddIWAir, Rv64AddIWChip<F>>;
#[cfg(feature = "cuda")]
type GpuWHarness =
    GpuTestChipHarness<F, Rv64AddIWExecutor, Rv64AddIWAir, Rv64AddIWChipGpu, Rv64AddIWChip<F>>;
#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv64AddIExecutor, Rv64AddIAir, Rv64AddIChipGpu, Rv64AddIChip<F>>;
#[cfg(feature = "cuda")]
type GpuBranchHarness = GpuTestChipHarness<
    F,
    Rv64BranchEqualExecutor,
    Rv64BranchEqualAir,
    Rv64BranchEqualChipGpu,
    Rv64BranchEqualChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddIAir, Rv64AddIExecutor, Rv64AddIChip<F>) {
    let air = Rv64AddIAir::new(
        Rv64BaseAluImmU16AdapterAir::new(execution_bridge, memory_bridge),
        AddICoreAir::new(
            range_checker_chip.bus(),
            BaseAluImmOpcode::CLASS_OFFSET,
            BaseAluImmOpcode::ADDI as usize,
        ),
    );
    let executor = Rv64AddIExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor,
        BaseAluImmOpcode::CLASS_OFFSET,
        BaseAluImmOpcode::ADDI as usize,
    );
    let chip = Rv64AddIChip::new(
        AddIFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_w_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddIWAir, Rv64AddIWExecutor, Rv64AddIWChip<F>) {
    let air = Rv64AddIWAir::new(
        Rv64BaseAluWImmU16AdapterAir::new(execution_bridge, memory_bridge, range_checker.bus()),
        AddICoreAir::new(
            range_checker.bus(),
            BaseAluWImmOpcode::CLASS_OFFSET,
            BaseAluWImmOpcode::ADDIW as usize,
        ),
    );
    let executor = Rv64AddIWExecutor::new(
        Rv64BaseAluWImmU16AdapterExecutor,
        BaseAluWImmOpcode::CLASS_OFFSET,
        BaseAluWImmOpcode::ADDIW as usize,
    );
    let chip = Rv64AddIWChip::new(
        AddIFiller::new(
            Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
            range_checker,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_w_harness(tester: &VmChipTestBuilder<F>) -> WHarness {
    let (air, executor, chip) = create_w_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
    );
    WHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_cuda_w_harness(tester: &GpuChipTestBuilder) -> GpuWHarness {
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_w_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64AddIWChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 8)
}

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64AddIChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 64)
}

#[cfg(feature = "cuda")]
fn create_cuda_branch_harness(tester: &GpuChipTestBuilder) -> GpuBranchHarness {
    let air = Rv64BranchEqualAir::new(
        Rv64BranchAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
    );
    let executor = Rv64BranchEqualExecutor::new(
        Rv64BranchAdapterExecutor,
        BranchEqualOpcode::CLASS_OFFSET,
        DEFAULT_PC_STEP,
    );
    let cpu_chip = Rv64BranchEqualChip::new(
        BranchEqualFiller::new(
            Rv64BranchAdapterFiller,
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64BranchEqualChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 32)
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let (imm, c) = if let Some(c) = c {
        ((u64::from_le_bytes(c) & 0xFFFFFF) as usize, c)
    } else {
        generate_rv64_is_type_immediate(rng)
    };

    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        b,
        c,
        Some(imm),
        BaseAluImmOpcode::ADDI.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let rs1 = u64::from_le_bytes(b);
    let signed_imm = ((imm as u32) << 20) as i32 >> 20;
    let expected = rs1.wrapping_add(signed_imm as i64 as u64);
    assert_eq!(
        expected.to_le_bytes().map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
}

fn set_and_execute_w<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    rs1: u64,
    imm: usize,
) {
    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        rs1.to_le_bytes(),
        (imm as u64).to_le_bytes(),
        Some(imm),
        BaseAluWImmOpcode::ADDIW.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let signed_imm = ((imm as u32) << 20) as i32 >> 20;
    let expected = (rs1 as u32).wrapping_add(signed_imm as u32) as i32 as i64 as u64;
    assert_eq!(
        expected.to_le_bytes().map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    );
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_rv64_addi_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            None,
            None,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv64_addiw_boundaries_and_sign_extension() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_w_harness(&tester);

    for (rs1, imm) in [
        (0x0000_0000_0000_0000u64, 0x00_0000usize),
        (0x0000_0000_0000_0000, 0x00_07ff),
        (0x0000_0000_0000_0000, 0x00_fff800),
        (0x0000_0000_0000_0000, 0x00_ffffff),
        (0x0000_0000_7fff_ffff, 0x00_000001),
    ] {
        set_and_execute_w(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            rs1,
            imm,
        );
    }

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_addiw_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_w_harness(&tester);

    for (rs1, imm) in [
        (0x0000_0000_0000_0000u64, 0x00_0000usize),
        (0x0000_0000_0000_0000, 0x00_07ff),
        (0x0000_0000_0000_0000, 0x00_fff800),
        (0x0000_0000_0000_0000, 0x00_ffffff),
        (0x0000_0000_7fff_ffff, 0x00_000001),
    ] {
        set_and_execute_w(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            rs1,
            imm,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluWImmU16AdapterRecord,
        &'a mut AddICoreRecord<RV64_WORD_U16_LIMBS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluWImmU16AdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_addi_tracegen_from_rvr_transcript() {
    const ITERATIONS: usize = 32;
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let addi = |rd: usize, rs1: usize, immediate: usize| {
        Instruction::<F>::from_usize(
            BaseAluImmOpcode::ADDI.global_opcode(),
            [
                reg(rd),
                reg(rs1),
                immediate,
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        )
    };
    let instructions = vec![
        addi(1, 0, 0),
        addi(2, 0, ITERATIONS),
        addi(1, 1, 1),
        Instruction::from_isize(
            BranchEqualOpcode::BNE.global_opcode(),
            reg(1) as isize,
            reg(2) as isize,
            -4,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        addi(3, 1, 0xff_ffff),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::from(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config).unwrap();
    let execution = executor
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(
            Vec::<Vec<u8>>::new(),
            RvrPreflightLimits::new(2 * ITERATIONS + 4, 4 * ITERATIONS + 6),
        )
        .unwrap();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_harness(&tester);
    let mut branch_harness = create_cuda_branch_harness(&tester);
    for (pc, instruction) in instructions[..2].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            (pc as u32) * 4,
        );
    }
    for _ in 0..ITERATIONS {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instructions[2],
            8,
        );
        tester.execute_with_pc(
            &mut branch_harness.executor,
            &mut branch_harness.dense_arena,
            &instructions[3],
            12,
        );
    }
    tester.execute_with_pc(
        &mut harness.executor,
        &mut harness.dense_arena,
        &instructions[4],
        16,
    );

    type Record<'a> = (
        &'a mut Rv64BaseAluImmU16AdapterRecord,
        &'a mut AddICoreRecord<BLOCK_FE_WIDTH>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmU16AdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(
        d_replay_plan
            .opcode_range(BaseAluImmOpcode::ADDI.global_opcode())
            .len(),
        ITERATIONS + 3
    );
    let mismatched_program = GpuRvrProgram::upload(&program, device_ctx).unwrap();
    assert!(matches!(
        harness.gpu_chip.generate_proving_ctx_from_rvr(
            &mismatched_program,
            &d_transcript,
            &d_replay_plan,
        ),
        Err(GpuRvrInputError::ProgramMismatch)
    ));
    let (_other_transcript, other_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert!(matches!(
        harness
            .gpu_chip
            .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &other_plan,),
        Err(GpuRvrInputError::SegmentMismatch)
    ));
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();

    let mut corrupt_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    corrupt_transcript.memory_log[1].value[0] ^= 1;
    let (d_corrupt, d_corrupt_plan) = d_program
        .upload_transcript(&corrupt_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let corrupt_chip = Rv64AddIChipGpu::new(corrupt_range_checker, tester.timestamp_max_bits());
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 9);

    let legacy_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let legacy_chip =
        Rv64AddIChipGpu::new(legacy_range_checker.clone(), tester.timestamp_max_bits());
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    let legacy_counts = legacy_range_checker.count.to_host_on(device_ctx).unwrap();
    assert_eq!(replay_counts, legacy_counts);
    let raw_count = |count: &F| {
        const { assert!(std::mem::size_of::<F>() == std::mem::size_of::<u32>()) };
        // The CUDA range-check buffer is typed as `F` for shared ownership but
        // kernels update it as an atomic `u32` histogram.
        unsafe { *(std::ptr::from_ref(count).cast::<u32>()) }
    };
    assert_eq!(
        replay_counts.iter().map(raw_count).sum::<u32>(),
        (ITERATIONS as u32 + 3) * 9
    );

    let expected_trace =
        <Rv64AddIChip<F> as Chip<MatrixRecordArena<F>, CpuBackend<SC>>>::generate_proving_ctx(
            &harness.cpu_chip,
            harness.matrix_arena,
        )
        .common_main;
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &replay_ctx.common_main, device_ctx);

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .load(
            branch_harness.air,
            branch_harness.gpu_chip,
            branch_harness.dense_arena,
        )
        .finalize()
        .simple_test()
        .expect("RVR transcript replay proof failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rv64_addi_rs1_memory_binding_negative_test() {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some([0; RV64_REGISTER_NUM_LIMBS]),
        Some([0; RV64_REGISTER_NUM_LIMBS]),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut AddICoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        // These limbs represent 2^64, so ADDI 0 still produces zero and all core
        // constraints hold. Only the rs1 memory-read interaction rejects the row.
        cols.rs1 = NONCANONICAL_ZERO.map(F::from_u32);
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}
