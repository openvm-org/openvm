use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
    BLOCK_FE_WIDTH,
};
use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::LessThanImmOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::Rv64IConfig,
    openvm_circuit::{
        arch::{
            rvr::{
                cuda::GpuRvrProgram, RvrPreflightEndpoint, RvrPreflightLimits,
                RvrPreflightTranscript,
            },
            MatrixRecordArena, VmExecutor,
        },
        utils::test_system_config,
    },
    openvm_circuit_primitives::Chip,
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::{
        data_transporter::{
            assert_eq_host_and_device_matrix_col_maj, transport_matrix_d2h_row_major,
        },
        prelude::SC,
    },
    openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
    openvm_instructions::{
        exe::VmExe,
        instruction::Instruction,
        program::Program,
        riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
        SystemOpcode,
    },
    openvm_stark_backend::{p3_field::PrimeField32, prover::ColMajorMatrix},
};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::Rv64BaseAluImmU16AdapterRecord, LessThanImmCoreRecord, Rv64LessThanImmChipGpu,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    LessThanImmCoreAir, LessThanImmCoreCols, LessThanImmFiller, Rv64LessThanImmAir,
    Rv64LessThanImmChip, Rv64LessThanImmExecutor,
};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor,
        Rv64BaseAluImmU16AdapterFiller, U16_BITS,
    },
    test_utils::rv64_rand_write_register_or_imm,
};

type F = BabyBear;
type Harness =
    TestChipHarness<F, Rv64LessThanImmExecutor, Rv64LessThanImmAir, Rv64LessThanImmChip<F>>;

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let air = Rv64LessThanImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        LessThanImmCoreAir::new(range_checker.bus(), LessThanImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64LessThanImmExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor::new(),
        LessThanImmOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LessThanImmChip::new(
        LessThanImmFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker),
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, 64)
}

fn encode_i12(imm: i16) -> usize {
    debug_assert!((-2048..=2047).contains(&imm));
    (imm as i32 as u32 & 0x00ff_ffff) as usize
}

fn expected(opcode: LessThanImmOpcode, source: u64, imm: i16) -> bool {
    match opcode {
        LessThanImmOpcode::SLTI => (source as i64) < i64::from(imm),
        LessThanImmOpcode::SLTIU => source < (imm as i64 as u64),
    }
}

#[test]
fn rv64_less_than_immediate_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for opcode in [LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU] {
        for source in [0, 1, i64::MAX as u64, 1u64 << 63, u64::MAX] {
            for imm in [-2048, -1, 0, 1, 2047] {
                let (instruction, rd) = rv64_rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; RV64_REGISTER_NUM_LIMBS],
                    Some(encode_i12(imm)),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

                let mut result = [F::ZERO; RV64_REGISTER_NUM_LIMBS];
                result[0] = F::from_bool(expected(opcode, source, imm));
                assert_eq!(result, tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd));
            }
        }
    }

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

#[test]
fn rv64_less_than_immediate_result_negative() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);
    let (instruction, _) = rv64_rand_write_register_or_imm(
        &mut tester,
        0u64.to_le_bytes(),
        [0; RV64_REGISTER_NUM_LIMBS],
        Some(encode_i12(1)),
        LessThanImmOpcode::SLTI.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut LessThanImmCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.cmp_result = F::ZERO;
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("altered comparison result should fail");
}

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64LessThanImmExecutor,
    Rv64LessThanImmAir,
    Rv64LessThanImmChipGpu,
    Rv64LessThanImmChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_checker = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));
    let air = Rv64LessThanImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        LessThanImmCoreAir::new(range_checker.bus(), LessThanImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64LessThanImmExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor::new(),
        LessThanImmOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LessThanImmChip::new(
        LessThanImmFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LessThanImmChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 64)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_less_than_immediate_boundaries_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_harness(&tester);

    for opcode in [LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU] {
        for (source, imm) in [
            (0, -2048),
            (1, -1),
            (i64::MAX as u64, 0),
            (1u64 << 63, 1),
            (u64::MAX, 2047),
        ] {
            let (instruction, _) = rv64_rand_write_register_or_imm(
                &mut tester,
                source.to_le_bytes(),
                [0; RV64_REGISTER_NUM_LIMBS],
                Some(encode_i12(imm)),
                opcode.global_opcode().as_usize(),
                &mut rng,
            );
            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &instruction,
            );
        }
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluImmU16AdapterRecord,
        &'a mut LessThanImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmU16AdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_less_than_immediate_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let instruction = |opcode: LessThanImmOpcode, rd: usize, rs1: usize, imm: i16| {
        Instruction::<F>::from_usize(
            opcode.global_opcode(),
            [
                reg(rd),
                reg(rs1),
                encode_i12(imm),
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        )
    };
    let instructions = [
        instruction(LessThanImmOpcode::SLTI, 1, 0, 1),
        instruction(LessThanImmOpcode::SLTIU, 2, 1, -1),
        instruction(LessThanImmOpcode::SLTI, 3, 1, -1),
        instruction(LessThanImmOpcode::SLTIU, 4, 0, -2048),
        instruction(LessThanImmOpcode::SLTI, 5, 0, 2047),
        instruction(LessThanImmOpcode::SLTIU, 6, 1, 1),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::from(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let execution = VmExecutor::new(config)
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(7, 12))
        .unwrap();

    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_harness(&tester);
    for (pc, instruction) in instructions[..6].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    type Record<'a> = (
        &'a mut Rv64BaseAluImmU16AdapterRecord,
        &'a mut LessThanImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
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
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(
        d_replay_plan
            .opcode_range(LessThanImmOpcode::SLTI.global_opcode())
            .len(),
        3
    );
    assert_eq!(
        d_replay_plan
            .opcode_range(LessThanImmOpcode::SLTIU.global_opcode())
            .len(),
        3
    );
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
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let corrupt_chip =
        Rv64LessThanImmChipGpu::new(corrupt_range_checker, tester.timestamp_max_bits());
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 49);

    let legacy_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let legacy_chip =
        Rv64LessThanImmChipGpu::new(legacy_range_checker.clone(), tester.timestamp_max_bits());
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_counts,
        legacy_range_checker.count.to_host_on(device_ctx).unwrap()
    );

    let expected_trace = <Rv64LessThanImmChip<F> as Chip<
        MatrixRecordArena<F>,
        CpuBackend<SC>,
    >>::generate_proving_ctx(&harness.cpu_chip, harness.matrix_arena)
    .common_main;
    let replay_trace = transport_matrix_d2h_row_major(&replay_ctx.common_main, device_ctx).unwrap();
    let canonical_rows = |matrix: &RowMajorMatrix<F>| {
        let mut rows = (0..matrix.height())
            .map(|row| matrix.row_slice(row).unwrap().to_vec())
            .collect::<Vec<_>>();
        rows.sort_unstable_by_key(|row| row[1].as_canonical_u32());
        rows
    };
    assert_eq!(
        canonical_rows(&expected_trace),
        canonical_rows(&replay_trace)
    );
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR SLTI/SLTIU transcript replay proof failed");
}
