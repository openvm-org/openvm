use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        ExecutionBridge,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::BaseAluImmOpcode;
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
        system::{
            cuda::memory::MemoryInventoryGPU,
            memory::online::{AddressMap, GuestMemory, TracingMemory},
        },
        utils::test_system_config,
    },
    openvm_circuit_primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
        Chip,
    },
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::{
        data_transporter::{
            assert_eq_host_and_device_matrix_col_maj, transport_matrix_d2h_row_major,
        },
        prelude::SC,
    },
    openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
    openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
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
        adapters::Rv64BaseAluImmAdapterRecord, BitwiseLogicImmCoreRecord,
        Rv64BitwiseLogicImmChipGpu,
    },
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{
    BitwiseLogicImmCoreAir, BitwiseLogicImmCoreCols, BitwiseLogicImmFiller, Rv64BitwiseLogicImmAir,
    Rv64BitwiseLogicImmChip, Rv64BitwiseLogicImmExecutor,
};
use crate::{
    adapters::{
        Rv64BaseAluImmAdapterAir, Rv64BaseAluImmAdapterExecutor, Rv64BaseAluImmAdapterFiller,
        RV64_BYTE_BITS,
    },
    test_utils::rv64_rand_write_register_or_imm,
};

type F = BabyBear;
type Harness = TestChipHarness<
    F,
    Rv64BitwiseLogicImmExecutor,
    Rv64BitwiseLogicImmAir,
    Rv64BitwiseLogicImmChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64BitwiseLogicImmAir,
    Rv64BitwiseLogicImmExecutor,
    Rv64BitwiseLogicImmChip<F>,
) {
    let air = Rv64BitwiseLogicImmAir::new(
        Rv64BaseAluImmAdapterAir::new(execution_bridge, memory_bridge),
        BitwiseLogicImmCoreAir::new(bitwise_chip.bus(), BaseAluImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64BitwiseLogicImmExecutor::new(
        Rv64BaseAluImmAdapterExecutor::new(),
        BaseAluImmOpcode::CLASS_OFFSET,
    );
    let chip = Rv64BitwiseLogicImmChip::new(
        BitwiseLogicImmFiller::new(Rv64BaseAluImmAdapterFiller::new(), bitwise_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let chip = Arc::new(BitwiseOperationLookupChip::new(
        BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS),
    ));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        chip.clone(),
        tester.memory_helper(),
    );
    (
        Harness::with_capacity(executor, air, cpu_chip, 64),
        (chip.air, chip),
    )
}

fn encode_i12(imm: i16) -> usize {
    debug_assert!((-2048..=2047).contains(&imm));
    (imm as i32 as u32 & 0x00ff_ffff) as usize
}

fn expected(opcode: BaseAluImmOpcode, source: u64, imm: i16) -> u64 {
    let imm = imm as i64 as u64;
    match opcode {
        BaseAluImmOpcode::XORI => source ^ imm,
        BaseAluImmOpcode::ORI => source | imm,
        BaseAluImmOpcode::ANDI => source & imm,
        BaseAluImmOpcode::ADDI => unreachable!(),
    }
}

#[test]
fn rv64_bitwise_immediate_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    for opcode in [
        BaseAluImmOpcode::XORI,
        BaseAluImmOpcode::ORI,
        BaseAluImmOpcode::ANDI,
    ] {
        for source in [0, 0x0123_4567_89ab_cdef, u64::MAX] {
            for imm in [-2048, -1, 0, 2047] {
                let (instruction, rd) = rv64_rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; RV64_REGISTER_NUM_LIMBS],
                    Some(encode_i12(imm)),
                    opcode.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.arena, &instruction);
                assert_eq!(
                    expected(opcode, source, imm).to_le_bytes().map(F::from_u8),
                    tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd),
                );
            }
        }
    }

    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

#[test]
fn rv64_bitwise_immediate_binding_negative() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);
    let (instruction, _) = rv64_rand_write_register_or_imm(
        &mut tester,
        0x0123_4567_89ab_cdefu64.to_le_bytes(),
        [0; RV64_REGISTER_NUM_LIMBS],
        Some(encode_i12(-1)),
        BaseAluImmOpcode::XORI.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut BitwiseLogicImmCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.imm_sign = F::ZERO;
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("altered immediate witness should fail");
}

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64BitwiseLogicImmExecutor,
    Rv64BitwiseLogicImmAir,
    Rv64BitwiseLogicImmChipGpu,
    Rv64BitwiseLogicImmChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_bitwise = Arc::new(BitwiseOperationLookupChip::new(default_bitwise_lookup_bus()));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64BitwiseLogicImmChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 64)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_bitwise_immediate_boundaries_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_harness(&tester);

    for opcode in [
        BaseAluImmOpcode::XORI,
        BaseAluImmOpcode::ORI,
        BaseAluImmOpcode::ANDI,
    ] {
        for imm in [-2048, -1, 0, 2047] {
            let (instruction, _) = rv64_rand_write_register_or_imm(
                &mut tester,
                0x0123_4567_89ab_cdefu64.to_le_bytes(),
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
        &'a mut Rv64BaseAluImmAdapterRecord,
        &'a mut BitwiseLogicImmCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmAdapterExecutor>::new(),
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
fn test_cuda_bitwise_immediate_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let instruction = |opcode: BaseAluImmOpcode, rd: usize, rs1: usize, imm: i16| {
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
        instruction(BaseAluImmOpcode::XORI, 2, 1, -2048),
        instruction(BaseAluImmOpcode::ORI, 3, 1, -1),
        instruction(BaseAluImmOpcode::ANDI, 4, 1, 0),
        instruction(BaseAluImmOpcode::XORI, 5, 1, 2047),
        instruction(BaseAluImmOpcode::ORI, 6, 1, -2048),
        instruction(BaseAluImmOpcode::ANDI, 7, 1, -1),
        instruction(BaseAluImmOpcode::XORI, 8, 1, 0),
        instruction(BaseAluImmOpcode::ORI, 9, 1, 2047),
        instruction(BaseAluImmOpcode::ANDI, 10, 1, -2048),
        instruction(BaseAluImmOpcode::XORI, 11, 0, -1),
        instruction(BaseAluImmOpcode::ORI, 12, 1, 0),
        instruction(BaseAluImmOpcode::ANDI, 1, 1, 2047),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let source = 0x0123_4567_89ab_cdefu64.to_le_bytes();
    let init_memory: SparseMemoryImage = source
        .into_iter()
        .enumerate()
        .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(1) + offset) as u32), byte))
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let execution = VmExecutor::new(config)
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(13, 24))
        .unwrap();

    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut initial_image = GuestMemory::new(AddressMap::from_mem_config(&tester.memory.config));
    initial_image.memory.set_from_sparse(&init_memory);
    tester.memory.memory = TracingMemory::from_image(initial_image);
    let device_ctx = tester.range_checker().device_ctx.clone();
    let hasher_chip = tester.memory.hasher_chip.clone().unwrap();
    tester.memory.inventory = MemoryInventoryGPU::new(
        tester.memory.config.clone(),
        hasher_chip,
        device_ctx.clone(),
    );
    tester
        .memory
        .inventory
        .set_initial_memory(&tester.memory.memory.data().memory);
    let mut harness = create_cuda_harness(&tester);
    for (pc, instruction) in instructions[..12].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    type Record<'a> = (
        &'a mut Rv64BaseAluImmAdapterRecord,
        &'a mut BitwiseLogicImmCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmAdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let bitwise_lookup = tester.bitwise_op_lookup();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    for opcode in [
        BaseAluImmOpcode::XORI,
        BaseAluImmOpcode::ORI,
        BaseAluImmOpcode::ANDI,
    ] {
        assert_eq!(d_replay_plan.opcode_range(opcode.global_opcode()).len(), 4);
    }
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_range_counts = range_checker.count.to_host_on(device_ctx).unwrap();
    let replay_bitwise_counts = bitwise_lookup.count.to_host_on(device_ctx).unwrap();

    let mut corrupt_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    let write_timestamp = corrupt_transcript.program_log[4].timestamp + 1;
    let write = corrupt_transcript
        .memory_log
        .iter_mut()
        .find(|event| event.timestamp == write_timestamp)
        .unwrap();
    write.value[1] ^= 1;
    let (d_corrupt, d_corrupt_plan) = d_program
        .upload_transcript(&corrupt_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_chip = Rv64BitwiseLogicImmChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone())),
        tester.timestamp_max_bits(),
    );
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 98);

    let legacy_range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let legacy_bitwise_lookup = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let legacy_chip = Rv64BitwiseLogicImmChipGpu::new(
        legacy_range_checker.clone(),
        legacy_bitwise_lookup.clone(),
        tester.timestamp_max_bits(),
    );
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_range_counts,
        legacy_range_checker.count.to_host_on(device_ctx).unwrap()
    );
    assert_eq!(
        replay_bitwise_counts,
        legacy_bitwise_lookup.count.to_host_on(device_ctx).unwrap()
    );

    let expected_trace = <Rv64BitwiseLogicImmChip<F> as Chip<
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
        .expect("RVR XORI/ORI/ANDI transcript replay proof failed");
}
