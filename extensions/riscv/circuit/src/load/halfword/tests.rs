use std::{borrow::BorrowMut, sync::Arc};

#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
    GpuTestChipHarness,
};
use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    MemoryConfig,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{riscv::RV64_MEMORY_AS, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADHU};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::utils::create_seeded_rng;
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
    openvm_cuda_backend::{data_transporter::transport_matrix_d2h_row_major, prelude::SC},
    openvm_cuda_common::copy::MemCopyD2H,
    openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        instruction::Instruction,
        program::Program,
        riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        SystemOpcode,
    },
};

use crate::{
    adapters::{
        rv64_bytes_to_u16_block, Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, RV64_BYTE_BITS,
    },
    load::{
        common::load_write_data, core::LoadCoreCols, LoadHalfwordCoreAir, LoadHalfwordFiller,
        Rv64LoadHalfwordAir, Rv64LoadHalfwordChip, Rv64LoadHalfwordExecutor,
        LOAD_HALFWORD_OVERLAP_CELLS,
    },
    test_utils::memory::{set_and_execute_load, F, MAX_INS_CAPACITY},
};
#[cfg(feature = "cuda")]
use crate::{
    load::Rv64LoadHalfwordChipGpu,
    test_utils::memory::{dummy_range_checker, transfer_load_records},
};

type HalfwordHarness =
    TestChipHarness<F, Rv64LoadHalfwordExecutor, Rv64LoadHalfwordAir, Rv64LoadHalfwordChip<F>>;

fn create_halfword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    HalfwordHarness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let air = Rv64LoadHalfwordAir::new(
        Rv64LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadHalfwordExecutor::new(
        Rv64LoadMultiByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadHalfwordChip::<F>::new(
        LoadHalfwordFiller::new(
            Rv64LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        HalfwordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn positive_loadhu_shift6_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        Some([6, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(RV64_MEMORY_AS as usize),
    );
    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn rand_load_halfword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADHU,
            None,
            None,
            None,
            None,
        );
    }
    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn run_loadhu_sanity_test() {
    let read_data = [
        rv64_bytes_to_u16_block([175, 33, 198, 250, 131, 74, 186, 29]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        load_write_data(LOADHU, read_data, 0),
        rv64_bytes_to_u16_block([175, 33, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 2),
        rv64_bytes_to_u16_block([198, 250, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 4),
        rv64_bytes_to_u16_block([131, 74, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 6),
        rv64_bytes_to_u16_block([186, 29, 0, 0, 0, 0, 0, 0])
    );
    // Misaligned within one block.
    assert_eq!(
        load_write_data(LOADHU, read_data, 3),
        rv64_bytes_to_u16_block([250, 131, 0, 0, 0, 0, 0, 0])
    );
    // Misaligned across the block boundary.
    assert_eq!(
        load_write_data(LOADHU, read_data, 7),
        rv64_bytes_to_u16_block([29, 61, 0, 0, 0, 0, 0, 0])
    );
}

#[test]
fn negative_split_write_data_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core: &mut LoadCoreCols<F, LOAD_HALFWORD_OVERLAP_CELLS> = core_row.borrow_mut();
        core.read_data[0][0] += F::ONE;
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked halfword load trace should fail");
}

#[cfg(feature = "cuda")]
type GpuHalfwordHarness = GpuTestChipHarness<
    F,
    Rv64LoadHalfwordExecutor,
    Rv64LoadHalfwordAir,
    Rv64LoadHalfwordChipGpu,
    Rv64LoadHalfwordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_halfword_harness(tester: &GpuChipTestBuilder) -> GpuHalfwordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadHalfwordAir::new(
        Rv64LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadHalfwordExecutor::new(
        Rv64LoadMultiByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadHalfwordChip::<F>::new(
        LoadHalfwordFiller::new(
            Rv64LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadHalfwordChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_load_halfword_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_halfword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADHU,
            None,
            None,
            None,
            Some(RV64_MEMORY_AS as usize),
        );
    }
    transfer_load_records(&mut harness);
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_loadhu_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let load = |rd: usize, rs1: usize, imm: usize, imm_sign: usize| {
        Instruction::<F>::from_usize(
            LOADHU.global_opcode(),
            [
                reg(rd),
                reg(rs1),
                imm,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                usize::from(rd != 0),
                imm_sign,
            ],
        )
    };
    let instructions = [
        load(2, 1, 0, 0),
        load(3, 1, 1, 0),
        load(4, 1, 2, 0),
        load(5, 1, 3, 0),
        load(6, 1, 4, 0),
        load(7, 1, 5, 0),
        load(0, 1, 6, 0),
        load(8, 10, u16::MAX as usize, 1),
        load(10, 10, u16::MAX as usize, 1),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let mut init_memory: SparseMemoryImage = [(1usize, 0x80u64), (10, 0x1_0000)]
        .into_iter()
        .flat_map(|(register, value)| {
            value
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(move |(offset, byte)| {
                    ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte)
                })
        })
        .collect();
    let low_bytes = [
        0x00, 0x80, 0x34, 0x12, 0xff, 0x7f, 0x00, 0x80, 0xfe, 0xff, 0x55, 0xaa, 0x11, 0x22, 0x33,
        0x44,
    ];
    init_memory.extend(
        low_bytes
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, 0x80 + offset as u32), byte)),
    );
    let carry_bytes = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x80, 0xfe, 0xff, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0x99,
    ];
    init_memory.extend(
        carry_bytes
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, 0xfff8 + offset as u32), byte)),
    );
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
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(16, 48))
        .unwrap();

    let mut tester =
        GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
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

    let mut harness = create_cuda_halfword_harness(&tester);
    for (pc, instruction) in instructions[..9].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    transfer_load_records(&mut harness);

    let range_checker = tester.range_checker();
    let bitwise_lookup = tester.bitwise_op_lookup();
    let d_program = GpuRvrProgram::upload(&program, &memory_config, &device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.opcode_range(LOADHU.global_opcode()).len(), 9);
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_range_counts = range_checker.count.to_host_on(&device_ctx).unwrap();
    let replay_bitwise_counts = bitwise_lookup.count.to_host_on(&device_ctx).unwrap();

    let legacy_range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
        default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let legacy_bitwise_lookup = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let legacy_chip = Rv64LoadHalfwordChipGpu::new(
        legacy_range_checker.clone(),
        legacy_bitwise_lookup.clone(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_range_counts,
        legacy_range_checker.count.to_host_on(&device_ctx).unwrap()
    );
    assert_eq!(
        replay_bitwise_counts,
        legacy_bitwise_lookup.count.to_host_on(&device_ctx).unwrap()
    );

    let expected_trace = <Rv64LoadHalfwordChip<F> as Chip<
        MatrixRecordArena<F>,
        CpuBackend<SC>,
    >>::generate_proving_ctx(&harness.cpu_chip, harness.matrix_arena)
    .common_main;
    let replay_trace =
        transport_matrix_d2h_row_major(&replay_ctx.common_main, &device_ctx).unwrap();
    let legacy_trace =
        transport_matrix_d2h_row_major(&legacy_ctx.common_main, &device_ctx).unwrap();
    assert_eq!(expected_trace, replay_trace);
    assert_eq!(replay_trace, legacy_trace);

    let address_bits = tester.address_bits();
    let timestamp_max_bits = tester.timestamp_max_bits();
    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR LHU transcript replay proof failed");

    let single_instructions = [
        load(2, 1, 0, 0),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let single_program = Program::from_instructions(&single_instructions);
    let single_execution = VmExecutor::new(Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    })
    .unwrap()
    .rvr_preflight_instance(
        &VmExe::new(single_program.clone()).with_init_memory(init_memory.clone()),
        None,
    )
    .unwrap()
    .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(4, 8))
    .unwrap();
    let d_single_program =
        GpuRvrProgram::upload(&single_program, &memory_config, &device_ctx).unwrap();

    let mut corrupt_result = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    let write_timestamp = corrupt_result.program_log[0].timestamp + 3;
    corrupt_result
        .memory_log
        .iter_mut()
        .find(|event| event.timestamp == write_timestamp)
        .unwrap()
        .value[0] ^= 1;
    let (d_corrupt, d_corrupt_plan) = d_single_program
        .upload_transcript(&corrupt_result, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_range = Arc::new(VariableRangeCheckerChipGPU::new(
        default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let corrupt_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let corrupt_chip = Rv64LoadHalfwordChipGpu::new(
        corrupt_range.clone(),
        corrupt_bitwise.clone(),
        address_bits,
        timestamp_max_bits,
    );
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_single_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 239);
    assert!(corrupt_range
        .count
        .to_host_on(&device_ctx)
        .unwrap()
        .iter()
        .all(|&count| count == F::ZERO));
    assert!(corrupt_bitwise
        .count
        .to_host_on(&device_ctx)
        .unwrap()
        .iter()
        .all(|&count| count == F::ZERO));

    let mut max_address_transcript = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    max_address_transcript.memory_log[0].value =
        [(u32::MAX - 1) & u16::MAX as u32, u16::MAX as u32, 0, 0];
    max_address_transcript.memory_log[1].pointer = (u32::MAX & !7) / 2;
    max_address_transcript.memory_log[1].value = [0, 0, 0, 0x8000];
    max_address_transcript.memory_log[2].value = [0x8000, 0, 0, 0];
    let mut wide_memory_config = memory_config.clone();
    wide_memory_config.pointer_max_bits = 32;
    wide_memory_config.addr_spaces[RV64_MEMORY_AS as usize].num_cells = 1usize << 31;
    let wide_byte_ptr_bits =
        openvm_circuit::arch::to_byte_ptr_bits(wide_memory_config.pointer_max_bits);
    let d_wide_single_program =
        GpuRvrProgram::upload(&single_program, &wide_memory_config, &device_ctx).unwrap();
    let (d_max_address, d_max_address_plan) = d_wide_single_program
        .upload_transcript(&max_address_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let max_address_chip = Rv64LoadHalfwordChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone())),
        wide_byte_ptr_bits,
        timestamp_max_bits,
    );
    max_address_chip
        .generate_proving_ctx_from_rvr(&d_wide_single_program, &d_max_address, &d_max_address_plan)
        .unwrap();
    assert_eq!(d_max_address.error_code().unwrap(), 0);

    let mut narrow_memory_config = memory_config.clone();
    narrow_memory_config.pointer_max_bits = 19;
    narrow_memory_config.addr_spaces[RV64_MEMORY_AS as usize].num_cells = 1usize << 19;
    let narrow_byte_ptr_bits =
        openvm_circuit::arch::to_byte_ptr_bits(narrow_memory_config.pointer_max_bits);
    let narrow_limit = 1u32 << narrow_byte_ptr_bits;
    let d_narrow_single_program =
        GpuRvrProgram::upload(&single_program, &narrow_memory_config, &device_ctx).unwrap();
    let mut narrow_boundary_transcript = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    narrow_boundary_transcript.memory_log[0].value = [
        (narrow_limit - 2) & u16::MAX as u32,
        (narrow_limit - 2) >> 16,
        0,
        0,
    ];
    narrow_boundary_transcript.memory_log[1].pointer = (narrow_limit - 8) / 2;
    narrow_boundary_transcript.memory_log[1].value = [0, 0, 0, 0x8000];
    narrow_boundary_transcript.memory_log[2].value = [0x8000, 0, 0, 0];
    let (d_narrow_boundary, d_narrow_boundary_plan) = d_narrow_single_program
        .upload_transcript(
            &narrow_boundary_transcript,
            RvrPreflightEndpoint::Terminated,
        )
        .unwrap();
    let narrow_boundary_chip = Rv64LoadHalfwordChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone())),
        narrow_byte_ptr_bits,
        timestamp_max_bits,
    );
    narrow_boundary_chip
        .generate_proving_ctx_from_rvr(
            &d_narrow_single_program,
            &d_narrow_boundary,
            &d_narrow_boundary_plan,
        )
        .unwrap();
    assert_eq!(d_narrow_boundary.error_code().unwrap(), 0);

    for (boundary_program, expected_error) in [
        (
            Program::from_instructions(&[
                load(2, 1, 1, 0),
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
            ]),
            238,
        ),
        (
            Program::from_instructions(&[
                load(2, 1, 2, 0),
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
            ]),
            237,
        ),
    ] {
        let d_boundary_program =
            GpuRvrProgram::upload(&boundary_program, &narrow_memory_config, &device_ctx).unwrap();
        let (d_boundary, d_boundary_plan) = d_boundary_program
            .upload_transcript(
                &narrow_boundary_transcript,
                RvrPreflightEndpoint::Terminated,
            )
            .unwrap();
        let boundary_range = Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ));
        let boundary_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
        let boundary_chip = Rv64LoadHalfwordChipGpu::new(
            boundary_range.clone(),
            boundary_bitwise.clone(),
            narrow_byte_ptr_bits,
            timestamp_max_bits,
        );
        boundary_chip
            .generate_proving_ctx_from_rvr(&d_boundary_program, &d_boundary, &d_boundary_plan)
            .unwrap();
        assert_eq!(d_boundary.error_code().unwrap(), expected_error);
        assert!(boundary_range
            .count
            .to_host_on(&device_ctx)
            .unwrap()
            .iter()
            .all(|&count| count == F::ZERO));
        assert!(boundary_bitwise
            .count
            .to_host_on(&device_ctx)
            .unwrap()
            .iter()
            .all(|&count| count == F::ZERO));
    }

    for (overflow_program, expected_error) in [
        (
            Program::from_instructions(&[
                load(2, 1, 2, 0),
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
            ]),
            237,
        ),
        (
            Program::from_instructions(&[
                load(2, 1, 1, 0),
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
            ]),
            238,
        ),
    ] {
        let d_overflow_program =
            GpuRvrProgram::upload(&overflow_program, &wide_memory_config, &device_ctx).unwrap();
        let (d_overflow, d_overflow_plan) = d_overflow_program
            .upload_transcript(&max_address_transcript, RvrPreflightEndpoint::Terminated)
            .unwrap();
        let overflow_range = Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ));
        let overflow_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
        let overflow_chip = Rv64LoadHalfwordChipGpu::new(
            overflow_range.clone(),
            overflow_bitwise.clone(),
            wide_byte_ptr_bits,
            timestamp_max_bits,
        );
        overflow_chip
            .generate_proving_ctx_from_rvr(&d_overflow_program, &d_overflow, &d_overflow_plan)
            .unwrap();
        assert_eq!(d_overflow.error_code().unwrap(), expected_error);
        assert!(overflow_range
            .count
            .to_host_on(&device_ctx)
            .unwrap()
            .iter()
            .all(|&count| count == F::ZERO));
        assert!(overflow_bitwise
            .count
            .to_host_on(&device_ctx)
            .unwrap()
            .iter()
            .all(|&count| count == F::ZERO));
    }

    let x0_program = Program::from_instructions(&[
        load(0, 1, 0, 0),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ]);
    let x0_execution = VmExecutor::new(Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    })
    .unwrap()
    .rvr_preflight_instance(
        &VmExe::new(x0_program.clone()).with_init_memory(init_memory),
        None,
    )
    .unwrap()
    .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(4, 8))
    .unwrap();
    let mut corrupt_gap = RvrPreflightTranscript {
        program_log: x0_execution.transcript.program_log.clone(),
        memory_log: x0_execution.transcript.memory_log.clone(),
        initial_write_log: x0_execution.transcript.initial_write_log.clone(),
    };
    let mut event_in_gap = corrupt_gap.memory_log[1];
    event_in_gap.timestamp = corrupt_gap.program_log[0].timestamp + 3;
    corrupt_gap.memory_log.push(event_in_gap);
    let d_x0_program = GpuRvrProgram::upload(&x0_program, &memory_config, &device_ctx).unwrap();
    let (d_corrupt_gap, d_corrupt_gap_plan) = d_x0_program
        .upload_transcript(&corrupt_gap, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let gap_range = Arc::new(VariableRangeCheckerChipGPU::new(
        default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let gap_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let gap_chip = Rv64LoadHalfwordChipGpu::new(
        gap_range.clone(),
        gap_bitwise.clone(),
        address_bits,
        timestamp_max_bits,
    );
    gap_chip
        .generate_proving_ctx_from_rvr(&d_x0_program, &d_corrupt_gap, &d_corrupt_gap_plan)
        .unwrap();
    assert_eq!(d_corrupt_gap.error_code().unwrap(), 235);
    assert!(gap_range
        .count
        .to_host_on(&device_ctx)
        .unwrap()
        .iter()
        .all(|&count| count == F::ZERO));
    assert!(gap_bitwise
        .count
        .to_host_on(&device_ctx)
        .unwrap()
        .iter()
        .all(|&count| count == F::ZERO));
}
