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
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADBU};
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
        program::Program,
        riscv::RV64_REGISTER_NUM_LIMBS,
        SystemOpcode,
    },
};

use crate::{
    adapters::{
        rv64_bytes_to_u16_block, Rv64LoadByteAdapterAir, Rv64LoadByteAdapterExecutor,
        Rv64LoadByteAdapterFiller, RV64_BYTE_BITS,
    },
    load::{
        common::load_write_data, LoadByteCoreAir, LoadByteCoreCols, LoadByteFiller,
        Rv64LoadByteAir, Rv64LoadByteChip, Rv64LoadByteExecutor,
    },
    test_utils::memory::{set_and_execute_load, F, MAX_INS_CAPACITY},
};
#[cfg(feature = "cuda")]
use crate::{
    load::Rv64LoadByteChipGpu,
    test_utils::memory::{dummy_range_checker, transfer_load_byte_records},
};

type ByteHarness = TestChipHarness<F, Rv64LoadByteExecutor, Rv64LoadByteAir, Rv64LoadByteChip<F>>;

fn create_byte_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    ByteHarness,
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
    let air = Rv64LoadByteAir::new(
        Rv64LoadByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadByteExecutor::new(
        Rv64LoadByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadByteChip::<F>::new(
        LoadByteFiller::new(
            Rv64LoadByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        ByteHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn rand_load_byte_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADBU,
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
#[should_panic(expected = "effective address exceeds implemented memory address space")]
fn negative_load_address_wraparound_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, _) = create_byte_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADBU,
        Some([0xf8, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
        Some(16),
        Some(0),
        None,
    );
}

#[test]
#[should_panic(expected = "effective address exceeds implemented memory address space")]
fn negative_load_address_underflow_test() {
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, _) = create_byte_harness(&mut tester);
    let rs1_ptr = 8;
    tester.write_bytes(RV64_REGISTER_AS as usize, rs1_ptr, [F::ZERO; 8]);

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            LOADBU.global_opcode(),
            [
                0,
                rs1_ptr,
                u16::MAX as usize,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                0,
                1,
            ],
        ),
    );
}

#[test]
fn run_loadbu_sanity_test() {
    let read_data = [
        rv64_bytes_to_u16_block([131, 74, 186, 29, 138, 45, 202, 76]),
        rv64_bytes_to_u16_block([0; 8]),
    ];
    for (shift, expected) in [
        (0, [131, 0, 0, 0, 0, 0, 0, 0]),
        (1, [74, 0, 0, 0, 0, 0, 0, 0]),
        (2, [186, 0, 0, 0, 0, 0, 0, 0]),
        (3, [29, 0, 0, 0, 0, 0, 0, 0]),
        (4, [138, 0, 0, 0, 0, 0, 0, 0]),
        (5, [45, 0, 0, 0, 0, 0, 0, 0]),
        (6, [202, 0, 0, 0, 0, 0, 0, 0]),
        (7, [76, 0, 0, 0, 0, 0, 0, 0]),
    ] {
        assert_eq!(
            load_write_data(LOADBU, read_data, shift),
            rv64_bytes_to_u16_block(expected)
        );
    }
}

#[test]
fn negative_split_opcode_role_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADBU,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core: &mut LoadByteCoreCols<F> = core_row.borrow_mut();
        core.selector[0] += F::ONE;
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked byte load trace should fail");
}

#[cfg(feature = "cuda")]
type GpuByteHarness = GpuTestChipHarness<
    F,
    Rv64LoadByteExecutor,
    Rv64LoadByteAir,
    Rv64LoadByteChipGpu,
    Rv64LoadByteChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_byte_harness(tester: &GpuChipTestBuilder) -> GpuByteHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadByteAir::new(
        Rv64LoadByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadByteExecutor::new(
        Rv64LoadByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadByteChip::<F>::new(
        LoadByteFiller::new(
            Rv64LoadByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadByteChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_load_byte_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_byte_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADBU,
            None,
            None,
            None,
            Some(RV64_MEMORY_AS as usize),
        );
    }
    transfer_load_byte_records(&mut harness);
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_loadbu_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let load = |rd: usize, rs1: usize, imm: usize, imm_sign: usize| {
        Instruction::<F>::from_usize(
            LOADBU.global_opcode(),
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
    // The final row aliases rd with rs1. The x0 row still reserves T+2, but emits no write.
    let instructions = [
        load(2, 1, 0, 0),
        load(3, 1, 1, 0),
        load(4, 1, 2, 0),
        load(5, 1, 3, 0),
        load(6, 1, 4, 0),
        load(7, 1, 5, 0),
        load(0, 1, 6, 0),
        load(9, 9, u16::MAX as usize, 1),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let base = 0x80u64;
    let bytes = [0x80, 0x7f, 0xfe, 0x01, 0xff, 0x00, 0xaa, 0x55];
    let mut init_memory: SparseMemoryImage = base
        .to_le_bytes()
        .into_iter()
        .enumerate()
        .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(1) + offset) as u32), byte))
        .collect();
    init_memory.extend(
        (base + 8)
            .to_le_bytes()
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(9) + offset) as u32), byte)),
    );
    init_memory.extend(
        bytes
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, base as u32 + offset as u32), byte)),
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
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(16, 32))
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

    let mut harness = create_cuda_byte_harness(&tester);
    for (pc, instruction) in instructions[..8].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    transfer_load_byte_records(&mut harness);

    let range_checker = tester.range_checker();
    let bitwise_lookup = tester.bitwise_op_lookup();
    let d_program = GpuRvrProgram::upload(&program, &memory_config, &device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.opcode_range(LOADBU.global_opcode()).len(), 8);
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
    let legacy_chip = Rv64LoadByteChipGpu::new(
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

    let expected_trace =
        <Rv64LoadByteChip<F> as Chip<MatrixRecordArena<F>, CpuBackend<SC>>>::generate_proving_ctx(
            &harness.cpu_chip,
            harness.matrix_arena,
        )
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
        .expect("RVR LBU transcript replay proof failed");

    // Use a one-row fixture so a rejected row must leave the complete histograms empty.
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
    let first_write_timestamp = corrupt_result.program_log[0].timestamp + 2;
    corrupt_result
        .memory_log
        .iter_mut()
        .find(|event| event.timestamp == first_write_timestamp)
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
    let corrupt_chip = Rv64LoadByteChipGpu::new(
        corrupt_range.clone(),
        corrupt_bitwise.clone(),
        address_bits,
        timestamp_max_bits,
    );
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_single_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 229);
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

    // Non-u16 cells are rejected by shared transcript upload before a kernel can launch.
    let mut corrupt_u16 = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    corrupt_u16.memory_log[0].value[0] = u16::MAX as u32 + 1;
    assert!(d_single_program
        .upload_transcript(&corrupt_u16, RvrPreflightEndpoint::Terminated)
        .is_err());

    // The executor accepts the largest u32 byte address even when the configured byte-pointer
    // width is 33. Replay must preserve that boundary without truncating larger effective values.
    let mut max_address_transcript = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    max_address_transcript.memory_log[0].value = [u16::MAX as u32, u16::MAX as u32, 0, 0];
    max_address_transcript.memory_log[1].pointer = (u32::MAX & !7) / 2;
    max_address_transcript.memory_log[2].value = [0x55, 0, 0, 0];
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
    let max_address_chip = Rv64LoadByteChipGpu::new(
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

    let overflow_program = Program::from_instructions(&[
        load(2, 1, 1, 0),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ]);
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
    let overflow_chip = Rv64LoadByteChipGpu::new(
        overflow_range.clone(),
        overflow_bitwise.clone(),
        wide_byte_ptr_bits,
        timestamp_max_bits,
    );
    overflow_chip
        .generate_proving_ctx_from_rvr(&d_overflow_program, &d_overflow, &d_overflow_plan)
        .unwrap();
    assert_eq!(d_overflow.error_code().unwrap(), 227);
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
        &VmExe::new(x0_program.clone()).with_init_memory(init_memory.clone()),
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
    // Reinterpret the pre-TERMINATE boundary as a suspended segment sentinel.
    corrupt_gap.program_log.truncate(2);
    corrupt_gap.program_log[1].timestamp -= 1;
    let d_x0_program = GpuRvrProgram::upload(&x0_program, &memory_config, &device_ctx).unwrap();
    let gap_endpoint = RvrPreflightEndpoint::Suspended {
        resume_pc: corrupt_gap.program_log[1].pc,
        final_timestamp: corrupt_gap.program_log[1].timestamp,
    };
    let (d_corrupt_gap, d_corrupt_gap_plan) = d_x0_program
        .upload_transcript(&corrupt_gap, gap_endpoint)
        .unwrap();
    let gap_range = Arc::new(VariableRangeCheckerChipGPU::new(
        default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let gap_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let gap_chip = Rv64LoadByteChipGpu::new(
        gap_range.clone(),
        gap_bitwise.clone(),
        address_bits,
        timestamp_max_bits,
    );
    gap_chip
        .generate_proving_ctx_from_rvr(&d_x0_program, &d_corrupt_gap, &d_corrupt_gap_plan)
        .unwrap();
    assert_eq!(d_corrupt_gap.error_code().unwrap(), 222);
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

    // A negative immediate is added in signed width, rather than wrapping modulo u32.
    let underflow_program = Program::from_instructions(&[
        load(2, 1, u16::MAX as usize, 1),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ]);
    let mut underflow_transcript = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    underflow_transcript.memory_log[0].value = [0; 4];
    let d_underflow_program =
        GpuRvrProgram::upload(&underflow_program, &memory_config, &device_ctx).unwrap();
    let (d_underflow, d_underflow_plan) = d_underflow_program
        .upload_transcript(&underflow_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let underflow_range = Arc::new(VariableRangeCheckerChipGPU::new(
        default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let underflow_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let underflow_chip = Rv64LoadByteChipGpu::new(
        underflow_range.clone(),
        underflow_bitwise.clone(),
        address_bits,
        timestamp_max_bits,
    );
    underflow_chip
        .generate_proving_ctx_from_rvr(&d_underflow_program, &d_underflow, &d_underflow_plan)
        .unwrap();
    assert_eq!(d_underflow.error_code().unwrap(), 227);
    assert!(underflow_range
        .count
        .to_host_on(&device_ctx)
        .unwrap()
        .iter()
        .all(|&count| count == F::ZERO));
    assert!(underflow_bitwise
        .count
        .to_host_on(&device_ctx)
        .unwrap()
        .iter()
        .all(|&count| count == F::ZERO));
}
