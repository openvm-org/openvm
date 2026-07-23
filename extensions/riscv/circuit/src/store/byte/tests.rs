use std::{borrow::BorrowMut, sync::Arc};

#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
    GpuTestChipHarness,
};
use openvm_circuit::arch::testing::{
    TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::arch::MemoryConfig;
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
#[cfg(feature = "cuda")]
use openvm_instructions::{riscv::RV64_MEMORY_AS, PUBLIC_VALUES_AS};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, STOREB};
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
        riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        SystemOpcode,
    },
};

use crate::{
    adapters::{
        rv64_bytes_to_u16_block, Rv64StoreByteAdapterAir, Rv64StoreByteAdapterExecutor,
        Rv64StoreByteAdapterFiller, RV64_BYTE_BITS,
    },
    store::{
        common::store_write_data, Rv64StoreByteAir, Rv64StoreByteChip, Rv64StoreByteExecutor,
        StoreByteCoreAir, StoreByteCoreCols, StoreByteFiller,
    },
    test_utils::memory::{set_and_execute_store, store_memory_config, F, MAX_INS_CAPACITY},
};
#[cfg(feature = "cuda")]
use crate::{
    store::Rv64StoreByteChipGpu,
    test_utils::memory::{
        dummy_range_checker, store_gpu_memory_config, transfer_store_byte_records,
    },
};

type StoreByteHarness =
    TestChipHarness<F, Rv64StoreByteExecutor, Rv64StoreByteAir, Rv64StoreByteChip<F>>;

fn create_store_byte_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreByteHarness,
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
    let air = Rv64StoreByteAir::new(
        Rv64StoreByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreByteExecutor::new(
        Rv64StoreByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64StoreByteChip::<F>::new(
        StoreByteFiller::new(
            Rv64StoreByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        StoreByteHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn rand_store_byte_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_byte_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            STOREB,
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
fn run_storeb_sanity_test() {
    let read_data = rv64_bytes_to_u16_block([221, 104, 58, 147, 175, 33, 198, 250]);
    let prev_data = [
        rv64_bytes_to_u16_block([199, 83, 243, 12, 90, 121, 64, 205]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 0),
        [
            rv64_bytes_to_u16_block([221, 83, 243, 12, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 1),
        [
            rv64_bytes_to_u16_block([199, 221, 243, 12, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 2),
        [
            rv64_bytes_to_u16_block([199, 83, 221, 12, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 3),
        [
            rv64_bytes_to_u16_block([199, 83, 243, 221, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 4),
        [
            rv64_bytes_to_u16_block([199, 83, 243, 12, 221, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 5),
        [
            rv64_bytes_to_u16_block([199, 83, 243, 12, 90, 221, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 6),
        [
            rv64_bytes_to_u16_block([199, 83, 243, 12, 90, 121, 221, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 7),
        [
            rv64_bytes_to_u16_block([199, 83, 243, 12, 90, 121, 64, 221]),
            prev_data[1]
        ]
    );
}

#[test]
fn negative_split_write_data_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_byte_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREB,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core: &mut StoreByteCoreCols<F> = core_row.borrow_mut();
        core.read_data[0] += F::ONE;
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked byte store trace should fail");
}

#[cfg(feature = "cuda")]
type GpuStoreByteHarness = GpuTestChipHarness<
    F,
    Rv64StoreByteExecutor,
    Rv64StoreByteAir,
    Rv64StoreByteChipGpu,
    Rv64StoreByteChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_store_byte_harness(tester: &GpuChipTestBuilder) -> GpuStoreByteHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64StoreByteAir::new(
        Rv64StoreByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreByteExecutor::new(
        Rv64StoreByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64StoreByteChip::<F>::new(
        StoreByteFiller::new(
            Rv64StoreByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64StoreByteChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case::test_case(RV64_MEMORY_AS as usize)]
#[test_case::test_case(PUBLIC_VALUES_AS as usize)]
fn test_cuda_rand_store_byte_tracegen(mem_as: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_store_byte_harness(&tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            STOREB,
            None,
            None,
            None,
            Some(mem_as),
        );
    }
    transfer_store_byte_records(&mut harness);
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_storeb_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let store = |rs2: usize, rs1: usize, imm: usize, imm_sign: usize| {
        Instruction::<F>::from_usize(
            STOREB.global_opcode(),
            [
                reg(rs2),
                reg(rs1),
                imm,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                imm_sign,
            ],
        )
    };
    // The first row seeds the block, the next seven overwrite every byte shift, the negative
    // immediate repeats shift seven, and the final row aliases rs1 with rs2.
    let instructions = [
        store(2, 1, 0, 0),
        store(2, 1, 1, 0),
        store(2, 1, 2, 0),
        store(2, 1, 3, 0),
        store(2, 1, 4, 0),
        store(2, 1, 5, 0),
        store(2, 1, 6, 0),
        store(2, 1, 7, 0),
        store(2, 3, u16::MAX as usize, 1),
        store(4, 4, 0, 0),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let mut init_memory = [
        (1usize, 0x80u64),
        (2, 0xfedc_ba98_7654_32ddu64),
        (3, 0x88),
        (4, 0x84),
    ]
    .into_iter()
    .flat_map(|(register, value)| {
        value
            .to_le_bytes()
            .into_iter()
            .enumerate()
            .map(move |(offset, byte)| ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte))
    })
    .collect::<SparseMemoryImage>();
    init_memory.extend(
        [0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87]
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_MEMORY_AS, 0x80 + offset as u32), byte)),
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
    assert_eq!(execution.transcript.initial_write_log.len(), 1);

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

    let mut harness = create_cuda_store_byte_harness(&tester);
    for (pc, instruction) in instructions[..10].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    transfer_store_byte_records(&mut harness);

    let range_checker = tester.range_checker();
    let bitwise_lookup = tester.bitwise_op_lookup();
    let d_program = GpuRvrProgram::upload(&program, &memory_config, &device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.opcode_range(STOREB.global_opcode()).len(), 10);
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
    let legacy_chip = Rv64StoreByteChipGpu::new(
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
        <Rv64StoreByteChip<F> as Chip<MatrixRecordArena<F>, CpuBackend<SC>>>::generate_proving_ctx(
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
        .expect("RVR STOREB transcript replay proof failed");

    // Use a one-row fixture to prove rejected rows cannot partially update either histogram.
    let single_instructions = [
        store(2, 1, 0, 0),
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
    let mut corrupt_post = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    corrupt_post.memory_log[2].value[0] ^= 1;
    let (d_corrupt, d_corrupt_plan) = d_single_program
        .upload_transcript(&corrupt_post, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_range = Arc::new(VariableRangeCheckerChipGPU::new(
        default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let corrupt_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let corrupt_chip = Rv64StoreByteChipGpu::new(
        corrupt_range.clone(),
        corrupt_bitwise.clone(),
        address_bits,
        timestamp_max_bits,
    );
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_single_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 260);
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

    // Invalid execution fields, noncanonical register pointers, and the separate RV64IO
    // public-values shape all fail closed before trace or lookup writes.
    for invalid_instruction in [
        Instruction::<F>::from_usize(
            STOREB.global_opcode(),
            [
                reg(2),
                reg(1),
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                0,
                0,
            ],
        ),
        Instruction::<F>::from_usize(
            STOREB.global_opcode(),
            [
                reg(2),
                1,
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::<F>::from_usize(
            STOREB.global_opcode(),
            [
                reg(2),
                reg(1),
                0,
                RV64_REGISTER_AS as usize,
                PUBLIC_VALUES_AS as usize,
                1,
                0,
            ],
        ),
    ] {
        let invalid_program = Program::from_instructions(&[
            invalid_instruction,
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
        ]);
        let d_invalid_program =
            GpuRvrProgram::upload(&invalid_program, &memory_config, &device_ctx).unwrap();
        let (d_invalid, d_invalid_plan) = d_invalid_program
            .upload_transcript(
                &single_execution.transcript,
                RvrPreflightEndpoint::Terminated,
            )
            .unwrap();
        let invalid_range = Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ));
        let invalid_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
        let invalid_chip = Rv64StoreByteChipGpu::new(
            invalid_range.clone(),
            invalid_bitwise.clone(),
            address_bits,
            timestamp_max_bits,
        );
        invalid_chip
            .generate_proving_ctx_from_rvr(&d_invalid_program, &d_invalid, &d_invalid_plan)
            .unwrap();
        assert_eq!(d_invalid.error_code().unwrap(), 254);
        assert!(invalid_range
            .count
            .to_host_on(&device_ctx)
            .unwrap()
            .iter()
            .all(|&count| count == F::ZERO));
        assert!(invalid_bitwise
            .count
            .to_host_on(&device_ctx)
            .unwrap()
            .iter()
            .all(|&count| count == F::ZERO));
    }

    // The containing block must be the one selected by the effective byte address.
    let mut wrong_block = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    wrong_block.memory_log[2].pointer += RV64_REGISTER_NUM_LIMBS as u32;
    wrong_block.initial_write_log[0].pointer = wrong_block.memory_log[2].pointer;
    let (d_wrong_block, d_wrong_block_plan) = d_single_program
        .upload_transcript(&wrong_block, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let wrong_block_chip = Rv64StoreByteChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone())),
        address_bits,
        timestamp_max_bits,
    );
    wrong_block_chip
        .generate_proving_ctx_from_rvr(&d_single_program, &d_wrong_block, &d_wrong_block_plan)
        .unwrap();
    assert_eq!(d_wrong_block.error_code().unwrap(), 258);

    // Signed address arithmetic must reject underflow, low-word overflow, and an otherwise-valid
    // u32 address outside the configured byte-pointer domain.
    let configured_oob = 1u32
        .checked_shl(address_bits as u32)
        .expect("test memory uses a sub-32-bit byte-pointer domain");
    for (base, imm, imm_sign) in [
        (0u32, u16::MAX as usize, 1usize),
        (u32::MAX, 1, 0),
        (configured_oob, 0, 0),
    ] {
        let boundary_program = Program::from_instructions(&[
            store(2, 1, imm, imm_sign),
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
        ]);
        let d_boundary_program =
            GpuRvrProgram::upload(&boundary_program, &memory_config, &device_ctx).unwrap();
        let mut boundary_transcript = RvrPreflightTranscript {
            program_log: single_execution.transcript.program_log.clone(),
            memory_log: single_execution.transcript.memory_log.clone(),
            initial_write_log: single_execution.transcript.initial_write_log.clone(),
        };
        boundary_transcript.memory_log[0].value = [base & u16::MAX as u32, base >> 16, 0, 0];
        let (d_boundary, d_boundary_plan) = d_boundary_program
            .upload_transcript(&boundary_transcript, RvrPreflightEndpoint::Terminated)
            .unwrap();
        let boundary_range = Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ));
        let boundary_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
        let boundary_chip = Rv64StoreByteChipGpu::new(
            boundary_range.clone(),
            boundary_bitwise.clone(),
            address_bits,
            timestamp_max_bits,
        );
        boundary_chip
            .generate_proving_ctx_from_rvr(&d_boundary_program, &d_boundary, &d_boundary_plan)
            .unwrap();
        assert_eq!(d_boundary.error_code().unwrap(), 257);
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

    // The largest u32 byte address remains valid when the configured domain includes it.
    let mut max_address = RvrPreflightTranscript {
        program_log: single_execution.transcript.program_log.clone(),
        memory_log: single_execution.transcript.memory_log.clone(),
        initial_write_log: single_execution.transcript.initial_write_log.clone(),
    };
    max_address.memory_log[0].value = [u16::MAX as u32, u16::MAX as u32, 0, 0];
    let max_block_pointer = (u32::MAX & !7) / 2;
    max_address.memory_log[2].pointer = max_block_pointer;
    max_address.initial_write_log[0].pointer = max_block_pointer;
    let mut expected_post = max_address.initial_write_log[0].initial_value;
    expected_post[3] =
        (expected_post[3] & u8::MAX as u32) | ((max_address.memory_log[1].value[0] & 0xff) << 8);
    max_address.memory_log[2].value = expected_post;
    let mut wide_memory_config = memory_config.clone();
    wide_memory_config.pointer_max_bits = 32;
    wide_memory_config.addr_spaces[RV64_MEMORY_AS as usize].num_cells = 1usize << 31;
    let wide_byte_ptr_bits =
        openvm_circuit::arch::to_byte_ptr_bits(wide_memory_config.pointer_max_bits);
    let d_wide_program =
        GpuRvrProgram::upload(&single_program, &wide_memory_config, &device_ctx).unwrap();
    let (d_max_address, d_max_address_plan) = d_wide_program
        .upload_transcript(&max_address, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let max_address_chip = Rv64StoreByteChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone())),
        wide_byte_ptr_bits,
        timestamp_max_bits,
    );
    max_address_chip
        .generate_proving_ctx_from_rvr(&d_wide_program, &d_max_address, &d_max_address_plan)
        .unwrap();
    assert_eq!(d_max_address.error_code().unwrap(), 0);

    // A repeated register read must equal its predecessor even when a malicious transcript keeps
    // the later store event structurally valid.
    let repeated_program = Program::from_instructions(&[
        store(2, 1, 0, 0),
        store(2, 1, 1, 0),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ]);
    let repeated_execution = VmExecutor::new(Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    })
    .unwrap()
    .rvr_preflight_instance(
        &VmExe::new(repeated_program.clone()).with_init_memory(init_memory),
        None,
    )
    .unwrap()
    .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(4, 8))
    .unwrap();
    let mut bad_predecessor = RvrPreflightTranscript {
        program_log: repeated_execution.transcript.program_log,
        memory_log: repeated_execution.transcript.memory_log,
        initial_write_log: repeated_execution.transcript.initial_write_log,
    };
    bad_predecessor.memory_log[4].value[0] ^= 1;
    let d_repeated_program =
        GpuRvrProgram::upload(&repeated_program, &memory_config, &device_ctx).unwrap();
    let (d_bad_predecessor, d_bad_predecessor_plan) = d_repeated_program
        .upload_transcript(&bad_predecessor, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let bad_predecessor_chip = Rv64StoreByteChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx)),
        address_bits,
        timestamp_max_bits,
    );
    bad_predecessor_chip
        .generate_proving_ctx_from_rvr(
            &d_repeated_program,
            &d_bad_predecessor,
            &d_bad_predecessor_plan,
        )
        .unwrap();
    assert_eq!(d_bad_predecessor.error_code().unwrap(), 259);
}
