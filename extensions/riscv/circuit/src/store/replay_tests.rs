use std::{borrow::Borrow, sync::Arc};

use openvm_circuit::{
    arch::{
        rvr::{
            cuda::GpuRvrProgram, RvrPreflightEndpoint, RvrPreflightLimits, RvrPreflightTranscript,
        },
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness, TestBuilder,
        },
        to_byte_ptr_bits, MatrixRecordArena, MemoryConfig, VmExecutor,
    },
    system::{
        cuda::memory::MemoryInventoryGPU,
        memory::online::{AddressMap, GuestMemory, TracingMemory},
    },
    utils::test_system_config,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupChip, BitwiseOperationLookupChipGPU},
    var_range::VariableRangeCheckerChipGPU,
    Chip,
};
use openvm_cpu_backend::CpuBackend;
use openvm_cuda_backend::{
    data_transporter::transport_matrix_d2h_row_major,
    prelude::{F, SC},
};
use openvm_cuda_common::copy::MemCopyD2H;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, STORED, STOREH, STOREW};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

use crate::{
    adapters::{
        Rv64StoreMultiByteAdapterAir, Rv64StoreMultiByteAdapterExecutor,
        Rv64StoreMultiByteAdapterFiller, RV64_BYTE_BITS,
    },
    store::{
        Rv64StoreDoublewordAir, Rv64StoreDoublewordChip, Rv64StoreDoublewordChipGpu,
        Rv64StoreDoublewordExecutor, Rv64StoreHalfwordAir, Rv64StoreHalfwordChip,
        Rv64StoreHalfwordChipGpu, Rv64StoreHalfwordExecutor, Rv64StoreWordAir, Rv64StoreWordChip,
        Rv64StoreWordChipGpu, Rv64StoreWordExecutor, StoreDoublewordCoreAir, StoreDoublewordFiller,
        StoreHalfwordCoreAir, StoreHalfwordFiller, StoreWordCoreAir, StoreWordFiller,
    },
    test_utils::memory::{dummy_range_checker, transfer_store_records, MAX_INS_CAPACITY},
    Rv64IConfig,
};

macro_rules! store_replay_test {
    (
        $name:ident,
        $opcode:ident,
        $width:expr,
        $air:ident,
        $core_air:ident,
        $executor:ident,
        $cpu_chip:ident,
        $gpu_chip:ident,
        $filler:ident
    ) => {
        #[test]
        fn $name() {
            let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
            let store = |rs2: usize, rs1: usize, imm: usize, imm_sign: usize| {
                Instruction::<F>::from_usize(
                    $opcode.global_opcode(),
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

            // Every byte shift is represented. This includes all crossing layouts for the width,
            // repeated writes to both blocks, a negative immediate, and an rs1/rs2 alias.
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
                (2, 0xfedc_ba98_7654_3210u64),
                (3, 0x88),
                (4, 0x84),
            ]
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
            .collect::<SparseMemoryImage>();
            init_memory.extend(
                [
                    0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87, 0x98, 0xa9, 0xba, 0xcb, 0xdc,
                    0xed, 0xfe, 0x0f,
                ]
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((RV64_MEMORY_AS, 0x80 + offset as u32), byte)),
            );

            let config = Rv64IConfig {
                system: test_system_config(),
                ..Default::default()
            };
            let memory_config = config.system.memory_config.clone();
            let execution = VmExecutor::new(config)
                .unwrap()
                .rvr_preflight_instance(
                    &VmExe::new(program.clone()).with_init_memory(init_memory.clone()),
                    None,
                )
                .unwrap()
                .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(16, 64))
                .unwrap();

            let mut tester =
                GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus())
                    .with_bitwise_op_lookup(default_bitwise_lookup_bus());
            let mut initial_image =
                GuestMemory::new(AddressMap::from_mem_config(&tester.memory.config));
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

            let range_checker = dummy_range_checker();
            let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
                default_bitwise_lookup_bus(),
            ));
            let air = $air::new(
                Rv64StoreMultiByteAdapterAir::new(
                    tester.memory_bridge(),
                    tester.execution_bridge(),
                    range_checker.bus(),
                    tester.address_bits(),
                ),
                $core_air::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
            );
            let executor = $executor::new(
                Rv64StoreMultiByteAdapterExecutor::new(tester.address_bits()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
            );
            let cpu_chip = $cpu_chip::<F>::new(
                $filler::new(
                    Rv64StoreMultiByteAdapterFiller::new(
                        tester.address_bits(),
                        range_checker.clone(),
                    ),
                    Rv64LoadStoreOpcode::CLASS_OFFSET,
                    bitwise_chip,
                ),
                tester.dummy_memory_helper(),
            );
            let gpu_chip = $gpu_chip::new(
                tester.range_checker(),
                tester.bitwise_op_lookup(),
                tester.address_bits(),
                tester.timestamp_max_bits(),
            );
            let mut harness = GpuTestChipHarness::with_capacity(
                executor,
                air,
                gpu_chip,
                cpu_chip,
                MAX_INS_CAPACITY,
            );
            for (pc, instruction) in instructions[..10].iter().enumerate() {
                tester.execute_with_pc(
                    &mut harness.executor,
                    &mut harness.dense_arena,
                    instruction,
                    pc as u32 * 4,
                );
            }
            transfer_store_records(&mut harness);

            let range_checker = tester.range_checker();
            let bitwise_lookup = tester.bitwise_op_lookup();
            let d_program = GpuRvrProgram::upload(&program, &memory_config, &device_ctx).unwrap();
            let (d_transcript, d_replay_plan) = d_program
                .upload_transcript(&execution.transcript, execution.endpoint)
                .unwrap();
            assert_eq!(
                d_replay_plan.opcode_range($opcode.global_opcode()).len(),
                10
            );
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
            let legacy_bitwise_lookup =
                Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
            let legacy_chip = $gpu_chip::new(
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
                <$cpu_chip<F> as Chip<MatrixRecordArena<F>, CpuBackend<SC>>>::generate_proving_ctx(
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
                .expect(concat!(
                    "RVR ",
                    stringify!($opcode),
                    " transcript replay proof failed"
                ));

            // A single crossing row exercises second-block validation. Corruption must be
            // rejected before either shared histogram is updated.
            let crossing_program = Program::from_instructions(&[
                store(2, 1, 7, 0),
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
            ]);
            let crossing_execution = VmExecutor::new(Rv64IConfig {
                system: test_system_config(),
                ..Default::default()
            })
            .unwrap()
            .rvr_preflight_instance(
                &VmExe::new(crossing_program.clone()).with_init_memory(init_memory.clone()),
                None,
            )
            .unwrap()
            .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(4, 8))
            .unwrap();
            let mut corrupt_crossing = RvrPreflightTranscript {
                program_log: crossing_execution.transcript.program_log.clone(),
                memory_log: crossing_execution.transcript.memory_log.clone(),
                initial_write_log: crossing_execution.transcript.initial_write_log.clone(),
            };
            let second_write_timestamp = corrupt_crossing.program_log[0].timestamp + 3;
            corrupt_crossing
                .memory_log
                .iter_mut()
                .find(|event| event.timestamp == second_write_timestamp)
                .unwrap()
                .value[0] ^= 1;
            let d_crossing_program =
                GpuRvrProgram::upload(&crossing_program, &memory_config, &device_ctx).unwrap();
            let (d_corrupt_crossing, d_corrupt_crossing_plan) = d_crossing_program
                .upload_transcript(&corrupt_crossing, RvrPreflightEndpoint::Terminated)
                .unwrap();
            let corrupt_range = Arc::new(VariableRangeCheckerChipGPU::new(
                default_var_range_checker_bus(),
                device_ctx.clone(),
            ));
            let corrupt_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
            let corrupt_chip = $gpu_chip::new(
                corrupt_range.clone(),
                corrupt_bitwise.clone(),
                address_bits,
                timestamp_max_bits,
            );
            corrupt_chip
                .generate_proving_ctx_from_rvr(
                    &d_crossing_program,
                    &d_corrupt_crossing,
                    &d_corrupt_crossing_plan,
                )
                .unwrap();
            assert_eq!(d_corrupt_crossing.error_code().unwrap(), 270);
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

            // The non-crossing fourth timestamp is a gap, not an unvalidated memory event.
            let noncross_program = Program::from_instructions(&[
                store(2, 1, 0, 0),
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
            ]);
            let noncross_execution = VmExecutor::new(Rv64IConfig {
                system: test_system_config(),
                ..Default::default()
            })
            .unwrap()
            .rvr_preflight_instance(
                &VmExe::new(noncross_program.clone()).with_init_memory(init_memory.clone()),
                None,
            )
            .unwrap()
            .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(4, 8))
            .unwrap();
            let mut corrupt_gap = RvrPreflightTranscript {
                program_log: noncross_execution.transcript.program_log.clone(),
                memory_log: noncross_execution.transcript.memory_log.clone(),
                initial_write_log: noncross_execution.transcript.initial_write_log.clone(),
            };
            let mut event_in_gap = corrupt_gap.memory_log[2];
            event_in_gap.timestamp = corrupt_gap.program_log[0].timestamp + 3;
            corrupt_gap.memory_log.push(event_in_gap);
            let d_noncross_program =
                GpuRvrProgram::upload(&noncross_program, &memory_config, &device_ctx).unwrap();
            let (d_corrupt_gap, d_corrupt_gap_plan) = d_noncross_program
                .upload_transcript(&corrupt_gap, RvrPreflightEndpoint::Terminated)
                .unwrap();
            let gap_range = Arc::new(VariableRangeCheckerChipGPU::new(
                default_var_range_checker_bus(),
                device_ctx.clone(),
            ));
            let gap_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
            let gap_chip = $gpu_chip::new(
                gap_range.clone(),
                gap_bitwise.clone(),
                address_bits,
                timestamp_max_bits,
            );
            gap_chip
                .generate_proving_ctx_from_rvr(
                    &d_noncross_program,
                    &d_corrupt_gap,
                    &d_corrupt_gap_plan,
                )
                .unwrap();
            assert_eq!(d_corrupt_gap.error_code().unwrap(), 265);
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

            // Public-values stores share the same opcode and row shape. Replay must carry the
            // instruction's address space into the row rather than replacing it with AS2.
            let public_store = Instruction::<F>::from_usize(
                $opcode.global_opcode(),
                [
                    reg(2),
                    reg(1),
                    0,
                    RV64_REGISTER_AS as usize,
                    PUBLIC_VALUES_AS as usize,
                    1,
                    0,
                ],
            );
            let public_program = Program::from_instructions(&[
                public_store,
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
            ]);
            let mut public_init = init_memory.clone();
            for offset in 0..RV64_REGISTER_NUM_LIMBS {
                public_init.insert((RV64_REGISTER_AS, (reg(1) + offset) as u32), 0);
            }
            let public_execution = VmExecutor::new(Rv64IConfig {
                system: test_system_config(),
                ..Default::default()
            })
            .unwrap()
            .rvr_preflight_instance(
                &VmExe::new(public_program.clone()).with_init_memory(public_init),
                None,
            )
            .unwrap()
            .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(4, 8))
            .unwrap();
            assert_eq!(
                public_execution.transcript.memory_log[2].address_space(),
                PUBLIC_VALUES_AS
            );
            let d_public_program =
                GpuRvrProgram::upload(&public_program, &memory_config, &device_ctx).unwrap();
            let (d_public, d_public_plan) = d_public_program
                .upload_transcript(&public_execution.transcript, public_execution.endpoint)
                .unwrap();
            let public_range = Arc::new(VariableRangeCheckerChipGPU::new(
                default_var_range_checker_bus(),
                device_ctx.clone(),
            ));
            let public_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
            let public_chip = $gpu_chip::new(
                public_range.clone(),
                public_bitwise.clone(),
                address_bits,
                timestamp_max_bits,
            );
            let public_ctx = public_chip
                .generate_proving_ctx_from_rvr(&d_public_program, &d_public, &d_public_plan)
                .unwrap();
            assert_eq!(d_public.error_code().unwrap(), 0);
            let public_trace =
                transport_matrix_d2h_row_major(&public_ctx.common_main, &device_ctx).unwrap();
            let public_row = public_trace.row_slice(0).unwrap();
            let (adapter_row, _) = public_row.split_at(Rv64StoreMultiByteAdapterCols::<F>::width());
            let public_adapter: &Rv64StoreMultiByteAdapterCols<F> = adapter_row.borrow();
            assert_eq!(public_adapter.mem_as, F::from_u32(PUBLIC_VALUES_AS));

            // Any address space outside the two AIR-supported store spaces fails before lookup
            // histograms are touched.
            let unsupported_store = Instruction::<F>::from_usize(
                $opcode.global_opcode(),
                [
                    reg(2),
                    reg(1),
                    0,
                    RV64_REGISTER_AS as usize,
                    PUBLIC_VALUES_AS as usize + 1,
                    1,
                    0,
                ],
            );
            let unsupported_program = Program::from_instructions(&[
                unsupported_store,
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
            ]);
            let d_unsupported_program =
                GpuRvrProgram::upload(&unsupported_program, &memory_config, &device_ctx).unwrap();
            let (d_unsupported, d_unsupported_plan) = d_unsupported_program
                .upload_transcript(&public_execution.transcript, public_execution.endpoint)
                .unwrap();
            let unsupported_range = Arc::new(VariableRangeCheckerChipGPU::new(
                default_var_range_checker_bus(),
                device_ctx.clone(),
            ));
            let unsupported_bitwise =
                Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
            let unsupported_chip = $gpu_chip::new(
                unsupported_range.clone(),
                unsupported_bitwise.clone(),
                address_bits,
                timestamp_max_bits,
            );
            unsupported_chip
                .generate_proving_ctx_from_rvr(
                    &d_unsupported_program,
                    &d_unsupported,
                    &d_unsupported_plan,
                )
                .unwrap();
            assert_eq!(d_unsupported.error_code().unwrap(), 264);
            assert!(unsupported_range
                .count
                .to_host_on(&device_ctx)
                .unwrap()
                .iter()
                .all(|&count| count == F::ZERO));
            assert!(unsupported_bitwise
                .count
                .to_host_on(&device_ctx)
                .unwrap()
                .iter()
                .all(|&count| count == F::ZERO));

            // Address arithmetic uses a u64 end bound. A transcript that would wrap the final
            // byte past the RV64 u32 memory domain must fail before pointer or lookup processing.
            let mut overflow = RvrPreflightTranscript {
                program_log: noncross_execution.transcript.program_log,
                memory_log: noncross_execution.transcript.memory_log,
                initial_write_log: noncross_execution.transcript.initial_write_log,
            };
            let overflowing_ptr = u32::MAX - $width as u32 + 2;
            overflow.memory_log[0].value = [
                (overflowing_ptr & u16::MAX as u32) as u16,
                (overflowing_ptr >> 16) as u16,
                0,
                0,
            ];
            let mut wide_memory_config = memory_config;
            wide_memory_config.pointer_max_bits = 32;
            wide_memory_config.addr_spaces[RV64_MEMORY_AS as usize].num_cells = 1usize << 31;
            let wide_byte_ptr_bits = to_byte_ptr_bits(wide_memory_config.pointer_max_bits);
            let d_wide_program =
                GpuRvrProgram::upload(&noncross_program, &wide_memory_config, &device_ctx).unwrap();
            let (d_overflow, d_overflow_plan) = d_wide_program
                .upload_transcript(&overflow, RvrPreflightEndpoint::Terminated)
                .unwrap();
            let overflow_range = Arc::new(VariableRangeCheckerChipGPU::new(
                default_var_range_checker_bus(),
                device_ctx.clone(),
            ));
            let overflow_bitwise = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
            let overflow_chip = $gpu_chip::new(
                overflow_range.clone(),
                overflow_bitwise.clone(),
                wide_byte_ptr_bits,
                timestamp_max_bits,
            );
            overflow_chip
                .generate_proving_ctx_from_rvr(&d_wide_program, &d_overflow, &d_overflow_plan)
                .unwrap();
            assert_eq!(d_overflow.error_code().unwrap(), 267);
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
    };
}

store_replay_test!(
    test_cuda_storeh_tracegen_from_rvr_transcript,
    STOREH,
    2,
    Rv64StoreHalfwordAir,
    StoreHalfwordCoreAir,
    Rv64StoreHalfwordExecutor,
    Rv64StoreHalfwordChip,
    Rv64StoreHalfwordChipGpu,
    StoreHalfwordFiller
);
store_replay_test!(
    test_cuda_storew_tracegen_from_rvr_transcript,
    STOREW,
    4,
    Rv64StoreWordAir,
    StoreWordCoreAir,
    Rv64StoreWordExecutor,
    Rv64StoreWordChip,
    Rv64StoreWordChipGpu,
    StoreWordFiller
);
store_replay_test!(
    test_cuda_stored_tracegen_from_rvr_transcript,
    STORED,
    8,
    Rv64StoreDoublewordAir,
    StoreDoublewordCoreAir,
    Rv64StoreDoublewordExecutor,
    Rv64StoreDoublewordChip,
    Rv64StoreDoublewordChipGpu,
    StoreDoublewordFiller
);
