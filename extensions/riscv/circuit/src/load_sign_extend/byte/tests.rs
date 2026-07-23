use std::{borrow::BorrowMut, sync::Arc};

#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness,
};
use openvm_circuit::arch::testing::{
    TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADB};
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
            rvr::{cuda::GpuRvrProgram, RvrPreflightLimits},
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
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        SystemOpcode,
    },
};

#[cfg(feature = "cuda")]
use crate::load_sign_extend::{
    test_utils::{dummy_range_checker, transfer_load_sign_extend_byte_records},
    Rv64LoadSignExtendByteChipGpu,
};
use crate::{
    adapters::{
        Rv64LoadByteAdapterAir, Rv64LoadByteAdapterExecutor, Rv64LoadByteAdapterFiller,
        RV64_BYTE_BITS,
    },
    load_sign_extend::{
        byte::{
            LoadSignExtendByteCoreAir, LoadSignExtendByteCoreCols, LoadSignExtendByteFiller,
            Rv64LoadSignExtendByteAir, Rv64LoadSignExtendByteChip, Rv64LoadSignExtendByteExecutor,
        },
        test_utils::{memory_config_for, set_and_execute, F, MAX_INS_CAPACITY},
    },
};

type ByteHarness = TestChipHarness<
    F,
    Rv64LoadSignExtendByteExecutor,
    Rv64LoadSignExtendByteAir,
    Rv64LoadSignExtendByteChip<F>,
>;

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
    let air = Rv64LoadSignExtendByteAir::new(
        Rv64LoadByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadSignExtendByteCoreAir::new(
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.bus(),
            range_checker.bus(),
        ),
    );
    let executor = Rv64LoadSignExtendByteExecutor::new(
        Rv64LoadByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadSignExtendByteChip::<F>::new(
        LoadSignExtendByteFiller::new(
            Rv64LoadByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        ByteHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn rand_load_sign_extend_byte_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADB,
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
fn positive_loadb_shift7_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADB,
        Some([7, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
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
fn negative_split_signed_load_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADB,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core: &mut LoadSignExtendByteCoreCols<F> = core_row.borrow_mut();
        core.data_most_sig_bit += F::ONE;
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked signed byte load trace should fail");
}

#[cfg(feature = "cuda")]
type GpuByteHarness = GpuTestChipHarness<
    F,
    Rv64LoadSignExtendByteExecutor,
    Rv64LoadSignExtendByteAir,
    Rv64LoadSignExtendByteChipGpu,
    Rv64LoadSignExtendByteChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_byte_harness(tester: &GpuChipTestBuilder) -> GpuByteHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadSignExtendByteAir::new(
        Rv64LoadByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadSignExtendByteCoreAir::new(
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.bus(),
            range_checker.bus(),
        ),
    );
    let executor = Rv64LoadSignExtendByteExecutor::new(
        Rv64LoadByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadSignExtendByteChip::<F>::new(
        LoadSignExtendByteFiller::new(
            Rv64LoadByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadSignExtendByteChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_load_sign_extend_byte_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_byte_harness(&tester);
    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADB,
            None,
            None,
            None,
        );
    }
    transfer_load_sign_extend_byte_records(&mut harness);
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_loadb_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let load = |rd: usize, rs1: usize, imm: usize, imm_sign: usize| {
        Instruction::<F>::from_usize(
            LOADB.global_opcode(),
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

    let mut harness = create_cuda_byte_harness(&tester);
    for (pc, instruction) in instructions[..8].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    transfer_load_sign_extend_byte_records(&mut harness);

    let range_checker = tester.range_checker();
    let bitwise_lookup = tester.bitwise_op_lookup();
    let d_program = GpuRvrProgram::upload(&program, &memory_config, &device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.opcode_range(LOADB.global_opcode()).len(), 8);
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_range_counts = range_checker.count.to_host_on(&device_ctx).unwrap();
    let replay_bitwise_counts = bitwise_lookup.count.to_host_on(&device_ctx).unwrap();

    let legacy_range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let legacy_bitwise_lookup = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let legacy_chip = Rv64LoadSignExtendByteChipGpu::new(
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

    let expected_trace = <Rv64LoadSignExtendByteChip<F> as Chip<
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

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR LB transcript replay proof failed");
}
