use std::{borrow::BorrowMut, sync::Arc};

#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
    GpuTestChipHarness,
};
use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    MemoryConfig, Postflight, PreflightHistory, PreflightProgramEvent, TraceFiller, BLOCK_FE_WIDTH,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
#[cfg(feature = "cuda")]
use openvm_instructions::PUBLIC_VALUES_AS;
use openvm_instructions::{
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
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

use super::trace::generate_trace_from_postflight;
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
fn postflight_store_byte_trace_matches_record_arena_trace_with_overwrites() {
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let range_checker = tester.range_checker();
    let (mut harness, (_, bitwise)) = create_store_byte_harness(&mut tester);
    let stores = [
        Instruction::from_usize(
            STOREB.global_opcode(),
            [
                16,
                8,
                1,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            STOREB.global_opcode(),
            [
                24,
                8,
                6,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            STOREB.global_opcode(),
            [
                32,
                8,
                1,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
    ];
    let sentinel = stores[0].clone();
    unsafe {
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_REGISTER_AS,
            4,
            [0x100, 0, 0, 0],
        );
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_REGISTER_AS,
            8,
            [0xaa, 0, 0, 0],
        );
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_REGISTER_AS,
            12,
            [0xbb, 0, 0, 0],
        );
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_REGISTER_AS,
            16,
            [0xcc, 0, 0, 0],
        );
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_MEMORY_AS,
            0x80,
            [0x2211, 0x4433, 0x6655, 0x8877],
        );
    }
    for (index, instruction) in stores.iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.arena,
            instruction,
            index as u32 * 4,
        );
    }

    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 4,
            },
            PreflightProgramEvent {
                pc: 8,
                timestamp: 7,
            },
            PreflightProgramEvent {
                pc: 12,
                timestamp: 10,
            },
        ],
        memory: tester.memory.memory.take_log(),
    };
    let writes: Vec<_> = history
        .memory
        .accesses
        .iter()
        .filter(|event| event.is_write())
        .collect();
    assert_eq!(writes.len(), 3);
    assert!(writes
        .iter()
        .all(|event| event.address_space() == RV64_MEMORY_AS && event.pointer == 0x80));
    assert_eq!(writes[0].value, [0xaa11, 0x4433, 0x6655, 0x8877]);
    assert_eq!(writes[1].value, [0xaa11, 0x4433, 0x6655, 0x88bb]);
    assert_eq!(writes[2].value, [0xcc11, 0x4433, 0x6655, 0x88bb]);

    let program = Program::new_without_debug_infos(
        &[
            stores[0].clone(),
            stores[1].clone(),
            stores[2].clone(),
            sentinel,
        ],
        0,
    );
    let memory_config = MemoryConfig::default();
    let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();
    let actual = generate_trace_from_postflight(&harness.chip, &postflight).unwrap();
    let actual_range = range_checker.generate_trace::<F>();
    let actual_bitwise = bitwise.generate_trace::<F>();

    let rows_used = harness.arena.trace_offset / harness.arena.width;
    let mut expected_values = harness.arena.trace_buffer;
    expected_values.truncate(rows_used.next_power_of_two() * harness.arena.width);
    let mut expected = RowMajorMatrix::new(expected_values, harness.arena.width);
    harness.chip.inner.fill_trace(
        &harness.chip.mem_helper.as_borrowed(),
        &mut expected,
        rows_used,
    );
    let expected_range = range_checker.generate_trace::<F>();
    let expected_bitwise = bitwise.generate_trace::<F>();

    assert_eq!(actual.width(), expected.width());
    assert_eq!(actual.height(), expected.height());
    assert_eq!(actual.values, expected.values);
    assert_eq!(actual_range.values, expected_range.values);
    assert_eq!(actual_bitwise.values, expected_bitwise.values);
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
