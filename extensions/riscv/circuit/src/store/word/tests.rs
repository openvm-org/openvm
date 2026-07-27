use std::{borrow::BorrowMut, sync::Arc};

#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
    GpuTestChipHarness,
};
use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    Postflight, PreflightHistory, PreflightProgramEvent, TraceFiller, BLOCK_FE_WIDTH,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, DEFERRAL_AS, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, STOREW};
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
        rv64_bytes_to_u16_block, Rv64StoreMultiByteAdapterAir, Rv64StoreMultiByteAdapterCols,
        Rv64StoreMultiByteAdapterExecutor, Rv64StoreMultiByteAdapterFiller, RV64_BYTE_BITS,
    },
    store::{
        common::store_write_data, Rv64StoreWordAir, Rv64StoreWordChip, Rv64StoreWordExecutor,
        StoreWordCoreAir, StoreWordFiller,
    },
    test_utils::memory::{set_and_execute_store, store_memory_config, F, MAX_INS_CAPACITY},
};
#[cfg(feature = "cuda")]
use crate::{
    store::Rv64StoreWordChipGpu,
    test_utils::memory::{dummy_range_checker, store_gpu_memory_config, transfer_store_records},
};

type StoreWordHarness =
    TestChipHarness<F, Rv64StoreWordExecutor, Rv64StoreWordAir, Rv64StoreWordChip<F>>;

fn create_store_word_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreWordHarness,
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
    let air = Rv64StoreWordAir::new(
        Rv64StoreMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreWordExecutor::new(
        Rv64StoreMultiByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64StoreWordChip::<F>::new(
        StoreWordFiller::new(
            Rv64StoreMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        StoreWordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn positive_storew_public_values_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_word_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREW,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(PUBLIC_VALUES_AS as usize),
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
fn rand_store_word_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_word_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            STOREW,
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
fn negative_store_address_wraparound_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, _) = create_store_word_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREW,
        Some([0xf8, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
        Some(16),
        Some(0),
        Some(PUBLIC_VALUES_AS as usize),
    );
}

#[test]
#[should_panic(expected = "effective address exceeds implemented memory address space")]
fn negative_store_address_underflow_test() {
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, _) = create_store_word_harness(&mut tester);
    let rs1_ptr = 8;
    tester.write_bytes(RV64_REGISTER_AS as usize, rs1_ptr, [F::ZERO; 8]);

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            STOREW.global_opcode(),
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
fn run_storew_sanity_test() {
    let read_data = rv64_bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = [
        rv64_bytes_to_u16_block([159, 213, 89, 34, 142, 67, 210, 88]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 0),
        [
            rv64_bytes_to_u16_block([138, 45, 202, 76, 142, 67, 210, 88]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 4),
        [
            rv64_bytes_to_u16_block([159, 213, 89, 34, 138, 45, 202, 76]),
            prev_data[1]
        ]
    );
    // Misaligned within one block.
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 3),
        [
            rv64_bytes_to_u16_block([159, 213, 89, 138, 45, 202, 76, 88]),
            prev_data[1]
        ]
    );
    // Misaligned across the block boundary.
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 6),
        [
            rv64_bytes_to_u16_block([159, 213, 89, 34, 142, 67, 138, 45]),
            rv64_bytes_to_u16_block([202, 76, 17, 203, 44, 118, 240, 5]),
        ]
    );
}

#[test]
fn postflight_store_word_trace_matches_record_arena_across_boundaries() {
    let memory_config = store_memory_config();
    let mut tester = VmChipTestBuilder::from_config(memory_config.clone());
    let range_checker = tester.range_checker();
    let (mut harness, (_, bitwise)) = create_store_word_harness(&mut tester);
    let stores = [
        Instruction::from_usize(
            STOREW.global_opcode(),
            [
                16,
                8,
                0,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            STOREW.global_opcode(),
            [
                24,
                8,
                3,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            STOREW.global_opcode(),
            [
                32,
                8,
                6,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
        Instruction::from_usize(
            STOREW.global_opcode(),
            [
                40,
                8,
                5,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
                1,
                0,
            ],
        ),
    ];
    unsafe {
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_REGISTER_AS,
            4,
            [0x100, 0, 0, 0],
        );
        for (pointer, value) in [
            (8, [0x2211, 0x4433, 0, 0]),
            (12, [0x6655, 0x8877, 0, 0]),
            (16, [0xaa99, 0xccbb, 0, 0]),
            (20, [0xeedd, 0x100f, 0, 0]),
        ] {
            tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
                RV64_REGISTER_AS,
                pointer,
                value,
            );
        }
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_MEMORY_AS,
            0x80,
            [0x0201, 0x0403, 0x0605, 0x0807],
        );
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_MEMORY_AS,
            0x84,
            [0x0a09, 0x0c0b, 0x0e0d, 0x100f],
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
        program: (0..=stores.len())
            .map(|index| PreflightProgramEvent {
                pc: index as u32 * 4,
                timestamp: 1 + index as u32 * 4,
            })
            .collect(),
        memory: tester.memory.memory.take_log(),
    };
    let writes: Vec<_> = history
        .memory
        .accesses
        .iter()
        .filter(|event| event.is_write())
        .collect();
    assert_eq!(writes.len(), 6);
    assert_eq!(
        writes
            .iter()
            .map(|event| event.timestamp)
            .collect::<Vec<_>>(),
        [3, 7, 11, 12, 15, 16]
    );
    assert_eq!(
        writes.iter().map(|event| event.pointer).collect::<Vec<_>>(),
        [0x80, 0x80, 0x80, 0x84, 0x80, 0x84]
    );

    let mut instructions = stores.to_vec();
    instructions.push(stores[0].clone());
    let program = Program::new_without_debug_infos(&instructions, 0);
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
fn negative_split_store_deferral_as_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_word_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREW,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, _) = trace_row.split_at_mut(adapter_width);
        let adapter: &mut Rv64StoreMultiByteAdapterCols<F> = adapter_row.borrow_mut();
        adapter.mem_as = F::from_u32(DEFERRAL_AS);
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked store adapter trace should fail");
}

#[cfg(feature = "cuda")]
type GpuStoreWordHarness = GpuTestChipHarness<
    F,
    Rv64StoreWordExecutor,
    Rv64StoreWordAir,
    Rv64StoreWordChipGpu,
    Rv64StoreWordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_store_word_harness(tester: &GpuChipTestBuilder) -> GpuStoreWordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64StoreWordAir::new(
        Rv64StoreMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreWordExecutor::new(
        Rv64StoreMultiByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64StoreWordChip::<F>::new(
        StoreWordFiller::new(
            Rv64StoreMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64StoreWordChipGpu::new(
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
fn test_cuda_rand_store_word_tracegen(mem_as: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_store_word_harness(&tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            STOREW,
            None,
            None,
            None,
            Some(mem_as),
        );
    }
    transfer_store_records(&mut harness);
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
