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
use openvm_instructions::{
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADD};
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
        rv64_bytes_to_u16_block, Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, RV64_BYTE_BITS,
    },
    load::{
        common::load_write_data, core::LoadCoreCols, LoadDoublewordCoreAir, LoadDoublewordFiller,
        Rv64LoadDoublewordAir, Rv64LoadDoublewordChip, Rv64LoadDoublewordExecutor,
        LOAD_DOUBLEWORD_OVERLAP_CELLS,
    },
    test_utils::memory::{set_and_execute_load, F, MAX_INS_CAPACITY},
};
#[cfg(feature = "cuda")]
use crate::{
    load::Rv64LoadDoublewordChipGpu,
    test_utils::memory::{dummy_range_checker, transfer_load_records},
};

type DoublewordHarness = TestChipHarness<
    F,
    Rv64LoadDoublewordExecutor,
    Rv64LoadDoublewordAir,
    Rv64LoadDoublewordChip<F>,
>;

fn create_doubleword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    DoublewordHarness,
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
    let air = Rv64LoadDoublewordAir::new(
        Rv64LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadDoublewordExecutor::new(
        Rv64LoadMultiByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadDoublewordChip::<F>::new(
        LoadDoublewordFiller::new(
            Rv64LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        DoublewordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn rand_load_doubleword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADD,
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
fn positive_loadd_pointer_limb_boundary_cross_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    // ptr = 0xfff9: the crossing block starts at 0x10000, exercising the pointer limb carry.
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADD,
        Some([0xf9, 0xff, 0x00, 0x00, 0, 0, 0, 0]),
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
fn run_loadd_sanity_test() {
    let read_data = [
        rv64_bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(load_write_data(LOADD, read_data, 0), read_data[0]);
    // Every nonzero doubleword shift crosses the block boundary.
    assert_eq!(
        load_write_data(LOADD, read_data, 5),
        rv64_bytes_to_u16_block([74, 186, 29, 61, 92, 17, 203, 44])
    );
}

#[test]
fn postflight_loadd_trace_matches_record_arena_trace() {
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, (_, bitwise)) = create_doubleword_harness(&mut tester);
    let range_checker = tester.range_checker();
    let noncrossing = Instruction::from_usize(
        LOADD.global_opcode(),
        [
            16,
            8,
            0,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
            1,
            0,
        ],
    );
    let crossing_x0 = Instruction::from_usize(
        LOADD.global_opcode(),
        [
            0,
            24,
            0,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
            0,
            0,
        ],
    );
    let sentinel = noncrossing.clone();
    unsafe {
        tester
            .memory
            .memory
            .data
            .write::<u16, BLOCK_FE_WIDTH>(RV64_REGISTER_AS, 4, [64, 0, 0, 0]);
        tester
            .memory
            .memory
            .data
            .write::<u16, BLOCK_FE_WIDTH>(RV64_REGISTER_AS, 12, [65, 0, 0, 0]);
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_MEMORY_AS,
            32,
            [0x1001, 0x2002, 0x3003, 0x4004],
        );
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_MEMORY_AS,
            36,
            [0x5005, 0x6006, 0x7007, 0x8008],
        );
    }
    tester.execute_with_pc(&mut harness.executor, &mut harness.arena, &noncrossing, 0);
    tester.execute_with_pc(&mut harness.executor, &mut harness.arena, &crossing_x0, 4);

    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 5,
            },
            PreflightProgramEvent {
                pc: 8,
                timestamp: 9,
            },
        ],
        memory: tester.memory.memory.take_log(),
    };
    let program = Program::new_without_debug_infos(&[noncrossing, crossing_x0, sentinel], 0);
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

fn assert_pranked_load_doubleword_fails(
    prank: impl Fn(&mut LoadCoreCols<F, LOAD_DOUBLEWORD_OVERLAP_CELLS>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADD,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        prank(core_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked doubleword load trace should fail");
}

#[test]
fn negative_split_write_data_test() {
    assert_pranked_load_doubleword_fails(|core| core.read_data[0][0] += F::ONE);
}

#[test]
fn negative_split_opcode_role_test() {
    assert_pranked_load_doubleword_fails(|core| core.selector[0] += F::ONE);
}

#[cfg(feature = "cuda")]
type GpuDoublewordHarness = GpuTestChipHarness<
    F,
    Rv64LoadDoublewordExecutor,
    Rv64LoadDoublewordAir,
    Rv64LoadDoublewordChipGpu,
    Rv64LoadDoublewordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_doubleword_harness(tester: &GpuChipTestBuilder) -> GpuDoublewordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadDoublewordAir::new(
        Rv64LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadDoublewordExecutor::new(
        Rv64LoadMultiByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadDoublewordChip::<F>::new(
        LoadDoublewordFiller::new(
            Rv64LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadDoublewordChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_load_doubleword_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_doubleword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADD,
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
