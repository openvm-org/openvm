use std::{borrow::BorrowMut, sync::Arc};

#[cfg(all(feature = "cuda", feature = "rvr"))]
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
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LoadStoreOpcode::{self, LOADB, LOADH, LOADHU, LOADW, LOADWU};
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
        bytes_to_u16_block, LoadMultiByteAdapterAir, LoadMultiByteAdapterFiller, BYTE_BITS,
    },
    load::{
        common::load_write_data, core::LoadCoreCols, LoadWordAir, LoadWordChip, LoadWordCoreAir,
        LoadWordExecutor, LoadWordFiller, LOAD_WORD_OVERLAP_CELLS,
    },
    load_sign_extend::common::load_sign_extend_write_data,
    test_utils::memory::{set_and_execute_load, F, MAX_INS_CAPACITY},
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{load::LoadWordChipGpu, test_utils::memory::dummy_range_checker};

type WordHarness = TestChipHarness<F, LoadWordExecutor, LoadWordAir, LoadWordChip<F>>;

fn create_word_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    WordHarness,
    (
        BitwiseOperationLookupAir<BYTE_BITS>,
        SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(bitwise_bus));
    let air = LoadWordAir::new(
        LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadWordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = LoadWordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let chip = LoadWordChip::<F>::new(
        LoadWordFiller::new(
            LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        WordHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            generate_trace_from_postflight,
        ),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn positive_loadwu_shift4_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_word_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        LOADWU,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
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
fn rand_load_word_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_word_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            LOADWU,
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
fn run_loadwu_sanity_test() {
    let read_data = [
        bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]),
        bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        load_write_data(LOADWU, read_data, 0),
        bytes_to_u16_block([138, 45, 202, 76, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADWU, read_data, 4),
        bytes_to_u16_block([131, 74, 186, 29, 0, 0, 0, 0])
    );
    // Misaligned within one block.
    assert_eq!(
        load_write_data(LOADWU, read_data, 3),
        bytes_to_u16_block([76, 131, 74, 186, 0, 0, 0, 0])
    );
    // Misaligned across the block boundary.
    assert_eq!(
        load_write_data(LOADWU, read_data, 6),
        bytes_to_u16_block([186, 29, 61, 92, 0, 0, 0, 0])
    );
}

#[test]
fn accepted_shift_sets() {
    let read_blocks = [
        bytes_to_u16_block([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]),
        bytes_to_u16_block([0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0, 0x11]),
    ];
    for shift in 0..8 {
        let _ = load_sign_extend_write_data(LOADB, read_blocks, shift);
        let _ = load_sign_extend_write_data(LOADH, read_blocks, shift);
        let _ = load_sign_extend_write_data(LOADW, read_blocks, shift);
        let _ = load_write_data(LOADHU, read_blocks, shift);
        let _ = load_write_data(LOADWU, read_blocks, shift);
    }
}

fn assert_pranked_load_word_fails(prank: impl Fn(&mut LoadCoreCols<F, LOAD_WORD_OVERLAP_CELLS>)) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_word_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        LOADWU,
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
        .expect_err("pranked word load trace should fail");
}

#[test]
fn negative_split_write_data_test() {
    assert_pranked_load_word_fails(|core| core.read_data[0][0] += F::ONE);
}

#[test]
fn negative_split_opcode_role_test() {
    assert_pranked_load_word_fails(|core| core.selector[0] += F::ONE);
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuWordHarness =
    GpuTestChipHarness<F, LoadWordExecutor, LoadWordAir, LoadWordChipGpu, LoadWordChip<F>>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_word_harness(tester: &GpuChipTestBuilder) -> GpuWordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = LoadWordAir::new(
        LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadWordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = LoadWordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let cpu_chip = LoadWordChip::<F>::new(
        LoadWordFiller::new(
            LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = LoadWordChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
        .with_trace_generators(
            generate_trace_from_postflight,
            |chip, program, transcript, plan| {
                chip.generate_proving_ctx_from_postflight(program, transcript, plan)
            },
        )
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_rand_load_word_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_word_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            LOADWU,
            None,
            None,
            None,
        );
    }
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
