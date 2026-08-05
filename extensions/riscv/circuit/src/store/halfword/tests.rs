use std::{borrow::BorrowMut, sync::Arc};

#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
    GpuTestChipHarness,
};
use openvm_circuit::arch::testing::{
    TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_instructions::riscv::MEMORY_AS;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LoadStoreOpcode::{self, STOREH};
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
        bytes_to_u16_block, StoreMultiByteAdapterAir, StoreMultiByteAdapterFiller,
        BYTE_BITS,
    },
    store::{
        common::store_write_data,
        core::{fill_padding_row, StoreCoreCols},
        StoreHalfwordAir, StoreHalfwordChip, StoreHalfwordExecutor,
        StoreHalfwordCoreAir, StoreHalfwordFiller, STORE_HALFWORD_VALUE_CELLS,
    },
    test_utils::memory::{set_and_execute_store, store_memory_config, F, MAX_INS_CAPACITY},
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{
    store::StoreHalfwordChipGpu,
    test_utils::memory::{dummy_range_checker, store_gpu_memory_config},
};

type StoreHalfwordHarness =
    TestChipHarness<F, StoreHalfwordExecutor, StoreHalfwordAir, StoreHalfwordChip<F>>;

fn create_store_halfword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreHalfwordHarness,
    (
        BitwiseOperationLookupAir<BYTE_BITS>,
        SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(
        bitwise_bus,
    ));
    let air = StoreHalfwordAir::new(
        StoreMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreHalfwordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = StoreHalfwordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let chip = StoreHalfwordChip::<F>::new(
        StoreHalfwordFiller::new(
            StoreMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        StoreHalfwordHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            generate_trace_from_postflight,
        )
        .with_padding(fill_padding_row),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn rand_store_halfword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_halfword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREH,
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
fn run_storeh_sanity_test() {
    let read_data = bytes_to_u16_block([250, 123, 67, 198, 175, 33, 198, 250]);
    let prev_data = [
        bytes_to_u16_block([144, 56, 175, 92, 90, 121, 64, 205]),
        bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 0),
        [
            bytes_to_u16_block([250, 123, 175, 92, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 2),
        [
            bytes_to_u16_block([144, 56, 250, 123, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 4),
        [
            bytes_to_u16_block([144, 56, 175, 92, 250, 123, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 6),
        [
            bytes_to_u16_block([144, 56, 175, 92, 90, 121, 250, 123]),
            prev_data[1]
        ]
    );
    // Misaligned within one block.
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 3),
        [
            bytes_to_u16_block([144, 56, 175, 250, 123, 121, 64, 205]),
            prev_data[1]
        ]
    );
    // Misaligned across the block boundary.
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 7),
        [
            bytes_to_u16_block([144, 56, 175, 92, 90, 121, 64, 250]),
            bytes_to_u16_block([123, 92, 17, 203, 44, 118, 240, 5]),
        ]
    );
}

#[test]
fn negative_split_opcode_role_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_halfword_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        STOREH,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core: &mut StoreCoreCols<F, STORE_HALFWORD_VALUE_CELLS> = core_row.borrow_mut();
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
        .expect_err("pranked halfword store trace should fail");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuStoreHalfwordHarness = GpuTestChipHarness<
    F,
    StoreHalfwordExecutor,
    StoreHalfwordAir,
    StoreHalfwordChipGpu,
    StoreHalfwordChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_store_halfword_harness(tester: &GpuChipTestBuilder) -> GpuStoreHalfwordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = StoreHalfwordAir::new(
        StoreMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreHalfwordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = StoreHalfwordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let cpu_chip = StoreHalfwordChip::<F>::new(
        StoreHalfwordFiller::new(
            StoreMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = StoreHalfwordChipGpu::new(
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
        .with_padding(fill_padding_row)
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_rand_store_halfword_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_store_halfword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREH,
            None,
            None,
            None,
            Some(MEMORY_AS as usize),
        );
    }
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
