use std::{borrow::BorrowMut, sync::Arc};

#[cfg(feature = "cuda")]
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
use openvm_instructions::LocalOpcode;
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
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        Default::default(),
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
