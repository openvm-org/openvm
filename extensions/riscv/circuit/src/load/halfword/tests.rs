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
use openvm_instructions::{riscv::RV64_MEMORY_AS, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADHU};
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
        rv64_bytes_to_u16_block, Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, RV64_BYTE_BITS,
    },
    load::{
        common::load_write_data, core::LoadCoreCols, LoadHalfwordCoreAir, LoadHalfwordFiller,
        Rv64LoadHalfwordAir, Rv64LoadHalfwordChip, Rv64LoadHalfwordExecutor,
        LOAD_HALFWORD_OVERLAP_CELLS,
    },
    test_utils::memory::{set_and_execute_load, F, MAX_INS_CAPACITY},
};
#[cfg(feature = "cuda")]
use crate::{
    load::Rv64LoadHalfwordChipGpu,
    test_utils::memory::{dummy_range_checker, transfer_load_records},
};

type HalfwordHarness =
    TestChipHarness<F, Rv64LoadHalfwordExecutor, Rv64LoadHalfwordAir, Rv64LoadHalfwordChip<F>>;

fn create_halfword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    HalfwordHarness,
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
    let air = Rv64LoadHalfwordAir::new(
        Rv64LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadHalfwordExecutor::new(
        Rv64LoadMultiByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadHalfwordChip::<F>::new(
        LoadHalfwordFiller::new(
            Rv64LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        HalfwordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn positive_loadhu_shift6_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        Some([6, 0, 0, 0, 0, 0, 0, 0]),
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
fn rand_load_halfword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADHU,
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
fn run_loadhu_sanity_test() {
    let read_data = [
        rv64_bytes_to_u16_block([175, 33, 198, 250, 131, 74, 186, 29]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        load_write_data(LOADHU, read_data, 0),
        rv64_bytes_to_u16_block([175, 33, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 2),
        rv64_bytes_to_u16_block([198, 250, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 4),
        rv64_bytes_to_u16_block([131, 74, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 6),
        rv64_bytes_to_u16_block([186, 29, 0, 0, 0, 0, 0, 0])
    );
    // Misaligned within one block.
    assert_eq!(
        load_write_data(LOADHU, read_data, 3),
        rv64_bytes_to_u16_block([250, 131, 0, 0, 0, 0, 0, 0])
    );
    // Misaligned across the block boundary.
    assert_eq!(
        load_write_data(LOADHU, read_data, 7),
        rv64_bytes_to_u16_block([29, 61, 0, 0, 0, 0, 0, 0])
    );
}

#[test]
fn negative_split_write_data_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core: &mut LoadCoreCols<F, LOAD_HALFWORD_OVERLAP_CELLS> = core_row.borrow_mut();
        core.read_data[0][0] += F::ONE;
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked halfword load trace should fail");
}

#[cfg(feature = "cuda")]
type GpuHalfwordHarness = GpuTestChipHarness<
    F,
    Rv64LoadHalfwordExecutor,
    Rv64LoadHalfwordAir,
    Rv64LoadHalfwordChipGpu,
    Rv64LoadHalfwordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_halfword_harness(tester: &GpuChipTestBuilder) -> GpuHalfwordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadHalfwordAir::new(
        Rv64LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadHalfwordExecutor::new(
        Rv64LoadMultiByteAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadHalfwordChip::<F>::new(
        LoadHalfwordFiller::new(
            Rv64LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadHalfwordChipGpu::new(
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
#[test]
fn test_cuda_rand_load_halfword_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_halfword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADHU,
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
