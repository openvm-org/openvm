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
use openvm_instructions::{riscv::MEMORY_AS, LocalOpcode};
use openvm_riscv_transpiler::LoadStoreOpcode::{self, LOADD};
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
        common::load_write_data, core::LoadCoreCols, LoadDoublewordAir, LoadDoublewordChip,
        LoadDoublewordCoreAir, LoadDoublewordExecutor, LoadDoublewordFiller,
        LOAD_DOUBLEWORD_OVERLAP_CELLS,
    },
    test_utils::memory::{set_and_execute_load, F, MAX_INS_CAPACITY},
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{load::LoadDoublewordChipGpu, test_utils::memory::dummy_range_checker};

type DoublewordHarness =
    TestChipHarness<F, LoadDoublewordExecutor, LoadDoublewordAir, LoadDoublewordChip<F>>;

fn create_doubleword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    DoublewordHarness,
    (
        BitwiseOperationLookupAir<BYTE_BITS>,
        SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(bitwise_bus));
    let air = LoadDoublewordAir::new(
        LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadDoublewordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = LoadDoublewordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let chip = LoadDoublewordChip::<F>::new(
        LoadDoublewordFiller::new(
            LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        DoublewordHarness::with_capacity(
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
fn rand_load_doubleword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
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
fn positive_loadd_max_address_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    // The default config exposes the full 2^32-byte memory AS; deterministically load the last
    // addressable doubleword (byte address 2^32 - 8).
    let rs1 = (u32::MAX - 7).to_le_bytes();
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        LOADD,
        Some([rs1[0], rs1[1], rs1[2], rs1[3], 0, 0, 0, 0]),
        Some(0),
        Some(0),
        None,
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
fn positive_loadd_pointer_limb_boundary_cross_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(MemoryConfig::default());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    // ptr = 0xfff9: the crossing block starts at 0x10000, exercising the pointer limb carry.
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        LOADD,
        Some([0xf9, 0xff, 0x00, 0x00, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(MEMORY_AS as usize),
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
        bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]),
        bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(load_write_data(LOADD, read_data, 0), read_data[0]);
    // Every nonzero doubleword shift crosses the block boundary.
    assert_eq!(
        load_write_data(LOADD, read_data, 5),
        bytes_to_u16_block([74, 186, 29, 61, 92, 17, 203, 44])
    );
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
        &mut harness.preflight,
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

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuDoublewordHarness = GpuTestChipHarness<
    F,
    LoadDoublewordExecutor,
    LoadDoublewordAir,
    LoadDoublewordChipGpu,
    LoadDoublewordChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_doubleword_harness(tester: &GpuChipTestBuilder) -> GpuDoublewordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = LoadDoublewordAir::new(
        LoadMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadDoublewordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = LoadDoublewordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let cpu_chip = LoadDoublewordChip::<F>::new(
        LoadDoublewordFiller::new(
            LoadMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = LoadDoublewordChipGpu::new(
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
            &mut harness.preflight,
            &mut rng,
            LOADD,
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
