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
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LoadStoreOpcode::{self, STOREB};
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
        bytes_to_u16_block, StoreByteAdapterAir, StoreByteAdapterCols, StoreByteAdapterFiller,
        BYTE_BITS,
    },
    store::{
        common::store_write_data, StoreByteAir, StoreByteChip, StoreByteCoreAir, StoreByteCoreCols,
        StoreByteExecutor, StoreByteFiller,
    },
    test_utils::memory::{
        set_and_execute_store, store_memory_config, F, MAX_INS_CAPACITY, WRAPPED_BASE_CASES,
    },
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{
    store::StoreByteChipGpu,
    test_utils::memory::{dummy_range_checker, store_gpu_memory_config},
};

type StoreByteHarness = TestChipHarness<F, StoreByteExecutor, StoreByteAir, StoreByteChip<F>>;

fn create_store_byte_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreByteHarness,
    (
        BitwiseOperationLookupAir<BYTE_BITS>,
        SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(bitwise_bus));
    let air = StoreByteAir::new(
        StoreByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreByteCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = StoreByteExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let chip = StoreByteChip::<F>::new(
        StoreByteFiller::new(
            StoreByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        StoreByteHarness::with_capacity(
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
fn rand_store_byte_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_byte_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREB,
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
fn positive_storeb_max_address_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_byte_harness(&mut tester);
    // The default config exposes the full 2^32-byte memory AS; deterministically store to the
    // last addressable byte (byte address 2^32 - 1).
    let rs1 = u32::MAX.to_le_bytes();
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        STOREB,
        Some([rs1[0], rs1[1], rs1[2], rs1[3], 0, 0, 0, 0]),
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
fn run_storeb_sanity_test() {
    let read_data = bytes_to_u16_block([221, 104, 58, 147, 175, 33, 198, 250]);
    let prev_data = [
        bytes_to_u16_block([199, 83, 243, 12, 90, 121, 64, 205]),
        bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 0),
        [
            bytes_to_u16_block([221, 83, 243, 12, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 1),
        [
            bytes_to_u16_block([199, 221, 243, 12, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 2),
        [
            bytes_to_u16_block([199, 83, 221, 12, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 3),
        [
            bytes_to_u16_block([199, 83, 243, 221, 90, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 4),
        [
            bytes_to_u16_block([199, 83, 243, 12, 221, 121, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 5),
        [
            bytes_to_u16_block([199, 83, 243, 12, 90, 221, 64, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 6),
        [
            bytes_to_u16_block([199, 83, 243, 12, 90, 121, 221, 205]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 7),
        [
            bytes_to_u16_block([199, 83, 243, 12, 90, 121, 64, 221]),
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
        &mut harness.preflight,
        &mut rng,
        STOREB,
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

#[test]
fn positive_storeb_wrapped_base_test() {
    // RV64I: rs1 + sign_extend(imm) wraps modulo 2^64, so a base with a nonzero upper word can
    // still address the 32-bit memory space.
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_byte_harness(&mut tester);
    for (rs1, imm, imm_sign) in WRAPPED_BASE_CASES {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREB,
            Some(rs1.to_le_bytes()),
            Some(imm),
            Some(imm_sign),
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

fn negative_storeb_base_high_flags_test_case(rs1: u64, imm: u32, imm_sign: u32, rs1_hi: [u32; 2]) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_byte_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        STOREB,
        Some(rs1.to_le_bytes()),
        Some(imm),
        Some(imm_sign),
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, _) = trace_row.split_at_mut(adapter_width);
        let adapter: &mut StoreByteAdapterCols<F> = adapter_row.borrow_mut();
        adapter.rs1_hi_neg = F::from_u32(rs1_hi[0]);
        adapter.rs1_hi_one = F::from_u32(rs1_hi[1]);
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked base-register high-word flags should fail");
}

#[test]
fn negative_storeb_base_high_flags_test() {
    // A zero upper word cannot be claimed as all ones or as one.
    for rs1_hi in [[1, 0], [0, 1], [1, 1]] {
        negative_storeb_base_high_flags_test_case(0x1000, 8, 0, rs1_hi);
    }
    // A wrapped base cannot be claimed as a zero upper word, nor as the other wrap.
    negative_storeb_base_high_flags_test_case(-8i64 as u64, 16, 0, [0, 0]);
    negative_storeb_base_high_flags_test_case(-8i64 as u64, 16, 0, [0, 1]);
    negative_storeb_base_high_flags_test_case((1 << 32) + 8, 0xffe8, 1, [0, 0]);
    negative_storeb_base_high_flags_test_case((1 << 32) + 8, 0xffe8, 1, [1, 0]);
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuStoreByteHarness =
    GpuTestChipHarness<F, StoreByteExecutor, StoreByteAir, StoreByteChipGpu, StoreByteChip<F>>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_store_byte_harness(tester: &GpuChipTestBuilder) -> GpuStoreByteHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = StoreByteAir::new(
        StoreByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreByteCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = StoreByteExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let cpu_chip = StoreByteChip::<F>::new(
        StoreByteFiller::new(
            StoreByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = StoreByteChipGpu::new(
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
fn test_cuda_rand_store_byte_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_store_byte_harness(&tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREB,
            None,
            None,
            None,
        );
    }
    for (rs1, imm, imm_sign) in WRAPPED_BASE_CASES {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREB,
            Some(rs1.to_le_bytes()),
            Some(imm),
            Some(imm_sign),
        );
    }
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
