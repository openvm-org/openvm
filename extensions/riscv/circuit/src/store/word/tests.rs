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
use openvm_instructions::{
    instruction::Instruction,
    riscv::{MEMORY_AS, REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::LoadStoreOpcode::{self, STOREW};
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
        bytes_to_u16_block, StoreMultiByteAdapterAir, StoreMultiByteAdapterCols,
        StoreMultiByteAdapterFiller, BYTE_BITS,
    },
    store::{
        common::store_write_data, StoreWordAir, StoreWordChip, StoreWordCoreAir, StoreWordExecutor,
        StoreWordFiller,
    },
    test_utils::memory::{
        set_and_execute_store, store_memory_config, F, MAX_INS_CAPACITY, WRAPPED_BASE_CASES,
    },
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{
    store::StoreWordChipGpu,
    test_utils::memory::{dummy_range_checker, store_gpu_memory_config},
};

type StoreWordHarness = TestChipHarness<F, StoreWordExecutor, StoreWordAir, StoreWordChip<F>>;

fn create_store_word_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreWordHarness,
    (
        BitwiseOperationLookupAir<BYTE_BITS>,
        SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(bitwise_bus));
    let air = StoreWordAir::new(
        StoreMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreWordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = StoreWordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let chip = StoreWordChip::<F>::new(
        StoreWordFiller::new(
            StoreMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        StoreWordHarness::with_capacity(
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
fn rand_store_word_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_word_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREW,
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
        &mut harness.preflight,
        &mut rng,
        STOREW,
        Some([0xf8, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
        Some(16),
        Some(0),
    );
}

#[test]
#[should_panic(expected = "effective address exceeds implemented memory address space")]
fn negative_store_address_underflow_test() {
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, _) = create_store_word_harness(&mut tester);
    let rs1_ptr = 8;
    tester.write_bytes(REGISTER_AS as usize, rs1_ptr, [F::ZERO; 8]);

    tester.execute(
        &mut harness.executor,
        &mut harness.preflight,
        &Instruction::from_usize(
            STOREW.global_opcode(),
            [
                0,
                rs1_ptr,
                u16::MAX as usize,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
                1,
                1,
            ],
        ),
    );
}

#[test]
fn run_storew_sanity_test() {
    let read_data = bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = [
        bytes_to_u16_block([159, 213, 89, 34, 142, 67, 210, 88]),
        bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 0),
        [
            bytes_to_u16_block([138, 45, 202, 76, 142, 67, 210, 88]),
            prev_data[1]
        ]
    );
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 4),
        [
            bytes_to_u16_block([159, 213, 89, 34, 138, 45, 202, 76]),
            prev_data[1]
        ]
    );
    // Misaligned within one block.
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 3),
        [
            bytes_to_u16_block([159, 213, 89, 138, 45, 202, 76, 88]),
            prev_data[1]
        ]
    );
    // Misaligned across the block boundary.
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 6),
        [
            bytes_to_u16_block([159, 213, 89, 34, 142, 67, 138, 45]),
            bytes_to_u16_block([202, 76, 17, 203, 44, 118, 240, 5]),
        ]
    );
}

#[test]
fn positive_storew_wrapped_base_test() {
    // RV64I: rs1 + sign_extend(imm) wraps modulo 2^64, so a base with a nonzero upper word can
    // still address the 32-bit memory space.
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_word_harness(&mut tester);
    for (rs1, imm, imm_sign) in WRAPPED_BASE_CASES {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREW,
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

fn negative_storew_base_high_flags_test_case(rs1: u64, imm: u32, imm_sign: u32, rs1_hi: [u32; 2]) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_word_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        STOREW,
        Some(rs1.to_le_bytes()),
        Some(imm),
        Some(imm_sign),
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, _) = trace_row.split_at_mut(adapter_width);
        let adapter: &mut StoreMultiByteAdapterCols<F> = adapter_row.borrow_mut();
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
fn negative_storew_base_high_flags_test() {
    // A zero upper word cannot be claimed as all ones or as one.
    for rs1_hi in [[1, 0], [0, 1], [1, 1]] {
        negative_storew_base_high_flags_test_case(0x1000, 8, 0, rs1_hi);
    }
    // A wrapped base cannot be claimed as a zero upper word, nor as the other wrap.
    negative_storew_base_high_flags_test_case(-8i64 as u64, 16, 0, [0, 0]);
    negative_storew_base_high_flags_test_case(-8i64 as u64, 16, 0, [0, 1]);
    negative_storew_base_high_flags_test_case((1 << 32) + 8, 0xffe8, 1, [0, 0]);
    negative_storew_base_high_flags_test_case((1 << 32) + 8, 0xffe8, 1, [1, 0]);
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuStoreWordHarness =
    GpuTestChipHarness<F, StoreWordExecutor, StoreWordAir, StoreWordChipGpu, StoreWordChip<F>>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_store_word_harness(tester: &GpuChipTestBuilder) -> GpuStoreWordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = StoreWordAir::new(
        StoreMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreWordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = StoreWordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
    let cpu_chip = StoreWordChip::<F>::new(
        StoreWordFiller::new(
            StoreMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = StoreWordChipGpu::new(
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
fn test_cuda_rand_store_word_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_store_word_harness(&tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STOREW,
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
            STOREW,
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
