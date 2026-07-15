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
use openvm_instructions::{
    instruction::Instruction,
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
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        Default::default(),
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
