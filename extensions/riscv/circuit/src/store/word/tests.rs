#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::TestBuilder;
#[cfg(feature = "cuda")]
use openvm_instructions::LocalOpcode;

use crate::test_utils::memory::{
    create_seeded_rng, create_store_word_harness, rv64_bytes_to_u16_block, set_and_execute_store,
    store_memory_config, store_write_data, VmChipTestBuilder, PUBLIC_VALUES_AS, STOREW,
};
#[cfg(feature = "cuda")]
use crate::test_utils::memory::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, dummy_range_checker,
    store_gpu_memory_config, transfer_store_records, Arc, BitwiseOperationLookupChip,
    GpuChipTestBuilder, GpuTestChipHarness, Rv64LoadStoreOpcode, Rv64StoreAdapterAir,
    Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller, Rv64StoreWordAir, Rv64StoreWordChip,
    Rv64StoreWordChipGpu, Rv64StoreWordExecutor, StoreWordCoreAir, StoreWordFiller, F,
    MAX_INS_CAPACITY, RV64_BYTE_BITS, RV64_MEMORY_AS,
};

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
    let (mut harness, _bitwise) = create_store_word_harness(&mut tester);
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
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreWordExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64StoreWordChip::<F>::new(
        StoreWordFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
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
