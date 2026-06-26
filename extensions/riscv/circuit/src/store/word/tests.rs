#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::TestBuilder;
#[cfg(feature = "cuda")]
use openvm_instructions::LocalOpcode;

use crate::test_utils::memory::{
    b, create_seeded_rng, create_store_word_harness, set_and_execute_store, store_memory_config,
    store_write_data, VmChipTestBuilder, PUBLIC_VALUES_AS, STOREW,
};
#[cfg(feature = "cuda")]
use crate::test_utils::memory::{
    default_var_range_checker_bus, dummy_range_checker, store_gpu_memory_config,
    transfer_store_records, GpuChipTestBuilder, GpuTestChipHarness, Rv64LoadStoreOpcode,
    Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller, Rv64StoreWordAir,
    Rv64StoreWordChip, Rv64StoreWordChipGpu, Rv64StoreWordExecutor, StoreWordCoreAir,
    StoreWordFiller, F, MAX_INS_CAPACITY, RV64_MEMORY_AS,
};

#[test]
fn positive_storew_public_values_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let mut harness = create_store_word_harness(&mut tester);
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
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn rand_store_word_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let mut harness = create_store_word_harness(&mut tester);
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
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
#[should_panic(expected = "effective address exceeds implemented memory address space")]
fn negative_store_address_wraparound_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let mut harness = create_store_word_harness(&mut tester);
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
    let read_data = b([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = b([159, 213, 89, 34, 142, 67, 210, 88]);
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 0),
        b([138, 45, 202, 76, 142, 67, 210, 88])
    );
    assert_eq!(
        store_write_data(STOREW, read_data, prev_data, 4),
        b([159, 213, 89, 34, 138, 45, 202, 76])
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
    let air = Rv64StoreWordAir::new(
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET),
    );
    let executor = Rv64StoreWordExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64StoreWordChip::<F>::new(
        StoreWordFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64StoreWordChipGpu::new(
        tester.range_checker(),
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
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus());
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
