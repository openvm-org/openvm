#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::TestBuilder;
#[cfg(feature = "cuda")]
use openvm_instructions::LocalOpcode;

use crate::test_utils::memory::{
    create_seeded_rng, create_store_halfword_harness, rv64_bytes_to_u16_block,
    set_and_execute_store, store_memory_config, store_write_data, VmChipTestBuilder, STOREH,
};
#[cfg(feature = "cuda")]
use crate::test_utils::memory::{
    default_var_range_checker_bus, dummy_range_checker, store_gpu_memory_config,
    transfer_store_records, GpuChipTestBuilder, GpuTestChipHarness, Rv64LoadStoreOpcode,
    Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller, Rv64StoreHalfwordAir,
    Rv64StoreHalfwordChip, Rv64StoreHalfwordChipGpu, Rv64StoreHalfwordExecutor,
    StoreHalfwordCoreAir, StoreHalfwordFiller, F, MAX_INS_CAPACITY, PUBLIC_VALUES_AS,
    RV64_MEMORY_AS,
};

#[test]
fn rand_store_halfword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let mut harness = create_store_halfword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
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
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn run_storeh_sanity_test() {
    let read_data = rv64_bytes_to_u16_block([250, 123, 67, 198, 175, 33, 198, 250]);
    let prev_data = rv64_bytes_to_u16_block([144, 56, 175, 92, 90, 121, 64, 205]);
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 0),
        rv64_bytes_to_u16_block([250, 123, 175, 92, 90, 121, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 2),
        rv64_bytes_to_u16_block([144, 56, 250, 123, 90, 121, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 4),
        rv64_bytes_to_u16_block([144, 56, 175, 92, 250, 123, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREH, read_data, prev_data, 6),
        rv64_bytes_to_u16_block([144, 56, 175, 92, 90, 121, 250, 123])
    );
}

#[cfg(feature = "cuda")]
type GpuStoreHalfwordHarness = GpuTestChipHarness<
    F,
    Rv64StoreHalfwordExecutor,
    Rv64StoreHalfwordAir,
    Rv64StoreHalfwordChipGpu,
    Rv64StoreHalfwordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_store_halfword_harness(tester: &GpuChipTestBuilder) -> GpuStoreHalfwordHarness {
    let range_checker = dummy_range_checker();
    let air = Rv64StoreHalfwordAir::new(
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET),
    );
    let executor = Rv64StoreHalfwordExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64StoreHalfwordChip::<F>::new(
        StoreHalfwordFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64StoreHalfwordChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case::test_case(RV64_MEMORY_AS as usize)]
#[test_case::test_case(PUBLIC_VALUES_AS as usize)]
fn test_cuda_rand_store_halfword_tracegen(mem_as: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus());
    let mut harness = create_cuda_store_halfword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            STOREH,
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
