use crate::test_utils::memory::{
    b, create_seeded_rng, create_store_byte_harness, set_and_execute_store, store_memory_config,
    store_write_data, VmChipTestBuilder, STOREB,
};
#[cfg(feature = "cuda")]
use crate::test_utils::memory::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, dummy_range_checker,
    store_gpu_memory_config, transfer_store_records, Arc, BitwiseOperationLookupChip,
    GpuChipTestBuilder, GpuTestChipHarness, Rv64LoadStoreOpcode, Rv64StoreAdapterAir,
    Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller, Rv64StoreByteAir, Rv64StoreByteChip,
    Rv64StoreByteChipGpu, Rv64StoreByteExecutor, StoreByteCoreAir, StoreByteFiller, F,
    MAX_INS_CAPACITY, PUBLIC_VALUES_AS, RV64_BYTE_BITS, RV64_MEMORY_AS,
};

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
    let read_data = b([221, 104, 58, 147, 175, 33, 198, 250]);
    let prev_data = b([199, 83, 243, 12, 90, 121, 64, 205]);
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 0),
        b([221, 83, 243, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 1),
        b([199, 221, 243, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 2),
        b([199, 83, 221, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 3),
        b([199, 83, 243, 221, 90, 121, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 4),
        b([199, 83, 243, 12, 221, 121, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 5),
        b([199, 83, 243, 12, 90, 221, 64, 205])
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 6),
        b([199, 83, 243, 12, 90, 121, 221, 205])
    );
    assert_eq!(
        store_write_data(STOREB, read_data, prev_data, 7),
        b([199, 83, 243, 12, 90, 121, 64, 221])
    );
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
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreByteExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64StoreByteChip::<F>::new(
        StoreByteFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64StoreByteChipGpu::new(
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
    transfer_store_records(&mut harness);
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
