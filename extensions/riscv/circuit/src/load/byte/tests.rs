use crate::test_utils::memory::{
    b, create_byte_harness, create_seeded_rng, load_memory_config, load_write_data,
    set_and_execute_load, VmChipTestBuilder, LOADBU,
};
#[cfg(feature = "cuda")]
use crate::test_utils::memory::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, dummy_range_checker,
    load_gpu_memory_config, transfer_load_records, Arc, BitwiseOperationLookupChip,
    GpuChipTestBuilder, GpuTestChipHarness, LoadByteCoreAir, LoadByteFiller, Rv64LoadAdapterAir,
    Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, Rv64LoadByteAir, Rv64LoadByteChip,
    Rv64LoadByteChipGpu, Rv64LoadByteExecutor, Rv64LoadStoreOpcode, F, MAX_INS_CAPACITY,
    RV64_BYTE_BITS, RV64_MEMORY_AS,
};

#[test]
fn rand_load_byte_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADBU,
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
fn negative_load_address_wraparound_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, _bitwise) = create_byte_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADBU,
        Some([0xf8, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
        Some(16),
        Some(0),
        None,
    );
}

#[test]
fn run_loadbu_sanity_test() {
    let read_data = b([131, 74, 186, 29, 138, 45, 202, 76]);
    for (shift, expected) in [
        (0, [131, 0, 0, 0, 0, 0, 0, 0]),
        (1, [74, 0, 0, 0, 0, 0, 0, 0]),
        (2, [186, 0, 0, 0, 0, 0, 0, 0]),
        (3, [29, 0, 0, 0, 0, 0, 0, 0]),
        (4, [138, 0, 0, 0, 0, 0, 0, 0]),
        (5, [45, 0, 0, 0, 0, 0, 0, 0]),
        (6, [202, 0, 0, 0, 0, 0, 0, 0]),
        (7, [76, 0, 0, 0, 0, 0, 0, 0]),
    ] {
        assert_eq!(load_write_data(LOADBU, read_data, shift), b(expected));
    }
}

#[cfg(feature = "cuda")]
type GpuByteHarness = GpuTestChipHarness<
    F,
    Rv64LoadByteExecutor,
    Rv64LoadByteAir,
    Rv64LoadByteChipGpu,
    Rv64LoadByteChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_byte_harness(tester: &GpuChipTestBuilder) -> GpuByteHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadByteAir::new(
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadByteExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadByteChip::<F>::new(
        LoadByteFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadByteChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_load_byte_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(load_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_byte_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADBU,
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
