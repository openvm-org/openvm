use test_case::test_case;

use super::*;

#[test]
fn positive_loadhu_shift6_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[LOADHU]));
    let mut harness = create_halfword_harness(&mut tester);
    set_and_execute_with(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        Some([6, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(2),
    );
    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(LOADHU, 100)]
#[test_case(STOREH, 100)]
fn rand_loadstore_halfword_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_halfword_harness(&mut tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
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
    let read_data = b([250, 123, 67, 198, 175, 33, 198, 250]);
    let prev_data = b([144, 56, 175, 92, 90, 121, 64, 205]);
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 0),
        b([250, 123, 175, 92, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 2),
        b([144, 56, 250, 123, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 4),
        b([144, 56, 175, 92, 250, 123, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 6),
        b([144, 56, 175, 92, 90, 121, 250, 123])
    );
}

#[test]
fn run_loadhu_sanity_test() {
    let read_data = b([175, 33, 198, 250, 131, 74, 186, 29]);
    let prev_data = b([90, 121, 64, 205, 142, 67, 210, 88]);
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 0),
        b([175, 33, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 2),
        b([198, 250, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 4),
        b([131, 74, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 6),
        b([186, 29, 0, 0, 0, 0, 0, 0])
    );
}

#[cfg(feature = "cuda")]
type GpuHalfwordHarness = GpuTestChipHarness<
    F,
    Rv64LoadStoreHalfwordExecutor,
    Rv64LoadStoreHalfwordAir,
    Rv64LoadStoreHalfwordChipGpu,
    Rv64LoadStoreHalfwordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_halfword_harness(tester: &GpuChipTestBuilder) -> GpuHalfwordHarness {
    let range_checker = dummy_range_checker();
    let air = Rv64LoadStoreHalfwordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreHalfwordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadStoreHalfwordChip::<F>::new(
        LoadStoreHalfwordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadStoreHalfwordChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADHU, 100)]
#[test_case(STOREH, 100)]
fn test_cuda_rand_loadstore_halfword_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_halfword_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute_with(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            None,
            None,
            None,
            Some(2),
        );
    }
    transfer_loadstore_records(&mut harness);
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
