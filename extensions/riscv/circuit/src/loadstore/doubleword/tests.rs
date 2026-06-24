use test_case::test_case;

use crate::loadstore::test_utils::*;

#[test_case(LOADD, 100)]
#[test_case(STORED, 100)]
fn rand_loadstore_doubleword_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_doubleword_harness(&mut tester);
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
fn run_loadd_stored_sanity_test() {
    let read_data = b([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = b([159, 213, 89, 34, 142, 67, 210, 88]);
    assert_eq!(run_write_data(LOADD, read_data, prev_data, 0), read_data);
    assert_eq!(run_write_data(STORED, read_data, prev_data, 0), read_data);
}

#[cfg(feature = "cuda")]
type GpuDoublewordHarness = GpuTestChipHarness<
    F,
    Rv64LoadStoreDoublewordExecutor,
    Rv64LoadStoreDoublewordAir,
    Rv64LoadStoreDoublewordChipGpu,
    Rv64LoadStoreDoublewordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_doubleword_harness(tester: &GpuChipTestBuilder) -> GpuDoublewordHarness {
    let range_checker = dummy_range_checker();
    let air = Rv64LoadStoreDoublewordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreDoublewordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadStoreDoublewordChip::<F>::new(
        LoadStoreDoublewordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadStoreDoublewordChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADD, 100)]
#[test_case(STORED, 100)]
fn test_cuda_rand_loadstore_doubleword_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_doubleword_harness(&tester);
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
