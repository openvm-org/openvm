use test_case::test_case;

use crate::loadstore::test_utils::*;

#[test]
fn positive_loadwu_shift4_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[LOADWU]));
    let mut harness = create_word_harness(&mut tester);
    set_and_execute_with(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADWU,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
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

#[test]
fn positive_storew_public_values_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[STOREW]));
    let mut harness = create_word_harness(&mut tester);
    set_and_execute_with(
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

#[test_case(LOADWU, 100)]
#[test_case(STOREW, 100)]
fn rand_loadstore_word_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_word_harness(&mut tester);
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
fn run_loadwu_storew_sanity_test() {
    let read_data = b([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = b([159, 213, 89, 34, 142, 67, 210, 88]);
    assert_eq!(
        run_write_data(LOADWU, read_data, prev_data, 0),
        b([138, 45, 202, 76, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(LOADWU, read_data, prev_data, 4),
        b([131, 74, 186, 29, 0, 0, 0, 0])
    );
    assert_eq!(
        run_write_data(STOREW, read_data, prev_data, 0),
        b([138, 45, 202, 76, 142, 67, 210, 88])
    );
    assert_eq!(
        run_write_data(STOREW, read_data, prev_data, 4),
        b([159, 213, 89, 34, 138, 45, 202, 76])
    );
}

#[test]
#[should_panic]
fn solve_loadw_rejects_shift_2() {
    run_write_data(LOADW, b([1, 2, 3, 4, 5, 6, 7, 8]), [0; BLOCK_FE_WIDTH], 2);
}

#[test]
#[should_panic]
fn solve_loadw_rejects_shift_6() {
    run_write_data(LOADW, b([1, 2, 3, 4, 5, 6, 7, 8]), [0; BLOCK_FE_WIDTH], 6);
}

#[test]
fn accepted_shift_sets() {
    let read_data = b([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]);
    for shift in 0..8 {
        let _ = run_write_data(LOADB, read_data, [0; BLOCK_FE_WIDTH], shift);
    }
    for shift in [0, 2, 4, 6] {
        let _ = run_write_data(LOADH, read_data, [0; BLOCK_FE_WIDTH], shift);
        let _ = run_write_data(LOADHU, read_data, [0; BLOCK_FE_WIDTH], shift);
    }
    for shift in [0, 4] {
        let _ = run_write_data(LOADW, read_data, [0; BLOCK_FE_WIDTH], shift);
        let _ = run_write_data(LOADWU, read_data, [0; BLOCK_FE_WIDTH], shift);
    }
}

#[cfg(feature = "cuda")]
type GpuWordHarness = GpuTestChipHarness<
    F,
    Rv64LoadStoreWordExecutor,
    Rv64LoadStoreWordAir,
    Rv64LoadStoreWordChipGpu,
    Rv64LoadStoreWordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_word_harness(tester: &GpuChipTestBuilder) -> GpuWordHarness {
    let range_checker = dummy_range_checker();
    let air = Rv64LoadStoreWordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreWordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadStoreWordChip::<F>::new(
        LoadStoreWordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadStoreWordChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADWU, 100)]
#[test_case(STOREW, 100)]
fn test_cuda_rand_loadstore_word_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_word_harness(&tester);
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
