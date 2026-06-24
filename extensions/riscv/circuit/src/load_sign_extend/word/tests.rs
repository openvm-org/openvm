use test_case::test_case;

use crate::load_sign_extend::test_utils::*;

#[test_case(LOADW, 100)]
fn rand_load_sign_extend_word_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let mut harness = create_word_harness(&mut tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
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
fn positive_loadw_shift4_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let mut harness = create_word_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADW,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
    );
    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
type GpuWordHarness = GpuTestChipHarness<
    F,
    Rv64LoadSignExtendWordExecutor,
    Rv64LoadSignExtendWordAir,
    Rv64LoadSignExtendWordChipGpu,
    Rv64LoadSignExtendWordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_word_harness(tester: &GpuChipTestBuilder) -> GpuWordHarness {
    let range_checker = dummy_range_checker();
    let air = Rv64LoadSignExtendWordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadSignExtendWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadSignExtendWordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadSignExtendWordChip::<F>::new(
        LoadSignExtendWordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadSignExtendWordChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADW, 100)]
fn test_cuda_rand_load_sign_extend_word_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_word_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }
    transfer_load_sign_extend_records(&mut harness);
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
