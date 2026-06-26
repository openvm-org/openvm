#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::TestBuilder;
#[cfg(feature = "cuda")]
use openvm_instructions::LocalOpcode;

use crate::load_sign_extend::test_utils::{
    create_seeded_rng, create_word_harness, memory_config_for, set_and_execute, VmChipTestBuilder,
    LOADW,
};
#[cfg(feature = "cuda")]
use crate::load_sign_extend::test_utils::{
    dummy_range_checker, transfer_load_sign_extend_records, GpuChipTestBuilder, GpuTestChipHarness,
    LoadSignExtendWordCoreAir, LoadSignExtendWordFiller, Rv64LoadAdapterAir,
    Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, Rv64LoadSignExtendWordAir,
    Rv64LoadSignExtendWordChip, Rv64LoadSignExtendWordChipGpu, Rv64LoadSignExtendWordExecutor,
    Rv64LoadStoreOpcode, F, MAX_INS_CAPACITY,
};

#[test]
fn rand_load_sign_extend_word_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let mut harness = create_word_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADW,
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
fn negative_load_sign_extend_address_wraparound_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let mut harness = create_word_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADW,
        Some([0xf8, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
        Some(16),
        Some(0),
    );
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
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadSignExtendWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadSignExtendWordExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadSignExtendWordChip::<F>::new(
        LoadSignExtendWordFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
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
#[test]
fn test_cuda_rand_load_sign_extend_word_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_word_harness(&tester);
    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADW,
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
