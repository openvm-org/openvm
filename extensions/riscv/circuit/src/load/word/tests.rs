#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::TestBuilder;
#[cfg(feature = "cuda")]
use openvm_instructions::LocalOpcode;

use crate::test_utils::memory::{
    create_seeded_rng, create_word_harness, load_memory_config, load_sign_extend_write_data,
    load_write_data, rv64_bytes_to_u16_block, set_and_execute_load, VmChipTestBuilder, LOADB,
    LOADH, LOADHU, LOADW, LOADWU, RV64_MEMORY_AS,
};
#[cfg(feature = "cuda")]
use crate::test_utils::memory::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, dummy_range_checker,
    load_gpu_memory_config, transfer_load_records, Arc, BitwiseOperationLookupChip,
    GpuChipTestBuilder, GpuTestChipHarness, LoadWordCoreAir, LoadWordFiller, Rv64LoadAdapterAir,
    Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, Rv64LoadStoreOpcode, Rv64LoadWordAir,
    Rv64LoadWordChip, Rv64LoadWordChipGpu, Rv64LoadWordExecutor, F, MAX_INS_CAPACITY,
    RV64_BYTE_BITS,
};

#[test]
fn positive_loadwu_shift4_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_word_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADWU,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(RV64_MEMORY_AS as usize),
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
fn rand_load_word_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_word_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADWU,
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
fn run_loadwu_sanity_test() {
    let read_data = [
        rv64_bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        load_write_data(LOADWU, read_data, 0),
        rv64_bytes_to_u16_block([138, 45, 202, 76, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADWU, read_data, 4),
        rv64_bytes_to_u16_block([131, 74, 186, 29, 0, 0, 0, 0])
    );
    // Misaligned within one block.
    assert_eq!(
        load_write_data(LOADWU, read_data, 3),
        rv64_bytes_to_u16_block([76, 131, 74, 186, 0, 0, 0, 0])
    );
    // Misaligned across the block boundary.
    assert_eq!(
        load_write_data(LOADWU, read_data, 6),
        rv64_bytes_to_u16_block([186, 29, 61, 92, 0, 0, 0, 0])
    );
}

#[test]
#[should_panic]
fn solve_loadw_rejects_shift_2() {
    load_sign_extend_write_data(LOADW, rv64_bytes_to_u16_block([1, 2, 3, 4, 5, 6, 7, 8]), 2);
}

#[test]
#[should_panic]
fn solve_loadw_rejects_shift_6() {
    load_sign_extend_write_data(LOADW, rv64_bytes_to_u16_block([1, 2, 3, 4, 5, 6, 7, 8]), 6);
}

#[test]
fn accepted_shift_sets() {
    let read_data = rv64_bytes_to_u16_block([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]);
    let read_blocks = [
        read_data,
        rv64_bytes_to_u16_block([0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0, 0x11]),
    ];
    for shift in 0..8 {
        let _ = load_sign_extend_write_data(LOADB, read_data, shift);
        let _ = load_write_data(LOADHU, read_blocks, shift);
        let _ = load_write_data(LOADWU, read_blocks, shift);
    }
    for shift in [0, 2, 4, 6] {
        let _ = load_sign_extend_write_data(LOADH, read_data, shift);
    }
    for shift in [0, 4] {
        let _ = load_sign_extend_write_data(LOADW, read_data, shift);
    }
}

#[cfg(feature = "cuda")]
type GpuWordHarness = GpuTestChipHarness<
    F,
    Rv64LoadWordExecutor,
    Rv64LoadWordAir,
    Rv64LoadWordChipGpu,
    Rv64LoadWordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_word_harness(tester: &GpuChipTestBuilder) -> GpuWordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadWordAir::new(
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadWordExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadWordChip::<F>::new(
        LoadWordFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadWordChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_load_word_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(load_gpu_memory_config(), default_var_range_checker_bus());
    let mut harness = create_cuda_word_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADWU,
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
