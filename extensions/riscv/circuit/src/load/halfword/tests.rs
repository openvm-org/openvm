#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::TestBuilder;
#[cfg(feature = "cuda")]
use openvm_instructions::LocalOpcode;

use crate::test_utils::memory::{
    create_halfword_harness, create_seeded_rng, load_memory_config, load_write_data,
    rv64_bytes_to_u16_block, set_and_execute_load, VmChipTestBuilder, LOADHU, RV64_MEMORY_AS,
};
#[cfg(feature = "cuda")]
use crate::test_utils::memory::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, dummy_range_checker,
    load_gpu_memory_config, transfer_load_records, Arc, BitwiseOperationLookupChip,
    GpuChipTestBuilder, GpuTestChipHarness, LoadHalfwordCoreAir, LoadHalfwordFiller,
    Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, Rv64LoadHalfwordAir,
    Rv64LoadHalfwordChip, Rv64LoadHalfwordChipGpu, Rv64LoadHalfwordExecutor, Rv64LoadStoreOpcode,
    F, MAX_INS_CAPACITY, RV64_BYTE_BITS,
};

#[test]
fn positive_loadhu_shift6_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        Some([6, 0, 0, 0, 0, 0, 0, 0]),
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
fn rand_load_halfword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADHU,
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
fn run_loadhu_sanity_test() {
    let read_data = [
        rv64_bytes_to_u16_block([175, 33, 198, 250, 131, 74, 186, 29]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        load_write_data(LOADHU, read_data, 0),
        rv64_bytes_to_u16_block([175, 33, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 2),
        rv64_bytes_to_u16_block([198, 250, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 4),
        rv64_bytes_to_u16_block([131, 74, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        load_write_data(LOADHU, read_data, 6),
        rv64_bytes_to_u16_block([186, 29, 0, 0, 0, 0, 0, 0])
    );
    // Misaligned within one block.
    assert_eq!(
        load_write_data(LOADHU, read_data, 3),
        rv64_bytes_to_u16_block([250, 131, 0, 0, 0, 0, 0, 0])
    );
    // Misaligned across the block boundary.
    assert_eq!(
        load_write_data(LOADHU, read_data, 7),
        rv64_bytes_to_u16_block([29, 61, 0, 0, 0, 0, 0, 0])
    );
}

#[cfg(feature = "cuda")]
type GpuHalfwordHarness = GpuTestChipHarness<
    F,
    Rv64LoadHalfwordExecutor,
    Rv64LoadHalfwordAir,
    Rv64LoadHalfwordChipGpu,
    Rv64LoadHalfwordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_halfword_harness(tester: &GpuChipTestBuilder) -> GpuHalfwordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadHalfwordAir::new(
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadHalfwordExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadHalfwordChip::<F>::new(
        LoadHalfwordFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadHalfwordChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_load_halfword_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(load_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_halfword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADHU,
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
