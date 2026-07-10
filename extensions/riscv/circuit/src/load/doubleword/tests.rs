#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::TestBuilder;
#[cfg(feature = "cuda")]
use openvm_instructions::LocalOpcode;

use crate::test_utils::memory::{
    create_doubleword_harness, create_seeded_rng, load_memory_config, load_write_data,
    rv64_bytes_to_u16_block, set_and_execute_load, VmChipTestBuilder, LOADD, RV64_MEMORY_AS,
};
#[cfg(feature = "cuda")]
use crate::test_utils::memory::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, dummy_range_checker,
    load_gpu_memory_config, transfer_load_records, Arc, BitwiseOperationLookupChip,
    GpuChipTestBuilder, GpuTestChipHarness, LoadDoublewordCoreAir, LoadDoublewordFiller,
    Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, Rv64LoadDoublewordAir,
    Rv64LoadDoublewordChip, Rv64LoadDoublewordChipGpu, Rv64LoadDoublewordExecutor,
    Rv64LoadStoreOpcode, F, MAX_INS_CAPACITY, RV64_BYTE_BITS,
};

#[test]
fn rand_load_doubleword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            LOADD,
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
fn positive_loadd_page_boundary_cross_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    // ptr = 0xfff9: the crossing block starts at 0x10000, exercising the pointer limb carry.
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADD,
        Some([0xf9, 0xff, 0x00, 0x00, 0, 0, 0, 0]),
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
fn run_loadd_sanity_test() {
    let read_data = [
        rv64_bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(load_write_data(LOADD, read_data, 0), read_data[0]);
    // Every nonzero doubleword shift crosses the block boundary.
    assert_eq!(
        load_write_data(LOADD, read_data, 5),
        rv64_bytes_to_u16_block([74, 186, 29, 61, 92, 17, 203, 44])
    );
}

#[cfg(feature = "cuda")]
type GpuDoublewordHarness = GpuTestChipHarness<
    F,
    Rv64LoadDoublewordExecutor,
    Rv64LoadDoublewordAir,
    Rv64LoadDoublewordChipGpu,
    Rv64LoadDoublewordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_doubleword_harness(tester: &GpuChipTestBuilder) -> GpuDoublewordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadDoublewordAir::new(
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadDoublewordExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadDoublewordChip::<F>::new(
        LoadDoublewordFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadDoublewordChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_load_doubleword_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(load_gpu_memory_config(), default_var_range_checker_bus());
    let mut harness = create_cuda_doubleword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_load(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            LOADD,
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
