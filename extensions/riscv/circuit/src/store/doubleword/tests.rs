#[cfg(feature = "cuda")]
use openvm_circuit::arch::testing::{
    default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness,
};
use openvm_circuit::arch::testing::{TestBuilder, TestChipHarness, VmChipTestBuilder};
use openvm_instructions::LocalOpcode;
#[cfg(feature = "cuda")]
use openvm_instructions::{riscv::RV64_MEMORY_AS, PUBLIC_VALUES_AS};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, STORED};
use openvm_stark_sdk::utils::create_seeded_rng;

use crate::{
    adapters::{
        rv64_bytes_to_u16_block, Rv64StoreAdapterAir, Rv64StoreAdapterExecutor,
        Rv64StoreAdapterFiller,
    },
    store::{
        common::store_write_data, Rv64StoreDoublewordAir, Rv64StoreDoublewordChip,
        Rv64StoreDoublewordExecutor, StoreDoublewordCoreAir, StoreDoublewordFiller,
    },
    test_utils::memory::{set_and_execute_store, store_memory_config, F, MAX_INS_CAPACITY},
};
#[cfg(feature = "cuda")]
use crate::{
    store::Rv64StoreDoublewordChipGpu,
    test_utils::memory::{dummy_range_checker, store_gpu_memory_config, transfer_store_records},
};

type StoreDoublewordHarness = TestChipHarness<
    F,
    Rv64StoreDoublewordExecutor,
    Rv64StoreDoublewordAir,
    Rv64StoreDoublewordChip<F>,
>;

fn create_store_doubleword_harness(tester: &mut VmChipTestBuilder<F>) -> StoreDoublewordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64StoreDoublewordAir::new(
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET),
    );
    let executor = Rv64StoreDoublewordExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64StoreDoublewordChip::<F>::new(
        StoreDoublewordFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    StoreDoublewordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[test]
fn rand_store_doubleword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let mut harness = create_store_doubleword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            STORED,
            None,
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
fn run_stored_sanity_test() {
    let read_data = rv64_bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = rv64_bytes_to_u16_block([159, 213, 89, 34, 142, 67, 210, 88]);
    assert_eq!(store_write_data(STORED, read_data, prev_data, 0), read_data);
}

#[cfg(feature = "cuda")]
type GpuStoreDoublewordHarness = GpuTestChipHarness<
    F,
    Rv64StoreDoublewordExecutor,
    Rv64StoreDoublewordAir,
    Rv64StoreDoublewordChipGpu,
    Rv64StoreDoublewordChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_store_doubleword_harness(tester: &GpuChipTestBuilder) -> GpuStoreDoublewordHarness {
    let range_checker = dummy_range_checker();
    let air = Rv64StoreDoublewordAir::new(
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET),
    );
    let executor = Rv64StoreDoublewordExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64StoreDoublewordChip::<F>::new(
        StoreDoublewordFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64StoreDoublewordChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        Default::default(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case::test_case(RV64_MEMORY_AS as usize)]
#[test_case::test_case(PUBLIC_VALUES_AS as usize)]
fn test_cuda_rand_store_doubleword_tracegen(mem_as: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus());
    let mut harness = create_cuda_store_doubleword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            STORED,
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
