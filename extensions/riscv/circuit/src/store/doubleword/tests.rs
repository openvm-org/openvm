use std::sync::Arc;

#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
    GpuTestChipHarness,
};
use openvm_circuit::arch::testing::{
    TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_instructions::PUBLIC_VALUES_AS;
use openvm_instructions::{riscv::RV64_MEMORY_AS, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, STORED};
use openvm_stark_sdk::utils::create_seeded_rng;

use super::trace::generate_trace_from_postflight;
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, Rv64StoreMultiByteAdapterAir, Rv64StoreMultiByteAdapterFiller,
        RV64_BYTE_BITS,
    },
    store::{
        common::store_write_data, core::fill_padding_row, Rv64StoreDoublewordAir,
        Rv64StoreDoublewordChip, Rv64StoreDoublewordExecutor, StoreDoublewordCoreAir,
        StoreDoublewordFiller,
    },
    test_utils::memory::{set_and_execute_store, store_memory_config, F, MAX_INS_CAPACITY},
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{
    store::Rv64StoreDoublewordChipGpu,
    test_utils::memory::{dummy_range_checker, store_gpu_memory_config},
};

type StoreDoublewordHarness = TestChipHarness<
    F,
    Rv64StoreDoublewordExecutor,
    Rv64StoreDoublewordAir,
    Rv64StoreDoublewordChip<F>,
>;

fn create_store_doubleword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreDoublewordHarness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let air = Rv64StoreDoublewordAir::new(
        Rv64StoreMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreDoublewordExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let chip = Rv64StoreDoublewordChip::<F>::new(
        StoreDoublewordFiller::new(
            Rv64StoreMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    (
        StoreDoublewordHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            generate_trace_from_postflight,
        )
        .with_padding(fill_padding_row),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[test]
fn rand_store_doubleword_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_doubleword_harness(&mut tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
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
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn positive_stored_max_address_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_doubleword_harness(&mut tester);
    // The default config exposes the full 2^32-byte memory AS; deterministically store to the
    // last addressable doubleword (byte address 2^32 - 8).
    let rs1 = (u32::MAX - 7).to_le_bytes();
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        STORED,
        Some([rs1[0], rs1[1], rs1[2], rs1[3], 0, 0, 0, 0]),
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
fn positive_stored_pointer_limb_boundary_cross_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_doubleword_harness(&mut tester);
    // ptr = 0xfff9: the crossing block starts at 0x10000, exercising the pointer limb carry.
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        STORED,
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
fn run_stored_sanity_test() {
    let read_data = rv64_bytes_to_u16_block([138, 45, 202, 76, 131, 74, 186, 29]);
    let prev_data = [
        rv64_bytes_to_u16_block([159, 213, 89, 34, 142, 67, 210, 88]),
        rv64_bytes_to_u16_block([61, 92, 17, 203, 44, 118, 240, 5]),
    ];
    assert_eq!(
        store_write_data(STORED, read_data, prev_data, 0),
        [read_data, prev_data[1]]
    );
    // Every nonzero doubleword shift crosses the block boundary.
    assert_eq!(
        store_write_data(STORED, read_data, prev_data, 5),
        [
            rv64_bytes_to_u16_block([159, 213, 89, 34, 142, 138, 45, 202]),
            rv64_bytes_to_u16_block([76, 131, 74, 186, 29, 118, 240, 5]),
        ]
    );
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuStoreDoublewordHarness = GpuTestChipHarness<
    F,
    Rv64StoreDoublewordExecutor,
    Rv64StoreDoublewordAir,
    Rv64StoreDoublewordChipGpu,
    Rv64StoreDoublewordChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_store_doubleword_harness(tester: &GpuChipTestBuilder) -> GpuStoreDoublewordHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64StoreDoublewordAir::new(
        Rv64StoreMultiByteAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreDoublewordExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
    let cpu_chip = Rv64StoreDoublewordChip::<F>::new(
        StoreDoublewordFiller::new(
            Rv64StoreMultiByteAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64StoreDoublewordChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
        .with_trace_generators(
            generate_trace_from_postflight,
            |chip, program, transcript, plan| {
                chip.generate_proving_ctx_from_postflight(program, transcript, plan)
            },
        )
        .with_padding(fill_padding_row)
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case::test_case(RV64_MEMORY_AS as usize)]
#[test_case::test_case(PUBLIC_VALUES_AS as usize)]
fn test_cuda_rand_store_doubleword_tracegen(mem_as: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::new(store_gpu_memory_config(), default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_store_doubleword_harness(&tester);
    for _ in 0..100 {
        set_and_execute_store(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            STORED,
            None,
            None,
            None,
            Some(mem_as),
        );
    }
    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
