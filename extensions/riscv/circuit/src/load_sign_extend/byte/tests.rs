use test_case::test_case;

use crate::load_sign_extend::test_utils::*;

#[test_case(LOADB, 100)]
fn rand_load_sign_extend_byte_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
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
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn positive_loadb_shift7_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADB,
        Some([7, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
    );
    tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
type GpuByteHarness = GpuTestChipHarness<
    F,
    Rv64LoadSignExtendByteExecutor,
    Rv64LoadSignExtendByteAir,
    Rv64LoadSignExtendByteChipGpu,
    Rv64LoadSignExtendByteChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_byte_harness(tester: &GpuChipTestBuilder) -> GpuByteHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadSignExtendByteAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadSignExtendByteCoreAir::new(
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.bus(),
            range_checker.bus(),
        ),
    );
    let executor = Rv64LoadSignExtendByteExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadSignExtendByteChip::<F>::new(
        LoadSignExtendByteFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadSignExtendByteChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADB, 100)]
fn test_cuda_rand_load_sign_extend_byte_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_byte_harness(&tester);
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
