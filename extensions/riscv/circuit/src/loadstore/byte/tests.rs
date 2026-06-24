use test_case::test_case;

use crate::loadstore::test_utils::*;

#[test_case(LOADBU, 100)]
#[test_case(STOREB, 100)]
fn rand_loadstore_byte_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
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
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn run_storeb_sanity_test() {
    let read_data = b([221, 104, 58, 147, 175, 33, 198, 250]);
    let prev_data = b([199, 83, 243, 12, 90, 121, 64, 205]);
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 0),
        b([221, 83, 243, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 1),
        b([199, 221, 243, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 2),
        b([199, 83, 221, 12, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 3),
        b([199, 83, 243, 221, 90, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 4),
        b([199, 83, 243, 12, 221, 121, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 5),
        b([199, 83, 243, 12, 90, 221, 64, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 6),
        b([199, 83, 243, 12, 90, 121, 221, 205])
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 7),
        b([199, 83, 243, 12, 90, 121, 64, 221])
    );
}

#[test]
fn run_loadbu_sanity_test() {
    let read_data = b([131, 74, 186, 29, 138, 45, 202, 76]);
    let prev_data = b([142, 67, 210, 88, 159, 213, 89, 34]);
    for (shift, expected) in [
        (0, [131, 0, 0, 0, 0, 0, 0, 0]),
        (1, [74, 0, 0, 0, 0, 0, 0, 0]),
        (2, [186, 0, 0, 0, 0, 0, 0, 0]),
        (3, [29, 0, 0, 0, 0, 0, 0, 0]),
        (4, [138, 0, 0, 0, 0, 0, 0, 0]),
        (5, [45, 0, 0, 0, 0, 0, 0, 0]),
        (6, [202, 0, 0, 0, 0, 0, 0, 0]),
        (7, [76, 0, 0, 0, 0, 0, 0, 0]),
    ] {
        assert_eq!(
            run_write_data(LOADBU, read_data, prev_data, shift),
            b(expected)
        );
    }
}

#[cfg(feature = "cuda")]
type GpuByteHarness = GpuTestChipHarness<
    F,
    Rv64LoadStoreByteExecutor,
    Rv64LoadStoreByteAir,
    Rv64LoadStoreByteChipGpu,
    Rv64LoadStoreByteChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_byte_harness(tester: &GpuChipTestBuilder) -> GpuByteHarness {
    let range_checker = dummy_range_checker();
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let air = Rv64LoadStoreByteAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreByteCoreAir::new(
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.bus(),
            range_checker.bus(),
        ),
    );
    let executor = Rv64LoadStoreByteExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64LoadStoreByteChip::<F>::new(
        LoadStoreByteFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip,
            range_checker,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LoadStoreByteChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADBU, 100)]
#[test_case(STOREB, 100)]
fn test_cuda_rand_loadstore_byte_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_byte_harness(&tester);
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
