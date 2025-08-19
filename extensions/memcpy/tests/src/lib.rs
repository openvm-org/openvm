#[cfg(test)]
mod tests {
    use std::{array, borrow::BorrowMut, sync::Arc};

    use openvm_circuit::arch::testing::{TestChipHarness, VmChipTestBuilder};
    use openvm_circuit_primitives::var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerAir, VariableRangeCheckerBus,
        VariableRangeCheckerChip,
    };
    use openvm_instructions::LocalOpcode;
    use openvm_memcpy_circuit::{
        bus::MemcpyBus,
        extension::{Memcpy, MemcpyCpuProverExt},
        MemcpyIterAir, MemcpyIterCols, MemcpyIterExecutor, MemcpyIterFiller, MemcpyLoopAir,
        MemcpyLoopChip, MemcpyLoopExecutor, MEMCPY_LOOP_LIMB_BITS, MEMCPY_LOOP_NUM_LIMBS,
    };
    use openvm_memcpy_transpiler::Rv32MemcpyOpcode;
    use openvm_stark_backend::{
        p3_air::BaseAir,
        p3_field::{FieldAlgebra, PrimeField32},
        p3_matrix::{
            dense::{DenseMatrix, RowMajorMatrix},
            Matrix,
        },
        utils::disable_debug_builder,
    };
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    const MAX_INS_CAPACITY: usize = 128;
    type F = BabyBear;
    type Harness = TestChipHarness<F, MemcpyIterExecutor, MemcpyIterAir, MemcpyLoopChip<F>>;

    fn create_test_chip(
        tester: &VmChipTestBuilder<F>,
    ) -> (
        Harness,
        (
            VariableRangeCheckerAir<MEMCPY_LOOP_LIMB_BITS>,
            SharedVariableRangeCheckerChip<MEMCPY_LOOP_LIMB_BITS>,
        ),
    ) {
        let range_bus = VariableRangeCheckerBus::new(tester.new_bus_idx());
        let range_chip = Arc::new(VariableRangeCheckerChip::<MEMCPY_LOOP_LIMB_BITS>::new(
            range_bus,
        ));

        let memcpy_bus = MemcpyBus::new(tester.new_bus_idx());

        let air = MemcpyIterAir::new(
            tester.memory_bridge(),
            range_bus,
            memcpy_bus,
            tester.pointer_max_bits(),
        );
        let executor = MemcpyIterExecutor::new(Rv32MemcpyOpcode::CLASS_OFFSET);
        let chip = MemcpyLoopChip::new(
            tester.system_port(),
            range_bus,
            memcpy_bus,
            Rv32MemcpyOpcode::CLASS_OFFSET,
            tester.pointer_max_bits(),
            range_chip.clone(),
        );
        let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

        (harness, (range_chip.air, range_chip))
    }

    fn set_and_execute_memcpy(
        tester: &mut VmChipTestBuilder<F>,
        harness: &mut Harness,
        rng: &mut StdRng,
        shift: u32,
        source_data: &[u8],
        dest_offset: u32,
        source_offset: u32,
        len: u32,
    ) {
        // Write source data to memory
        for (i, &byte) in source_data.iter().enumerate() {
            tester.write(2, source_offset + i as u32, [F::from_canonical_u8(byte)]);
        }

        // Create instruction for memcpy_loop
        let instruction = openvm_instructions::instruction::Instruction {
            opcode: openvm_instructions::VmOpcode::from_usize(
                Rv32MemcpyOpcode::MEMCPY_LOOP.global_opcode().as_usize(),
            ),
            a: F::ZERO,
            b: F::ZERO,
            c: F::from_canonical_u32(shift),
            d: F::ZERO,
            e: F::ZERO,
            f: F::ZERO,
            g: F::ZERO,
        };

        tester.execute(harness, &instruction);

        // Verify the copy operation
        for i in 0..len.min(source_data.len() as u32) {
            let expected = source_data[i as usize];
            let actual = tester.read(2, dest_offset + i)[0].as_canonical_u8();
            assert_eq!(expected, actual, "Mismatch at offset {}", i);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // POSITIVE TESTS
    //
    // Randomly generate memcpy operations and execute, ensuring that the generated trace
    // passes all constraints.
    //////////////////////////////////////////////////////////////////////////////////////

    #[test_case(0, 100)]
    #[test_case(1, 100)]
    #[test_case(2, 100)]
    #[test_case(3, 100)]
    fn rand_memcpy_loop_test(shift: u32, num_ops: usize) {
        let mut rng = create_seeded_rng();

        let mut tester = VmChipTestBuilder::default();
        let (mut harness, range_checker) = create_test_chip(&tester);

        for _ in 0..num_ops {
            let source_data: Vec<u8> = (0..16).map(|_| rng.gen_range(0..=u8::MAX)).collect();
            let source_offset = rng.gen_range(0..1000);
            let dest_offset = rng.gen_range(2000..3000);
            let len = rng.gen_range(1..=16);

            set_and_execute_memcpy(
                &mut tester,
                &mut harness,
                &mut rng,
                shift,
                &source_data,
                dest_offset,
                source_offset,
                len,
            );
        }

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(range_checker)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }

    #[test_case(0, 100)]
    #[test_case(1, 100)]
    #[test_case(2, 100)]
    #[test_case(3, 100)]
    fn rand_memcpy_loop_test_persistent(shift: u32, num_ops: usize) {
        let mut rng = create_seeded_rng();

        let mut tester = VmChipTestBuilder::default_persistent();
        let (mut harness, range_checker) = create_test_chip(&tester);

        for _ in 0..num_ops {
            let source_data: Vec<u8> = (0..16).map(|_| rng.gen_range(0..=u8::MAX)).collect();
            let dest_offset = rng.gen_range(0..1000);
            let source_offset = rng.gen_range(0..1000);
            let len = rng.gen_range(1..=16);

            set_and_execute_memcpy(
                &mut tester,
                &mut harness,
                &mut rng,
                shift,
                &source_data,
                dest_offset,
                source_offset,
                len,
            );
        }

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(range_checker)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // NEGATIVE TESTS
    //
    // Given a fake trace of a single operation, setup a chip and run the test. We replace
    // part of the trace and check that the chip throws the expected error.
    //////////////////////////////////////////////////////////////////////////////////////

    // #[allow(clippy::too_many_arguments)]
    // fn run_negative_memcpy_test(
    //     shift: u32,
    //     prank_shift: u32,
    //     source_data: &[u8],
    //     dest_offset: u32,
    //     source_offset: u32,
    //     len: u32,
    //     prank_dest: Option<u32>,
    //     prank_source: Option<u32>,
    //     prank_len: Option<u32>,
    //     interaction_error: bool,
    // ) {
    //     let mut rng = create_seeded_rng();
    //     let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    //     let (mut chip, range_checker) = create_test_chip(&tester);

    //     set_and_execute_memcpy(
    //         &mut tester,
    //         &mut chip,
    //         &mut rng,
    //         shift,
    //         source_data,
    //         dest_offset,
    //         source_offset,
    //         len,
    //     );

    //     let adapter_width = BaseAir::<F>::width(&chip.air);
    //     let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
    //         let mut values = trace.row_slice(0).to_vec();
    //         let cols: &mut MemcpyIterCols<F> = values.split_at_mut(adapter_width).1.borrow_mut();
    //         cols.shift = [F::from_canonical_u32(prank_shift), F::ZERO];
    //         if let Some(prank_dest) = prank_dest {
    //             cols.dest = F::from_canonical_u32(prank_dest);
    //         }
    //         if let Some(prank_source) = prank_source {
    //             cols.source = F::from_canonical_u32(prank_source);
    //         }
    //         if let Some(prank_len) = prank_len {
    //             cols.len = [F::from_canonical_u32(prank_len), F::ZERO];
    //         }
    //         *trace = RowMajorMatrix::new(values, trace.width());
    //     };

    //     disable_debug_builder();
    //     let tester = tester
    //         .build()
    //         .load_and_prank_trace(chip, modify_trace)
    //         .load_periphery(range_checker)
    //         .finalize();

    //     if interaction_error {
    //         tester
    //             .simple_test()
    //             .expect_err("Expected verification to fail");
    //     } else {
    //         tester
    //             .simple_test()
    //             .expect_err("Expected verification to fail");
    //     }
    // }

    // #[test]
    // fn memcpy_wrong_shift_negative_test() {
    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    //     run_negative_memcpy_test(
    //         0, // original shift
    //         1, // prank shift
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         8,   // len
    //         None,
    //         None,
    //         None,
    //         true,
    //     );
    // }

    // #[test]
    // fn memcpy_wrong_dest_negative_test() {
    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    //     run_negative_memcpy_test(
    //         0, // shift
    //         0, // prank shift (same)
    //         &source_data,
    //         100,       // dest_offset
    //         200,       // source_offset
    //         8,         // len
    //         Some(150), // prank dest
    //         None,
    //         None,
    //         true,
    //     );
    // }

    // #[test]
    // fn memcpy_wrong_source_negative_test() {
    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    //     run_negative_memcpy_test(
    //         0, // shift
    //         0, // prank shift (same)
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         8,   // len
    //         None,
    //         Some(250), // prank source
    //         None,
    //         true,
    //     );
    // }

    // #[test]
    // fn memcpy_wrong_len_negative_test() {
    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    //     run_negative_memcpy_test(
    //         0, // shift
    //         0, // prank shift (same)
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         8,   // len
    //         None,
    //         None,
    //         Some(12), // prank len
    //         true,
    //     );
    // }

    // //////////////////////////////////////////////////////////////////////////////////////
    // // SANITY TESTS
    // //
    // // Ensure that memcpy operations produce the correct results.
    // //////////////////////////////////////////////////////////////////////////////////////

    // #[test]
    // fn memcpy_shift_0_sanity_test() {
    //     let mut rng = create_seeded_rng();
    //     let mut tester = VmChipTestBuilder::default();
    //     let (mut harness, range_checker) = create_test_chip(&tester);

    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    //     set_and_execute_memcpy(
    //         &mut tester,
    //         &mut harness,
    //         &mut rng,
    //         0, // shift
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         8,   // len
    //     );

    //     // Verify the copy operation
    //     for i in 0..8 {
    //         let expected = source_data[i];
    //         let actual = tester.read(2, 100 + i)[0].as_canonical_u8();
    //         assert_eq!(expected, actual, "Mismatch at offset {}", i);
    //     }

    //     let tester = tester
    //         .build()
    //         .load(harness)
    //         .load_periphery(range_checker)
    //         .finalize();
    //     tester.simple_test().expect("Verification failed");
    // }

    // #[test]
    // fn memcpy_shift_1_sanity_test() {
    //     let mut rng = create_seeded_rng();
    //     let mut tester = VmChipTestBuilder::default();
    //     let (mut harness, range_checker) = create_test_chip(&tester);

    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    //     set_and_execute_memcpy(
    //         &mut tester,
    //         &mut harness,
    //         &mut rng,
    //         1, // shift
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         8,   // len
    //     );

    //     // Verify the copy operation with shift=1
    //     for i in 0..8 {
    //         let expected = source_data[i];
    //         let actual = tester.read(2, 100 + i)[0].as_canonical_u8();
    //         assert_eq!(expected, actual, "Mismatch at offset {}", i);
    //     }

    //     let tester = tester
    //         .build()
    //         .load(harness)
    //         .load_periphery(range_checker)
    //         .finalize();
    //     tester.simple_test().expect("Verification failed");
    // }

    // #[test]
    // fn memcpy_shift_2_sanity_test() {
    //     let mut rng = create_seeded_rng();
    //     let mut tester = VmChipTestBuilder::default();
    //     let (mut harness, range_checker) = create_test_chip(&tester);

    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    //     set_and_execute_memcpy(
    //         &mut tester,
    //         &mut harness,
    //         &mut rng,
    //         2, // shift
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         8,   // len
    //     );

    //     // Verify the copy operation with shift=2
    //     for i in 0..8 {
    //         let expected = source_data[i];
    //         let actual = tester.read(2, 100 + i)[0].as_canonical_u8();
    //         assert_eq!(expected, actual, "Mismatch at offset {}", i);
    //     }

    //     let tester = tester
    //         .build()
    //         .load(harness)
    //         .load_periphery(range_checker)
    //         .finalize();
    //     tester.simple_test().expect("Verification failed");
    // }

    // #[test]
    // fn memcpy_shift_3_sanity_test() {
    //     let mut rng = create_seeded_rng();
    //     let mut tester = VmChipTestBuilder::default();
    //     let (mut harness, range_checker) = create_test_chip(&tester);

    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    //     set_and_execute_memcpy(
    //         &mut tester,
    //         &mut harness,
    //         &mut rng,
    //         3, // shift
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         8,   // len
    //     );

    //     // Verify the copy operation with shift=3
    //     for i in 0..8 {
    //         let expected = source_data[i];
    //         let actual = tester.read(2, 100 + i)[0].as_canonical_u8();
    //         assert_eq!(expected, actual, "Mismatch at offset {}", i);
    //     }

    //     let tester = tester
    //         .build()
    //         .load(harness)
    //         .load_periphery(range_checker)
    //         .finalize();
    //     tester.simple_test().expect("Verification failed");
    // }

    // //////////////////////////////////////////////////////////////////////////////////////
    // // EDGE CASE TESTS
    // //
    // // Test edge cases and boundary conditions.
    // //////////////////////////////////////////////////////////////////////////////////////

    // #[test]
    // fn memcpy_zero_length_test() {
    //     let mut rng = create_seeded_rng();
    //     let mut tester = VmChipTestBuilder::default();
    //     let (mut harness, range_checker) = create_test_chip(&tester);

    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    //     set_and_execute_memcpy(
    //         &mut tester,
    //         &mut harness,
    //         &mut rng,
    //         0, // shift
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         0,   // zero length
    //     );

    //     let tester = tester
    //         .build()
    //         .load(harness)
    //         .load_periphery(range_checker)
    //         .finalize();
    //     tester.simple_test().expect("Verification failed");
    // }

    // #[test]
    // fn memcpy_max_length_test() {
    //     let mut rng = create_seeded_rng();
    //     let mut tester = VmChipTestBuilder::default();
    //     let (mut harness, range_checker) = create_test_chip(&tester);

    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    //     set_and_execute_memcpy(
    //         &mut tester,
    //         &mut harness,
    //         &mut rng,
    //         0, // shift
    //         &source_data,
    //         100, // dest_offset
    //         200, // source_offset
    //         16,  // max length
    //     );

    //     // Verify the copy operation
    //     for i in 0..16 {
    //         let expected = source_data[i];
    //         let actual = tester.read(2, 100 + i)[0].as_canonical_u8();
    //         assert_eq!(expected, actual, "Mismatch at offset {}", i);
    //     }

    //     let tester = tester
    //         .build()
    //         .load(harness)
    //         .load_periphery(range_checker)
    //         .finalize();
    //     tester.simple_test().expect("Verification failed");
    // }

    // #[test]
    // fn memcpy_overlapping_regions_test() {
    //     let mut rng = create_seeded_rng();
    //     let mut tester = VmChipTestBuilder::default();
    //     let (mut harness, range_checker) = create_test_chip(&tester);

    //     let source_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    //     // Write initial data to destination
    //     for (i, &byte) in source_data.iter().enumerate() {
    //         tester.write(2, 100 + i as u32, [F::from_canonical_u8(byte)]);
    //     }

    //     set_and_execute_memcpy(
    //         &mut tester,
    //         &mut harness,
    //         &mut rng,
    //         0, // shift
    //         &source_data,
    //         102, // dest_offset (overlapping with source)
    //         100, // source_offset
    //         8,   // len
    //     );

    //     let tester = tester
    //         .build()
    //         .load(harness)
    //         .load_periphery(range_checker)
    //         .finalize();
    //     tester.simple_test().expect("Verification failed");
    // }
}
