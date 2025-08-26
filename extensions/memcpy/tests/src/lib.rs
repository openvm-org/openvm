#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit::{
        arch::{
            testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, MEMCPY_BUS, RANGE_CHECKER_BUS},
            Arena, PreflightExecutor,
        },
        system::{memory::SharedMemoryHelper, SystemPort},
    };
    use openvm_circuit_primitives::var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerAir, VariableRangeCheckerBus,
        VariableRangeCheckerChip,
    };
    use openvm_instructions::{instruction::Instruction, riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS}, LocalOpcode, VmOpcode};
    use openvm_memcpy_circuit::{
        MemcpyBus, MemcpyIterAir, MemcpyIterChip, MemcpyIterExecutor, MemcpyIterFiller,
        MemcpyLoopAir, MemcpyLoopChip, A1_REGISTER_PTR, A2_REGISTER_PTR, A3_REGISTER_PTR,
        A4_REGISTER_PTR,
    };
    use openvm_memcpy_transpiler::Rv32MemcpyOpcode;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
    use rand::Rng;
    use test_case::test_case;

    const MAX_INS_CAPACITY: usize = 128;
    type F = BabyBear;
    type Harness = TestChipHarness<F, MemcpyIterExecutor, MemcpyIterAir, MemcpyIterChip<F>>;

    fn create_harness_fields(
        address_bits: usize,
        system_port: SystemPort,
        range_chip: Arc<VariableRangeCheckerChip>,
        memory_helper: SharedMemoryHelper<F>,
    ) -> (MemcpyIterAir, MemcpyIterExecutor, MemcpyIterChip<F>, Arc<MemcpyLoopChip>) {
        let range_bus = range_chip.bus();
        let memcpy_bus = MemcpyBus::new(MEMCPY_BUS);

        let air = MemcpyIterAir::new(
            system_port.memory_bridge,
            range_bus,
            memcpy_bus,
            address_bits,
        );
        let executor = MemcpyIterExecutor::new(Rv32MemcpyOpcode::CLASS_OFFSET);
        let loop_chip = Arc::new(MemcpyLoopChip::new(
            system_port,
            range_bus,
            memcpy_bus,
            Rv32MemcpyOpcode::CLASS_OFFSET,
            address_bits,
            range_chip.clone(),
        ));
        let chip = MemcpyIterChip::new(
            MemcpyIterFiller::new(address_bits, range_chip, loop_chip.clone()),
            memory_helper,
        );
        (air, executor, chip, loop_chip)
    }

    fn create_harness(
        tester: &VmChipTestBuilder<F>,
    ) -> (
        Harness,
        (VariableRangeCheckerAir, SharedVariableRangeCheckerChip),
        (MemcpyLoopAir, Arc<MemcpyLoopChip>),
    ) {
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, tester.address_bits());
        let range_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let (air, executor, chip, loop_chip) = create_harness_fields(
            tester.address_bits(),
            tester.system_port(),
            range_chip.clone(),
            tester.memory_helper(),
        );
        let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

        (
            harness,
            (range_chip.air, range_chip),
            (loop_chip.air, loop_chip),
        )
    }

    fn set_and_execute_memcpy<RA: Arena, E: PreflightExecutor<F, RA>>(
        tester: &mut impl TestBuilder<F>,
        executor: &mut E,
        arena: &mut RA,
        shift: u32,
        source_data: &[u8],
        dest_offset: u32,
        source_offset: u32,
        len: u32,
    ) {
        // Write source data to memory by words (4 bytes)
        let source_words = source_data.len().div_ceil(4);
        for word_idx in 0..source_words {
            let word_start = word_idx * 4;
            let word_end = (word_idx + 1) * 4;
            let mut word_data = [F::ZERO; 4];

            for i in word_start..word_end {
                if i < source_data.len() {
                    word_data[i - word_start] = F::from_canonical_u8(source_data[i]);
                }
            }

            tester.write(RV32_MEMORY_AS as usize, (source_offset + word_idx as u32 * 4) as usize, word_data);
        }

        // Set up registers that the memcpy instruction will read from
        // destination address
        tester.write::<4>(
            RV32_REGISTER_AS as usize,
            if shift == 0 {
                A3_REGISTER_PTR
            } else {
                A1_REGISTER_PTR
            },
            dest_offset.to_le_bytes().map(F::from_canonical_u8),
        );
        // length
        tester.write::<4>(
            RV32_REGISTER_AS as usize,
            A2_REGISTER_PTR,
            len.to_le_bytes().map(F::from_canonical_u8),
        );
        // source address
        tester.write::<4>(
            RV32_REGISTER_AS as usize,
            if shift == 0 {
                A4_REGISTER_PTR
            } else {
                A3_REGISTER_PTR
            },
            source_offset.to_le_bytes().map(F::from_canonical_u8),
        );

        // Create instruction for memcpy_iter (uses same opcode as memcpy_loop)
        let instruction = Instruction {
            opcode: VmOpcode::from_usize(Rv32MemcpyOpcode::MEMCPY_LOOP.global_opcode().as_usize()),
            a: F::ZERO,
            b: F::ZERO,
            c: F::from_canonical_u32(shift),
            d: F::ZERO,
            e: F::ZERO,
            f: F::ZERO,
            g: F::ZERO,
        };

        tester.execute(executor, arena, &instruction);

        // Verify the copy operation by reading words
        // let dest_words = (len as usize + 3) / 4; // Round up to nearest word
        // for word_idx in 0..dest_words {
        //     let word_data = tester.read::<4>(2, (dest_offset + word_idx as u32 * 4) as usize);
        //     let word_start = word_idx * 4;

        //     for i in 0..4 {
        //         let byte_idx = word_start + i;
        //         if byte_idx < len as usize && byte_idx < source_data.len() {
        //             let expected = source_data[byte_idx];
        //             let actual = word_data[i].as_canonical_u32() as u8;
        //             assert_eq!(expected, actual, "Mismatch at offset {}", byte_idx);
        //         }
        //     }
        // }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // POSITIVE TESTS
    //
    // Randomly generate memcpy operations and execute, ensuring that the generated trace
    // passes all constraints.
    //////////////////////////////////////////////////////////////////////////////////////

    #[test_case(0, 1, 20)]
    #[test_case(1, 100, 20)]
    #[test_case(2, 100, 20)]
    #[test_case(3, 100, 20)]
    fn rand_memcpy_iter_test(shift: u32, num_ops: usize, len: u32) {
        let mut rng = create_seeded_rng();

        let mut tester = VmChipTestBuilder::default();
        let (mut harness, range_checker, memcpy_loop) = create_harness(&tester);

        for _ in 0..num_ops {
            let source_offset = rng.gen_range(0..250) * 4; // Ensure word alignment
            let dest_offset = rng.gen_range(500..750) * 4; // Ensure word alignment
            let source_data: Vec<u8> = (0..len.div_ceil(4) * 4)
                .map(|_| rng.gen_range(0..=u8::MAX))
                .collect();

            set_and_execute_memcpy(
                &mut tester,
                &mut harness.executor,
                &mut harness.arena, 
                shift,
                &source_data,
                dest_offset,
                source_offset,
                len,
            );
            tracing::info!(
                "source_offset: {}, dest_offset: {}, len: {}",
                source_offset,
                dest_offset,
                len
            );
        }

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(range_checker)
            .load_periphery(memcpy_loop)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }

    #[test_case(0, 100, 20)]
    #[test_case(1, 100, 20)]
    #[test_case(2, 100, 20)]
    #[test_case(3, 100, 20)]
    fn rand_memcpy_iter_test_persistent(shift: u32, num_ops: usize, len: u32) {
        let mut rng = create_seeded_rng();

        let mut tester = VmChipTestBuilder::default_persistent();
        let (mut harness, range_checker, _iter_air) = create_harness(&tester);

        for _ in 0..num_ops {
            let source_offset = rng.gen_range(0..250) * 4; // Ensure word alignment
            let dest_offset = rng.gen_range(500..750) * 4; // Ensure word alignment
            let source_data: Vec<u8> = (0..len.div_ceil(4) * 4)
                .map(|_| rng.gen_range(0..=u8::MAX))
                .collect();

            set_and_execute_memcpy(
                &mut tester,
                &mut harness.executor,
                &mut harness.arena,
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
}