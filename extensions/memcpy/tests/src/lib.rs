#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit::arch::testing::default_var_range_checker_bus;
    use openvm_circuit::{
        arch::{
            testing::{
                TestBuilder, TestChipHarness, VmChipTestBuilder, MEMCPY_BUS, RANGE_CHECKER_BUS,
            },
            Arena, Executor, MeteredExecutor, PreflightExecutor,
        },
        system::{memory::SharedMemoryHelper, SystemPort},
    };
    use openvm_circuit_primitives::var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerAir, VariableRangeCheckerBus,
        VariableRangeCheckerChip,
    };
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        LocalOpcode, VmOpcode,
    };
    use openvm_memcpy_circuit::{
        MemcpyBus, MemcpyIterAir, MemcpyIterChip, MemcpyIterExecutor, MemcpyIterFiller,
        MemcpyLoopAir, MemcpyLoopChip, A1_REGISTER_PTR, A2_REGISTER_PTR, A3_REGISTER_PTR,
        A4_REGISTER_PTR,
    };
    use openvm_memcpy_transpiler::Rv32MemcpyOpcode;
    use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
    use openvm_stark_backend::p3_matrix::{dense::DenseMatrix, Matrix};
    use openvm_stark_sdk::config::setup_tracing_with_log_level;
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
    use rand::Rng;
    use test_case::test_case;
    use tracing::Level;

    const MAX_INS_CAPACITY: usize = 128 * 100; // error was here, too small;
    type F = BabyBear;
    type Harness = TestChipHarness<F, MemcpyIterExecutor, MemcpyIterAir, MemcpyIterChip<F>>;

    fn create_harness_fields(
        address_bits: usize,
        system_port: SystemPort,
        range_chip: Arc<VariableRangeCheckerChip>,
        memory_helper: SharedMemoryHelper<F>,
    ) -> (
        MemcpyIterAir,
        MemcpyIterExecutor,
        MemcpyIterChip<F>,
        Arc<MemcpyLoopChip>,
    ) {
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
        let range_bus = default_var_range_checker_bus(); // this wrong? address_bits is too big
        let range_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let (air, executor, chip, loop_chip) = create_harness_fields(
            range_bus.range_max_bits, // this wrong? address_bits is too big
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
        // choose type of executor here
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
                //iterate from [word_start, word_end)
                if i < source_data.len() {
                    word_data[i - word_start] = F::from_canonical_u8(source_data[i]);
                    //store the correct chunk
                } //else rem is 0
            }

            //write the given word from the source data into memory
            tester.write(
                //writes into memory, at the given address space, at the pointer, with the given data
                RV32_MEMORY_AS as usize,
                (source_offset + word_idx as u32 * 4) as usize, // starts at word_idx * 4, with source_offset in memory
                // i think THIS PART HAS TO BE 4 aligned ooohhh
                word_data,
            );
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

        let mut d = dest_offset;
        let mut s = source_offset;
        let mut n = len;

        // Program registers for the custom opcode
        let (dst_reg, src_reg) = if shift == 0 {
            (A3_REGISTER_PTR, A4_REGISTER_PTR)
        } else {
            (A1_REGISTER_PTR, A3_REGISTER_PTR)
        };
        tester.write::<4>(
            RV32_REGISTER_AS as usize,
            dst_reg as usize,
            d.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<4>(
            RV32_REGISTER_AS as usize,
            src_reg as usize,
            s.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<4>(
            RV32_REGISTER_AS as usize,
            A2_REGISTER_PTR as usize,
            n.to_le_bytes().map(F::from_canonical_u8),
        );

        // Execute the MEMCPY_LOOP instruction once
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
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // POSITIVE TESTS
    //
    // Randomly generate memcpy operations and execute, ensuring that the generated trace
    // passes all constraints.
    //////////////////////////////////////////////////////////////////////////////////////

    #[test_case(0, 1, 100)] //shift if 0, we copy 4 values correctly, just offset of 0?
    #[test_case(1, 1, 52)] //1 - 1 - 52
    #[test_case(2, 1, 52)] //shift if 2, copy (4-2) values correctly, offset of 2
    #[test_case(3, 1, 52)]
    fn rand_memcpy_iter_test(shift: u32, num_ops: usize, len: u32) {
        //debug builder, catch in proof gen instead of verification step
        let mut rng = create_seeded_rng();
        setup_tracing_with_log_level(Level::DEBUG);

        let mut tester = VmChipTestBuilder::default();
        let (mut harness, range_checker, memcpy_loop) = create_harness(&tester);

        for tc in 0..num_ops {
            let base = rng.gen_range(4..250) * 4;
            let source_offset = base;
            let dest_offset = rng.gen_range(500..750) * 4; // Ensure word alignment

            let mut source_data: Vec<u8> = (0..len.div_ceil(4) * 4)
                .map(|_| rng.gen_range(0..u8::MAX))
                .collect(); //generates the data to be copied
                            // let source_data: Vec<u8s
                            // let source_data = [
                            //     177, 219, 134, 68, 154, 250, 240, 12, 74, 114, 224, 6, 86, 189, 15, 16, 197, 189,
                            //     115, 54, 46, 98, 253, 38, 124, 233, 200, 251, 107, 66, 67, 214, 4, 97, 65, 68, 9,
                            //     117, 222, 129, 116, 226, 17, 161, 48, 56, 177, 216, 117, 167, 53, 14,
                            // ];
            eprintln!(
                "test case: {}, source_offset: {}, dest_offset: {}, len: {}",
                tc, source_offset, dest_offset, len
            );
            eprintln!("source_data: {:?}", source_data);
            // set and execute memcpy should have the onus of handling the shift
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
        let csv = true;
        let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
            if csv {
                for row_idx in 0..trace.height() {
                    let row_data = trace.row_slice(row_idx);
                    let csv_line = row_data
                        .iter()
                        .map(|val| {
                            let numeric_val: u32 = val.as_canonical_u32();
                            numeric_val.to_string()
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    println!("{}", csv_line);
                }
            } else {
                eprintln!("=== TRACE DEBUG INFO ===");
                eprintln!(
                    "Trace dimensions: {} rows x {} cols",
                    trace.height(),
                    trace.width()
                );

                // Print all rows with aligned formatting
                for row_idx in 0..trace.height() {
                    let row_data = trace.row_slice(row_idx);
                    let formatted_values: Vec<String> = row_data
                        .iter()
                        .map(|val| format!("{:>10}", val.as_canonical_u32()))
                        .collect();
                    eprintln!("Row {:>3}: [{}]", row_idx, formatted_values.join(", "));
                }
                eprintln!("========================");
            }
        };

        let tester = tester
            .build()
            .load_and_prank_trace(harness, modify_trace) // Use this instead of load()
            .load_periphery(range_checker)
            .load_periphery(memcpy_loop)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }

    /*
        cargo test --manifest-path extensions/memcpy/tests/Cargo.toml tests::rand_memcpy_iter_test::_1_100_40_expects -- --nocapture 2>&1
        2013265920 = -1 in the field
        2013265909 = -12 in the field
        failed values are 2013265913, awfully close? this is value of -8 in the field
        cur, prev + 16
        ends up being cur -prev -16 == -8
        cur - prev = 8
        cur = prev + 8
    so we are incrementing source pointer by 8, which isnt enough

    check if it is last row actually, and if this computation is correct
        compile in debug mode with debug symbols

    Current Issue:
        if the length is nt a multiple of 16, in the last iteration, the next source wont be 16 away, because of the remainder %16
        SO: things to check:
            - is this AIR in the correct section of the loop? ie are we checking the correct code segment of memcpy
            - if it is in the correct section, why is it checking mod 16? it should only be checking chunks of 16
                - this might imply that the row checking is incorrect, since we are checking one extra iteration

        1. ensure that the constraints are correct, and make sense. ask shayan how constraints work; ask JPW if AIR constraints are correct; suspicion for why its correct

        2. ensure infomration being filled into columns is correct (based on trace gen)

        */

    #[test_case(0, 1, 100)]
    #[test_case(1, 1, 52)] // 1 1 52
    #[test_case(2, 1, 100)]
    #[test_case(3, 1, 100)]
    fn rand_memcpy_iter_test_persistent(shift: u32, num_ops: usize, len: u32) {
        let mut rng = create_seeded_rng();
        //check diff b/w default and default_persistent
        let mut tester = VmChipTestBuilder::default_persistent();
        let (mut harness, range_checker, memcpy_loop) = create_harness(&tester);

        for _ in 0..num_ops {
            let base = rng.gen_range(4..250) * 4;
            let source_offset = base;
            let dest_offset = rng.gen_range(500..750) * 4; // Ensure word alignment
            let source_data: Vec<u8> = (0..len.div_ceil(4) * 4)
                .map(|_| rng.gen_range(0..u8::MAX))
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
            .load_periphery(memcpy_loop)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }
}
