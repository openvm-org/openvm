#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit::arch::testing::default_var_range_checker_bus;
    use openvm_circuit::{
        arch::{
            testing::{
                TestBuilder, TestChipHarness, VmChipTestBuilder, MEMCPY_BUS, RANGE_CHECKER_BUS,
            },
            Arena, PreflightExecutor,
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
        tester: &mut impl TestBuilder<F>,
        executor: &mut E,
        arena: &mut RA,
        shift: u32,
        source_data: &[u8],
        dest_offset: u32, //these are 4 byte aligned tho??
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

    // why are all the tests improved
    // two adicity bug lol
    // also currently for execution, not copying the offsets well

    // ok so NOW
    // we can at least get proof generation going! but it is incorrect LOL!
    //i think at this point, have to fix the execution, so that the trace that is passed in is correct
    //that being said, even if it is correct, (0,1,64) still failing
    // other tests failing too, but for different reasons (probs bc code execution is wrong)

    /*
       1. figure out how to make correct checker for memcpy, with different shifts LOL
           shift means the offsets in memory
           rn checker seems a bit sus? is it properly taking into account shift?
       2. read base C code, to get better understanding


       OK SO: memcpy is only the loop
        precondition: destination is aligned to 4 bytes, src is offset by
            if d%4==1, we copy 3 values at the start, so s%4==3 at the end
                this will correspond with a shift value of 1
    */
    //u8 is 1 byte of memory
    //ayush alexander
    /*
    volatilite vs persistent memroy
     volatile: prove single program without segmentatiob
     persistent prove single program with segmentation, so memory has to be a continuation of previous segmentation
         memory is hashed with merkle tree
         persistent is chunks of 8 field elements, each field element is 4 bytes
    */
    #[test_case(0, 1, 64)]
    #[test_case(1, 100, 64)]
    #[test_case(2, 100, 40)]
    #[test_case(3, 100, 40)]
    fn memcpy_loop_test(shift: u32, num_ops: usize, len: u32) {
        let mut tester = VmChipTestBuilder::default();
        let (mut harness, range_checker, memcpy_loop) = create_harness(&tester);
        //returns airs and chips for range_checler and memcpy_loop
    }

    #[test_case(0, 1, 64)] //shift if 0, we copy 4 values correctly, just offset of 0?
    #[test_case(1, 100, 64)] //shift if 1, we copy (4-1) values correctly, just offset of 1?
    #[test_case(2, 100, 40)] //shift if 2, copy (4-2) values correctly, offset of 2
    #[test_case(3, 100, 40)]
    fn rand_memcpy_iter_test(shift: u32, num_ops: usize, len: u32) {
        //debug builder, catch in proof gen instead of verification step
        let mut rng = create_seeded_rng();
        setup_tracing_with_log_level(Level::DEBUG);

        let mut tester = VmChipTestBuilder::default();
        let (mut harness, range_checker, memcpy_loop) = create_harness(&tester);

        for tc in 0..num_ops {
            let base = rng.gen_range(0..250) * 4;
            let source_offset = base;
            let dest_offset = rng.gen_range(500..750) * 4; // Ensure word alignment
            let source_data: Vec<u8> = (0..len.div_ceil(4) * 4)
                .map(|_| rng.gen_range(0..=u8::MAX))
                .collect(); //generates the data to be copied

            eprintln!(
                "test case: {}, source_offset: {}, dest_offset: {}, len: {}",
                tc, source_offset, dest_offset, len
            );
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
            let base = rng.gen_range(0..250) * 4;
            let source_offset = base;
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
