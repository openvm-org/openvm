#[cfg(all(test, feature = "cuda"))]
mod tests {
    use std::sync::Arc;

    use openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness, TestBuilder,
        },
        DenseRecordArena,
    };
    use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
    use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
    use openvm_new_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
    use rand::{rngs::StdRng, Rng};

    use crate::{
        cuda::{KeccakfVmChipGpu, XorinVmChipGpu},
        keccakf::{
            air::KeccakfVmAir, trace::KeccakfVmRecordMut, KeccakfVmChip, KeccakfVmExecutor,
            KeccakfVmFiller,
        },
        xorin::{
            air::XorinVmAir, trace::XorinVmRecordMut, XorinVmChip, XorinVmExecutor, XorinVmFiller,
        },
    };

    type F = BabyBear;
    const MAX_INS_CAPACITY: usize = 4096;

    fn create_xorin_cuda_harness(
        tester: &GpuChipTestBuilder,
    ) -> GpuTestChipHarness<F, XorinVmExecutor, XorinVmAir, XorinVmChipGpu, XorinVmChip<F>> {
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            default_bitwise_lookup_bus(),
        ));

        let air = XorinVmAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            default_bitwise_lookup_bus(),
            tester.address_bits(),
            XorinOpcode::CLASS_OFFSET,
        );

        let executor = XorinVmExecutor::new(XorinOpcode::CLASS_OFFSET, tester.address_bits());

        let cpu_chip = XorinVmChip::new(
            XorinVmFiller::new(dummy_bitwise_chip, tester.address_bits()),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = XorinVmChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            tester.address_bits(),
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        executor: &mut XorinVmExecutor,
        arena: &mut DenseRecordArena,
        rng: &mut StdRng,
        len: Option<usize>,
    ) {
        use openvm_circuit::arch::testing::{memory::gen_pointer, TestBuilder};

        let len = len.unwrap_or_else(|| rng.gen_range(4..=136));
        // Length must be a multiple of 4
        let len = (len / 4) * 4;
        if len == 0 {
            return;
        }

        // gen_pointer(rng, 4) generates 4-byte aligned register indices
        let buffer_reg = gen_pointer(rng, 4);
        let input_reg = gen_pointer(rng, 4);
        let len_reg = gen_pointer(rng, 4);

        // gen_pointer(rng, len) generates pointers within bounds for data of size len
        let buffer_ptr = gen_pointer(rng, len);
        let input_ptr = gen_pointer(rng, len);

        // Write buffer pointer to register
        tester.write(
            1,
            buffer_reg,
            buffer_ptr.to_le_bytes().map(F::from_canonical_u8),
        );

        // Write input pointer to register
        tester.write(
            1,
            input_reg,
            input_ptr.to_le_bytes().map(F::from_canonical_u8),
        );

        // Write len VALUE directly to register (not a pointer)
        tester.write(
            1,
            len_reg,
            (len as u32).to_le_bytes().map(F::from_canonical_u8),
        );

        // Write buffer data
        let buffer_data: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
        for (i, chunk) in buffer_data.chunks(4).enumerate() {
            let mut word = [F::ZERO; 4];
            for (j, &byte) in chunk.iter().enumerate() {
                word[j] = F::from_canonical_u8(byte);
            }
            tester.write(2, buffer_ptr as usize + i * 4, word);
        }

        // Write input data
        let input_data: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
        for (i, chunk) in input_data.chunks(4).enumerate() {
            let mut word = [F::ZERO; 4];
            for (j, &byte) in chunk.iter().enumerate() {
                word[j] = F::from_canonical_u8(byte);
            }
            tester.write(2, input_ptr as usize + i * 4, word);
        }

        let instruction = Instruction::from_usize(
            XorinOpcode::XORIN.global_opcode(),
            [buffer_reg, input_reg, len_reg, 1, 2],
        );

        tester.execute(executor, arena, &instruction);
    }

    #[test]
    fn test_xorin_cuda_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let mut harness = create_xorin_cuda_harness(&tester);

        // Test a few xorin operations with random lengths
        let num_ops: usize = 5;
        for _ in 0..num_ops {
            set_and_execute(
                &mut tester,
                &mut harness.executor,
                &mut harness.dense_arena,
                &mut rng,
                None,
            );
        }

        // Test specific length edge cases
        for len in [4, 8, 16, 32, 64, 128, 136] {
            set_and_execute(
                &mut tester,
                &mut harness.executor,
                &mut harness.dense_arena,
                &mut rng,
                Some(len),
            );
        }

        harness
            .dense_arena
            .get_record_seeker::<XorinVmRecordMut, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[test]
    fn test_xorin_cuda_tracegen_single() {
        let mut rng = create_seeded_rng();
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let mut harness = create_xorin_cuda_harness(&tester);

        // Single operation test
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            Some(16),
        );

        harness
            .dense_arena
            .get_record_seeker::<XorinVmRecordMut, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    // ==================== Keccakf Tests ====================

    fn create_keccakf_cuda_harness(
        tester: &GpuChipTestBuilder,
    ) -> GpuTestChipHarness<F, KeccakfVmExecutor, KeccakfVmAir, KeccakfVmChipGpu, KeccakfVmChip<F>>
    {
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            default_bitwise_lookup_bus(),
        ));

        let air = KeccakfVmAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            default_bitwise_lookup_bus(),
            tester.address_bits(),
            KeccakfOpcode::CLASS_OFFSET,
        );

        let executor = KeccakfVmExecutor::new(KeccakfOpcode::CLASS_OFFSET, tester.address_bits());

        let cpu_chip = KeccakfVmChip::new(
            KeccakfVmFiller::new(dummy_bitwise_chip, tester.address_bits()),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = KeccakfVmChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            tester.address_bits(),
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute_keccakf(
        tester: &mut GpuChipTestBuilder,
        executor: &mut KeccakfVmExecutor,
        arena: &mut DenseRecordArena,
        rng: &mut StdRng,
    ) {
        use openvm_circuit::arch::testing::{memory::gen_pointer, TestBuilder};

        // Keccak state is 200 bytes (25 u64s)
        const KECCAK_STATE_BYTES: usize = 200;

        // Generate register address for buffer pointer
        let buffer_reg = gen_pointer(rng, 4);

        // Generate buffer pointer (aligned for 200 bytes)
        let buffer_ptr = gen_pointer(rng, KECCAK_STATE_BYTES);

        // Write buffer pointer to register
        tester.write(
            1,
            buffer_reg,
            buffer_ptr.to_le_bytes().map(F::from_canonical_u8),
        );

        // Write random state data to buffer (200 bytes)
        let state_data: Vec<u8> = (0..KECCAK_STATE_BYTES).map(|_| rng.gen()).collect();
        for (i, chunk) in state_data.chunks(4).enumerate() {
            let mut word = [F::ZERO; 4];
            for (j, &byte) in chunk.iter().enumerate() {
                word[j] = F::from_canonical_u8(byte);
            }
            tester.write(2, buffer_ptr as usize + i * 4, word);
        }

        let instruction = Instruction::from_usize(
            KeccakfOpcode::KECCAKF.global_opcode(),
            [buffer_reg, 0, 0, 1, 2],
        );

        tester.execute(executor, arena, &instruction);
    }

    #[test]
    fn test_keccakf_cuda_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let mut harness = create_keccakf_cuda_harness(&tester);

        // Test multiple keccakf operations
        let num_ops: usize = 3;
        for _ in 0..num_ops {
            set_and_execute_keccakf(
                &mut tester,
                &mut harness.executor,
                &mut harness.dense_arena,
                &mut rng,
            );
        }

        harness
            .dense_arena
            .get_record_seeker::<KeccakfVmRecordMut, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[test]
    fn test_keccakf_cuda_tracegen_single() {
        let mut rng = create_seeded_rng();
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let mut harness = create_keccakf_cuda_harness(&tester);

        // Single operation test
        set_and_execute_keccakf(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
        );

        harness
            .dense_arena
            .get_record_seeker::<KeccakfVmRecordMut, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[test]
    fn test_keccakf_cuda_tracegen_zero_state() {
        let mut rng = create_seeded_rng();
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let mut harness = create_keccakf_cuda_harness(&tester);

        // Test with zero state
        use openvm_circuit::arch::testing::{memory::gen_pointer, TestBuilder};

        const KECCAK_STATE_BYTES: usize = 200;

        let buffer_reg = gen_pointer(&mut rng, 4);
        let buffer_ptr = gen_pointer(&mut rng, KECCAK_STATE_BYTES);

        tester.write(
            1,
            buffer_reg,
            buffer_ptr.to_le_bytes().map(F::from_canonical_u8),
        );

        // Write zero state
        for i in 0..(KECCAK_STATE_BYTES / 4) {
            tester.write(2, buffer_ptr as usize + i * 4, [F::ZERO; 4]);
        }

        let instruction = Instruction::from_usize(
            KeccakfOpcode::KECCAKF.global_opcode(),
            [buffer_reg, 0, 0, 1, 2],
        );

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instruction,
        );

        harness
            .dense_arena
            .get_record_seeker::<KeccakfVmRecordMut, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
