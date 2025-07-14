use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::{
    adapters::NativeLoadStoreAdapterRecord, NativeLoadStoreAir, NativeLoadStoreCoreRecord,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    extensions::native::native_loadstore_cuda, primitives::var_range::VariableRangeCheckerChipGPU,
    DeviceChip,
};

#[derive(new)]
pub struct NativeLoadStoreChipGpu<const NUM_CELLS: usize> {
    pub air: NativeLoadStoreAir<NUM_CELLS>,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: DenseRecordArena,
}

impl<const NUM_CELLS: usize> ChipUsageGetter for NativeLoadStoreChipGpu<NUM_CELLS> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const fn record_size<const N: usize>() -> usize {
            size_of::<(
                NativeLoadStoreAdapterRecord<F, N>,
                NativeLoadStoreCoreRecord<F, N>,
            )>()
        }
        let record_size = record_size::<NUM_CELLS>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl<const NUM_CELLS: usize> DeviceChip<SC, GpuBackend> for NativeLoadStoreChipGpu<NUM_CELLS> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.allocated().to_device().unwrap();
        let height = self.current_trace_height();
        let padded_height = next_power_of_two_or_zero(height);
        let width = self.trace_width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, width);

        unsafe {
            native_loadstore_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                width as u32,
                &d_records,
                height as u32,
                &self.range_checker.count,
                NUM_CELLS as u32,
            )
            .unwrap();
        }

        trace
    }
}

#[cfg(test)]
mod test {
    use std::array;

    use openvm_circuit::arch::{EmptyAdapterCoreLayout, MatrixRecordArena, NewVmChipWrapper};
    use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode, VmOpcode};
    use openvm_native_circuit::{
        adapters::{NativeLoadStoreAdapterAir, NativeLoadStoreAdapterStep},
        NativeLoadStoreStep,
    };
    use openvm_native_compiler::{
        conversion::AS, NativeLoadStore4Opcode, NativeLoadStoreOpcode, BLOCK_LOAD_STORE_SIZE,
    };
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const MAX_INS_CAPACITY: usize = 128;
    type DenseChip<F, const NUM_CELLS: usize> = NewVmChipWrapper<
        F,
        NativeLoadStoreAir<NUM_CELLS>,
        NativeLoadStoreStep<NUM_CELLS>,
        DenseRecordArena,
    >;
    type SparseChip<F, const NUM_CELLS: usize> = NewVmChipWrapper<
        F,
        NativeLoadStoreAir<NUM_CELLS>,
        NativeLoadStoreStep<NUM_CELLS>,
        MatrixRecordArena<F>,
    >;

    fn create_test_dense_chip<const NUM_CELLS: usize>(
        tester: &GpuChipTestBuilder,
        offset: usize,
    ) -> DenseChip<F, NUM_CELLS> {
        let mut chip = DenseChip::<F, NUM_CELLS>::new(
            NativeLoadStoreAir::new(
                NativeLoadStoreAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
                openvm_native_circuit::NativeLoadStoreCoreAir::new(offset),
            ),
            NativeLoadStoreStep::new(NativeLoadStoreAdapterStep::new(offset), offset),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip<const NUM_CELLS: usize>(
        tester: &GpuChipTestBuilder,
        offset: usize,
    ) -> SparseChip<F, NUM_CELLS> {
        let mut chip = SparseChip::<F, NUM_CELLS>::new(
            NativeLoadStoreAir::new(
                NativeLoadStoreAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
                openvm_native_circuit::NativeLoadStoreCoreAir::new(offset),
            ),
            NativeLoadStoreStep::new(NativeLoadStoreAdapterStep::new(offset), offset),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn set_and_execute<const NUM_CELLS: usize>(
        tester: &mut GpuChipTestBuilder,
        chip: &mut DenseChip<F, NUM_CELLS>,
        rng: &mut StdRng,
        base_opcode: NativeLoadStoreOpcode,
        global_opcode: VmOpcode,
    ) {
        use openvm_circuit::arch::testing::memory::gen_pointer;

        // Set up pointer value
        let ptr_val = rng.gen_range(0..1000u32);
        let ptr_addr = gen_pointer(rng, 1);
        tester.write::<1>(
            AS::Native as usize,
            ptr_addr,
            [F::from_canonical_u32(ptr_val)],
        );

        // Set up data at source location for LOADW/STOREW
        let data_vals: [F; NUM_CELLS] =
            array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << 30))));
        let imm_offset = rng.gen_range(0..100u32);

        // For STOREW, write data to register a
        let a_addr = if matches!(
            base_opcode,
            NativeLoadStoreOpcode::STOREW | NativeLoadStoreOpcode::HINT_STOREW
        ) {
            let addr = gen_pointer(rng, NUM_CELLS);
            tester.write::<NUM_CELLS>(AS::Native as usize, addr, data_vals);
            addr
        } else {
            // For LOADW, this will be the destination
            gen_pointer(rng, NUM_CELLS)
        };

        // For LOADW, write data at ptr + imm_offset
        if matches!(base_opcode, NativeLoadStoreOpcode::LOADW) {
            tester.write::<NUM_CELLS>(
                AS::Native as usize,
                (ptr_val + imm_offset) as usize,
                data_vals,
            );
        }

        // For HINT_STOREW, push hint data
        if matches!(base_opcode, NativeLoadStoreOpcode::HINT_STOREW) {
            tester.streams.hint_stream.extend(data_vals);
        }

        let initial_pc = rng.gen_range(0..(1 << (PC_BITS - 1)));

        tester.execute_with_pc(
            chip,
            &Instruction::new(
                global_opcode,
                F::from_canonical_usize(a_addr),
                F::from_canonical_u32(imm_offset),
                F::from_canonical_usize(ptr_addr),
                F::from_canonical_u32(AS::Native as u32),
                F::from_canonical_u32(AS::Native as u32),
                F::ZERO,
                F::ZERO,
            ),
            initial_pc,
        );
    }

    fn test_native_loadstore_tracegen<const NUM_CELLS: usize>(
        base_opcode: NativeLoadStoreOpcode,
        global_opcode: VmOpcode,
        num_ops: usize,
        offset: usize,
    ) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();

        let mut dense_chip = create_test_dense_chip::<NUM_CELLS>(&tester, offset);

        for _ in 0..num_ops {
            set_and_execute(
                &mut tester,
                &mut dense_chip,
                &mut rng,
                base_opcode,
                global_opcode,
            );
        }

        let mut sparse_chip = create_test_sparse_chip::<NUM_CELLS>(&tester, offset);

        type Record<'a, const N: usize> = (
            &'a mut NativeLoadStoreAdapterRecord<F, N>,
            &'a mut NativeLoadStoreCoreRecord<F, N>,
        );

        dense_chip
            .arena
            .get_record_seeker::<Record<'_, NUM_CELLS>, _>()
            .transfer_to_matrix_arena(
                &mut sparse_chip.arena,
                EmptyAdapterCoreLayout::<F, NativeLoadStoreAdapterStep<NUM_CELLS>>::new(),
            );

        let gpu_chip = NativeLoadStoreChipGpu::<NUM_CELLS>::new(
            dense_chip.air,
            tester.range_checker(),
            dense_chip.arena,
        );

        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }

    #[test_case(NativeLoadStoreOpcode::LOADW, 100)]
    #[test_case(NativeLoadStoreOpcode::STOREW, 100)]
    #[test_case(NativeLoadStoreOpcode::HINT_STOREW, 100)]
    fn test_native_loadstore_1_tracegen(opcode: NativeLoadStoreOpcode, num_ops: usize) {
        test_native_loadstore_tracegen::<1>(
            opcode,
            opcode.global_opcode(),
            num_ops,
            NativeLoadStoreOpcode::CLASS_OFFSET,
        );
    }

    #[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::LOADW), 50)]
    #[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::STOREW), 50)]
    #[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::HINT_STOREW), 50)]
    fn test_native_loadstore_4_tracegen(opcode: NativeLoadStore4Opcode, num_ops: usize) {
        test_native_loadstore_tracegen::<BLOCK_LOAD_STORE_SIZE>(
            opcode.0,
            opcode.global_opcode(),
            num_ops,
            NativeLoadStore4Opcode::CLASS_OFFSET,
        );
    }
}
