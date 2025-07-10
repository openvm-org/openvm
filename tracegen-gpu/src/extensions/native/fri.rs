use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_native_circuit::{FriReducedOpeningAir, FriReducedOpeningRecordMut};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    extensions::native::fri_cuda, primitives::var_range::VariableRangeCheckerChipGPU, DeviceChip,
};

#[derive(new)]
pub struct FriReducedOpeningChipGpu {
    pub air: FriReducedOpeningAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for FriReducedOpeningChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        // TODO[arayi]: This is incorrect and temporary,
        // we probably need to get rid of `current_trace_height` or add a counter to `arena`
        self.arena.allocated().len()
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

// This is the info needed by each row to do parallel tracegen
#[repr(C)]
#[derive(new)]
pub struct RowInfo {
    pub record_offset: u32,
    pub local_idx: u32,
}

impl DeviceChip<SC, GpuBackend> for FriReducedOpeningChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let records = self.arena.allocated();

        // TODO[arayi]: Temporary hack to get mut access to `records`, should have `self` or `&mut self` as a parameter
        // **SAFETY**: `records` should be non-empty at this point
        let records =
            unsafe { std::slice::from_raw_parts_mut(records.as_ptr() as *mut u8, records.len()) };

        let mut record_info = Vec::<RowInfo>::with_capacity(records.len());
        let mut offset = 0;

        while offset < records.len() {
            let prev_offset = offset;
            let record =
                RecordSeeker::<DenseRecordArena, FriReducedOpeningRecordMut<F>, _>::get_record_at(
                    &mut offset,
                    records,
                );
            for idx in 0..record.header.length + 2 {
                record_info.push(RowInfo::new(prev_offset as u32, idx));
            }
        }
        debug_assert!(offset == records.len());

        let d_records = records.to_device().unwrap();
        let d_record_info = record_info.to_device().unwrap();

        let trace_height = next_power_of_two_or_zero(record_info.len());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());

        unsafe {
            fri_cuda::tracegen(
                trace.buffer(),
                trace_height as u32,
                &d_records,
                record_info.len() as u32,
                &d_record_info,
                &self.range_checker.count,
            )
            .unwrap();
        }

        trace
    }
}

#[cfg(test)]
mod test {
    use std::array;

    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS},
        InstructionExecutor, MatrixRecordArena, NewVmChipWrapper,
    };
    use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_native_circuit::{FriReducedOpeningStep, EXT_DEG};
    use openvm_native_compiler::{conversion::AS, FriOpcode};
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};

    use super::*;
    use crate::{extensions::native::write_native_array, testing::GpuChipTestBuilder};

    const MAX_INS_CAPACITY: usize = 1024;
    type DenseChip<F> =
        NewVmChipWrapper<F, FriReducedOpeningAir, FriReducedOpeningStep<F>, DenseRecordArena>;
    type SparseChip<F> =
        NewVmChipWrapper<F, FriReducedOpeningAir, FriReducedOpeningStep<F>, MatrixRecordArena<F>>;

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> DenseChip<F> {
        let mut chip = DenseChip::<F>::new(
            FriReducedOpeningAir::new(tester.execution_bridge(), tester.memory_bridge()),
            FriReducedOpeningStep::new(),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(tester: &mut GpuChipTestBuilder) -> SparseChip<F> {
        let mut chip = SparseChip::<F>::new(
            FriReducedOpeningAir::new(tester.execution_bridge(), tester.memory_bridge()),
            FriReducedOpeningStep::new(),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn set_and_execute<E: InstructionExecutor<F>>(
        tester: &mut GpuChipTestBuilder,
        chip: &mut E,
        rng: &mut StdRng,
    ) {
        let len = rng.gen_range(1..=28);
        let a_ptr = gen_pointer(rng, len);
        let b_ptr = gen_pointer(rng, len);
        let a_ptr_ptr =
            write_native_array::<1>(tester, rng, Some([F::from_canonical_usize(a_ptr)])).1;
        let b_ptr_ptr =
            write_native_array::<1>(tester, rng, Some([F::from_canonical_usize(b_ptr)])).1;

        let len_ptr = write_native_array::<1>(tester, rng, Some([F::from_canonical_usize(len)])).1;
        let (_alpha, alpha_ptr) = write_native_array::<EXT_DEG>(tester, rng, None);
        let out_ptr = gen_pointer(rng, EXT_DEG);
        let is_init = true;
        let is_init_ptr = write_native_array::<1>(tester, rng, Some([F::from_bool(is_init)])).1;

        for i in 0..len {
            let a = rng.gen();
            let b = array::from_fn(|_| rng.gen());
            tester.write::<EXT_DEG>(AS::Native as usize, b_ptr + i * EXT_DEG, b);
            if !is_init {
                tester.streams.hint_space[0].push(a);
            } else {
                tester.write_cell(AS::Native as usize, a_ptr + i, a);
            }
            tester.write(AS::Native as usize, b_ptr + i * EXT_DEG, b);
        }

        tester.execute(
            chip,
            &Instruction::from_usize(
                FriOpcode::FRI_REDUCED_OPENING.global_opcode(),
                [
                    a_ptr_ptr,
                    b_ptr_ptr,
                    len_ptr,
                    alpha_ptr,
                    out_ptr,
                    0, // hint id, will just use 0 for testing
                    is_init_ptr,
                ],
            ),
        );
    }

    #[test]
    fn test_fri_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));

        // CPU execution
        let mut dense_chip = create_test_dense_chip(&tester);
        let num_ops = 28;
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut dense_chip, &mut rng);
        }

        let mut sparse_chip = create_test_sparse_chip(&mut tester);
        dense_chip
            .arena
            .get_record_seeker::<FriReducedOpeningRecordMut<F>, _>()
            .transfer_to_matrix_arena(&mut sparse_chip.arena);

        // GPU tracegen
        let gpu_chip = FriReducedOpeningChipGpu::new(
            sparse_chip.air,
            tester.range_checker(),
            dense_chip.arena,
        );

        // `gpu_chip` does GPU tracegen, `sparse_chip` does CPU tracegen. Must check that they are the same
        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
