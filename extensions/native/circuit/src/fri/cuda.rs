use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::cuda::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::{FriReducedOpeningRecordMut, OVERALL_WIDTH};
use crate::cuda_abi::fri_cuda;

#[derive(new)]
pub struct FriReducedOpeningChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for FriReducedOpeningChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        // TODO[arayi]: Temporary hack to get mut access to `records`, should have `self` or `&mut
        // self` as a parameter **SAFETY**: `records` should be non-empty at this point
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
        let trace_width = OVERALL_WIDTH;
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            fri_cuda::tracegen(
                trace.buffer(),
                trace_height as u32,
                &d_records,
                record_info.len() as u32,
                &d_record_info,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

// This is the info needed by each row to do parallel tracegen
#[repr(C)]
#[derive(new)]
pub struct RowInfo {
    pub record_offset: u32,
    pub local_idx: u32,
}

#[cfg(test)]
mod test {
    use std::array;

    use openvm_circuit::arch::testing::memory::gen_pointer;
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_native_circuit::{
        FriReducedOpeningAir, FriReducedOpeningChip, FriReducedOpeningExecutor,
        FriReducedOpeningFiller, FriReducedOpeningRecordMut, EXT_DEG,
    };
    use openvm_native_compiler::{conversion::AS, FriOpcode};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::{
        extensions::native::write_native_array,
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
    };

    const MAX_INS_CAPACITY: usize = 1024;

    fn create_test_harness(
        tester: &GpuChipTestBuilder,
    ) -> GpuTestChipHarness<
        F,
        FriReducedOpeningExecutor,
        FriReducedOpeningAir,
        FriReducedOpeningChipGpu,
        FriReducedOpeningChip<F>,
    > {
        let air = FriReducedOpeningAir::new(tester.execution_bridge(), tester.memory_bridge());
        let executor = FriReducedOpeningExecutor;

        let cpu_chip =
            FriReducedOpeningChip::new(FriReducedOpeningFiller, tester.dummy_memory_helper());
        let gpu_chip =
            FriReducedOpeningChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<
            F,
            FriReducedOpeningExecutor,
            FriReducedOpeningAir,
            FriReducedOpeningChipGpu,
            FriReducedOpeningChip<F>,
        >,
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
                tester.write::<1>(AS::Native as usize, a_ptr + i, [a]);
            }
        }

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
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

    #[test_case(28)]
    fn test_fri_tracegen(num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();

        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng);
        }

        // Transfer records to matrix arena for sparse chip
        harness
            .dense_arena
            .get_record_seeker::<FriReducedOpeningRecordMut<F>, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
