use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::cuda::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::{NativeLoadStoreCoreCols, NativeLoadStoreCoreRecord};
use crate::{
    adapters::{NativeLoadStoreAdapterCols, NativeLoadStoreAdapterRecord},
    cuda_abi::native_loadstore_cuda,
};

#[derive(new)]
pub struct NativeLoadStoreChipGpu<const NUM_CELLS: usize> {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl<const NUM_CELLS: usize> Chip<DenseRecordArena, GpuBackend>
    for NativeLoadStoreChipGpu<NUM_CELLS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const fn record_size<const N: usize>() -> usize {
            size_of::<(
                NativeLoadStoreAdapterRecord<F, N>,
                NativeLoadStoreCoreRecord<F, N>,
            )>()
        }

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        let record_size = record_size::<NUM_CELLS>();
        assert_eq!(records.len() % record_size, 0);

        let height = records.len() / record_size;
        let padded_height = next_power_of_two_or_zero(height);
        let trace_width = NativeLoadStoreAdapterCols::<F, NUM_CELLS>::width()
            + NativeLoadStoreCoreCols::<F, NUM_CELLS>::width();
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        let d_records = records.to_device().unwrap();

        unsafe {
            native_loadstore_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                trace_width as u32,
                &d_records,
                &self.range_checker.count,
                NUM_CELLS as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

#[cfg(test)]
mod test {
    use std::array;

    use crate::adapters::{
        NativeLoadStoreAdapterAir, NativeLoadStoreAdapterExecutor, NativeLoadStoreAdapterFiller,
    };
    use crate::loadstore::{
        NativeLoadStoreAir, NativeLoadStoreChip, NativeLoadStoreCoreAir, NativeLoadStoreCoreFiller,
        NativeLoadStoreExecutor,
    };
    use crate::write_native_array;
    use openvm_circuit::arch::{testing::memory::gen_pointer, EmptyAdapterCoreLayout};
    use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode, VmOpcode};
    use openvm_native_compiler::{
        conversion::AS, NativeLoadStore4Opcode, NativeLoadStoreOpcode, BLOCK_LOAD_STORE_SIZE,
    };
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;

    const MAX_INS_CAPACITY: usize = 128;

    fn create_test_harness<const NUM_CELLS: usize>(
        tester: &GpuChipTestBuilder,
        offset: usize,
    ) -> GpuTestChipHarness<
        F,
        NativeLoadStoreExecutor<NUM_CELLS>,
        NativeLoadStoreAir<NUM_CELLS>,
        NativeLoadStoreChipGpu<NUM_CELLS>,
        NativeLoadStoreChip<F, NUM_CELLS>,
    > {
        let adapter_air =
            NativeLoadStoreAdapterAir::new(tester.memory_bridge(), tester.execution_bridge());
        let core_air = NativeLoadStoreCoreAir::new(offset);
        let air = NativeLoadStoreAir::new(adapter_air, core_air);

        let adapter_step = NativeLoadStoreAdapterExecutor::new(offset);
        let executor = NativeLoadStoreExecutor::new(adapter_step, offset);

        let core_filler = NativeLoadStoreCoreFiller::new(NativeLoadStoreAdapterFiller);
        let cpu_chip = NativeLoadStoreChip::new(core_filler, tester.dummy_memory_helper());

        let gpu_chip =
            NativeLoadStoreChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute<const NUM_CELLS: usize>(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<
            F,
            NativeLoadStoreExecutor<NUM_CELLS>,
            NativeLoadStoreAir<NUM_CELLS>,
            NativeLoadStoreChipGpu<NUM_CELLS>,
            NativeLoadStoreChip<F, NUM_CELLS>,
        >,
        rng: &mut StdRng,
        base_opcode: NativeLoadStoreOpcode,
        global_opcode: VmOpcode,
    ) {
        // Set up pointer value
        let ptr_val = rng.gen_range(0..1000u32);
        let (_, ptr_addr) = write_native_array(tester, rng, Some([F::from_canonical_u32(ptr_val)]));

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
            &mut harness.executor,
            &mut harness.dense_arena,
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
        let mut tester = GpuChipTestBuilder::default();

        let mut harness = create_test_harness::<NUM_CELLS>(&tester, offset);

        for _ in 0..num_ops {
            set_and_execute(
                &mut tester,
                &mut harness,
                &mut rng,
                base_opcode,
                global_opcode,
            );
        }

        type Record<'a, const N: usize> = (
            &'a mut NativeLoadStoreAdapterRecord<F, N>,
            &'a mut NativeLoadStoreCoreRecord<F, N>,
        );

        harness
            .dense_arena
            .get_record_seeker::<Record<'_, NUM_CELLS>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, NativeLoadStoreAdapterExecutor<NUM_CELLS>>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
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

    #[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::LOADW), 100)]
    #[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::STOREW), 100)]
    #[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::HINT_STOREW), 100)]
    fn test_native_loadstore_4_tracegen(opcode: NativeLoadStore4Opcode, num_ops: usize) {
        test_native_loadstore_tracegen::<BLOCK_LOAD_STORE_SIZE>(
            opcode.0,
            opcode.global_opcode(),
            num_ops,
            NativeLoadStore4Opcode::CLASS_OFFSET,
        );
    }
}
