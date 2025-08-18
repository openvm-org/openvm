use std::mem::size_of;

use derive_new::new;
use openvm_circuit::{
    arch::DenseRecordArena,
    system::phantom::{PhantomCols, PhantomRecord},
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend, types::F};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{
    prover::{hal::MatrixDimensions, types::AirProvingContext},
    Chip,
};

#[derive(new)]
pub struct PhantomChipGPU;

impl PhantomChipGPU {
    pub fn trace_height(arena: &DenseRecordArena) -> usize {
        let record_size = size_of::<PhantomRecord>();
        let records_len = arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    pub fn trace_width() -> usize {
        PhantomCols::<F>::width()
    }
}

impl Chip<DenseRecordArena, GpuBackend> for PhantomChipGPU {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let num_records = Self::trace_height(&arena);
        if num_records == 0 {
            return get_empty_air_proving_ctx();
        }
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, Self::trace_width());
        unsafe {
            cuda_abi::phantom::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &arena.allocated().to_device().unwrap(),
            )
            .expect("Failed to generate trace");
        }
        AirProvingContext::simple_no_pis(trace)
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::{
        arch::{
            testing::TestChipHarness, Arena, DenseRecordArena, EmptyMultiRowLayout, ExecutionState,
            MatrixRecordArena, VmChipWrapper,
        },
        system::phantom::{PhantomAir, PhantomExecutor, PhantomFiller, PhantomRecord},
        utils::next_power_of_two_or_zero,
    };
    use openvm_cuda_common::types::F;
    use openvm_instructions::{instruction::Instruction, LocalOpcode, SystemOpcode};
    use p3_air::BaseAir;
    use p3_field::{FieldAlgebra, PrimeField32};

    use crate::{system::phantom::PhantomChipGPU, testing::GpuChipTestBuilder};

    #[test]
    fn test_phantom_tracegen() {
        const NUM_NOPS: usize = 100;
        let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
        let mut tester = GpuChipTestBuilder::default();

        let executor = PhantomExecutor::<F>::new(Default::default(), phantom_opcode);
        let air = PhantomAir {
            execution_bridge: tester.execution_bridge(),
            phantom_opcode,
        };
        let gpu_chip = PhantomChipGPU::new();
        let mut harness = TestChipHarness::<F, _, _, _, DenseRecordArena>::with_capacity(
            executor, air, gpu_chip, NUM_NOPS,
        );

        let nop = Instruction::from_isize(phantom_opcode, 0, 0, 0, 0, 0);
        let mut state: ExecutionState<F> = ExecutionState::new(F::ZERO, F::ONE);
        for _ in 0..NUM_NOPS {
            tester.execute_with_pc(
                &mut harness.executor,
                &mut harness.arena,
                &nop,
                state.pc.as_canonical_u32(),
            );
            let new_state = tester.execution.0.records.last().unwrap().final_state;
            assert_eq!(state.pc + F::from_canonical_usize(4), new_state.pc);
            assert_eq!(state.timestamp + F::ONE, new_state.timestamp);
            state = new_state;
        }

        type Record<'a> = &'a mut PhantomRecord;
        let cpu_chip = VmChipWrapper::new(PhantomFiller, tester.dummy_memory_helper());
        let mut cpu_arena = MatrixRecordArena::<F>::with_capacity(
            next_power_of_two_or_zero(NUM_NOPS),
            <PhantomAir as BaseAir<F>>::width(&harness.air),
        );
        harness
            .arena
            .get_record_seeker::<Record, EmptyMultiRowLayout>()
            .transfer_to_matrix_arena(&mut cpu_arena);

        tester
            .build()
            .load_and_compare(
                harness.air,
                harness.chip,
                harness.arena,
                cpu_chip,
                cpu_arena,
            )
            .simple_test()
            .expect("Verification failed");
    }
}
