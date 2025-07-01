use std::{mem::size_of, sync::Arc};

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32MultAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
    MultiplicationCoreRecord, Rv32MultiplicationAir,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use super::cuda::mul::tracegen as mul_tracegen;
use crate::{primitives::var_range::VariableRangeCheckerChipGPU, DeviceChip};

pub struct Rv32MultiplicationChipGpu<'a> {
    pub air: Rv32MultiplicationAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<'a> Rv32MultiplicationChipGpu<'a> {
    pub fn new(
        air: Rv32MultiplicationAir,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        arena: Option<&'a DenseRecordArena>,
    ) -> Self {
        Self {
            air,
            range_checker,
            arena,
        }
    }
}

impl ChipUsageGetter for Rv32MultiplicationChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            Rv32MultAdapterRecord,
            MultiplicationCoreRecord<{ RV32_REGISTER_NUM_LIMBS }, { RV32_CELL_BITS }>,
        )>();
        let buf = &self.arena.unwrap().allocated();
        let total = buf.len();
        assert_eq!(total % RECORD_SIZE, 0);
        total / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Rv32MultiplicationChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let buf = &self.arena.unwrap().allocated();
        let d_records = buf.to_device().unwrap();
        let height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(height, self.trace_width());
        unsafe {
            mul_tracegen(
                trace.buffer(),
                height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
            )
            .unwrap();
        }
        trace
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::{
        testing::memory::gen_pointer, DenseRecordArena, EmptyAdapterCoreLayout, MatrixRecordArena,
        NewVmChipWrapper, VmAirWrapper,
    };
    use openvm_circuit_primitives::range_tuple::{
        RangeTupleCheckerBus, SharedRangeTupleCheckerChip,
    };
    use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{Rv32MultAdapterAir, Rv32MultAdapterStep},
        MultiplicationCoreAir, MultiplicationCoreRecord, Rv32MultiplicationAir,
        Rv32MultiplicationStep,
    };
    use openvm_rv32im_transpiler::MulOpcode;
    use openvm_stark_backend::verifier::VerificationError;
    use openvm_stark_sdk::utils::create_seeded_rng;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const MAX_INS_CAPACITY: usize = 128;

    type DenseMulChip<F> =
        NewVmChipWrapper<F, Rv32MultiplicationAir, Rv32MultiplicationStep, DenseRecordArena>;
    type SparseMulChip<F> =
        NewVmChipWrapper<F, Rv32MultiplicationAir, Rv32MultiplicationStep, MatrixRecordArena<F>>;

    fn create_dense_mul_chip(
        tester: &GpuChipTestBuilder,
        range_tuple: SharedRangeTupleCheckerChip<2>,
    ) -> DenseMulChip<F> {
        let mut chip = DenseMulChip::<F>::new(
            VmAirWrapper::new(
                Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                MultiplicationCoreAir::new(*range_tuple.bus(), MulOpcode::CLASS_OFFSET),
            ),
            Rv32MultiplicationStep::new(
                Rv32MultAdapterStep::new(),
                range_tuple.clone(),
                MulOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_sparse_mul_chip(
        tester: &GpuChipTestBuilder,
        range_tuple: SharedRangeTupleCheckerChip<2>,
    ) -> SparseMulChip<F> {
        let mut chip = SparseMulChip::<F>::new(
            VmAirWrapper::new(
                Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                MultiplicationCoreAir::new(*range_tuple.bus(), MulOpcode::CLASS_OFFSET),
            ),
            Rv32MultiplicationStep::new(
                Rv32MultAdapterStep::new(),
                range_tuple.clone(),
                MulOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[test]
    fn rand_mul_tracegen_test() {
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();
        let mut rng = create_seeded_rng();

        // Default range tuple sizes for MUL (8-bit limbs)
        let sizes = [1 << 8, 8 * (1 << 8)];
        let range_tuple_bus = RangeTupleCheckerBus::new(0, sizes);
        let cpu_range_tuple_chip = SharedRangeTupleCheckerChip::new(range_tuple_bus);

        let mut dense_chip = create_dense_mul_chip(&tester, cpu_range_tuple_chip.clone());
        let mut gpu_chip =
            Rv32MultiplicationChipGpu::new(dense_chip.air, tester.range_checker(), None);
        let mut cpu_chip = create_sparse_mul_chip(&tester, cpu_range_tuple_chip);

        for _ in 0..100 {
            let ptr1 = gen_pointer(&mut rng, 4);
            let ptr2 = gen_pointer(&mut rng, 4);
            tester.execute(
                &mut dense_chip,
                &Instruction::from_usize(
                    MulOpcode::MUL.global_opcode(),
                    [
                        ptr1 as usize,
                        ptr2 as usize,
                        0,
                        RV32_REGISTER_AS as usize,
                        0,
                    ],
                ),
            );
        }

        type Record<'a> = (
            &'a mut Rv32MultAdapterRecord,
            &'a mut MultiplicationCoreRecord<{ RV32_REGISTER_NUM_LIMBS }, { RV32_CELL_BITS }>,
        );
        dense_chip
            .arena
            .get_record_seeker::<Record<'_>, _>()
            .transfer_to_matrix_arena(
                &mut cpu_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32MultAdapterStep>::new(),
            );
        gpu_chip.arena = Some(&dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, cpu_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
