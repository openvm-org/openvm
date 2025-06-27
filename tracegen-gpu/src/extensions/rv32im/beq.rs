use std::{mem::size_of, sync::Arc};

use super::cuda::beq::tracegen;
use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32BranchAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
    BranchEqualCoreRecord, Rv32BranchEqualAir,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::CudaError},
    prelude::F,
    prover_backend::GpuBackend,
    types::SC,
};

pub struct Rv32BranchEqualChipGpu<'a> {
    pub air: Rv32BranchEqualAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<'a> Rv32BranchEqualChipGpu<'a> {
    pub fn new(
        air: Rv32BranchEqualAir,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
        arena: Option<&'a DenseRecordArena>,
    ) -> Self {
        Self {
            air,
            range_checker,
            bitwise_lookup,
            arena,
        }
    }
}

impl ChipUsageGetter for Rv32BranchEqualChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BranchAdapterRecord,
            BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records_len = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Rv32BranchEqualChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();
        }
        trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS},
        DenseRecordArena, EmptyAdapterCoreLayout, MatrixRecordArena, NewVmChipWrapper,
        VmAirWrapper,
    };
    use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
    use openvm_instructions::{
        instruction::Instruction,
        program::{DEFAULT_PC_STEP, PC_BITS},
        LocalOpcode,
    };
    use openvm_rv32im_transpiler::BranchEqualOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use std::array;

    use openvm_rv32im_circuit::{
        adapters::{
            Rv32BranchAdapterAir, Rv32BranchAdapterRecord, Rv32BranchAdapterStep,
            RV32_REGISTER_NUM_LIMBS,
        },
        BranchEqualCoreAir, BranchEqualCoreRecord, BranchEqualStep, Rv32BranchEqualAir,
        Rv32BranchEqualStep,
    };

    use crate::testing::GpuChipTestBuilder;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;
    use stark_backend_gpu::prelude::F;
    use test_case::test_case;

    const IMM_BITS: usize = 12;
    const MAX_INS_CAPACITY: usize = 128;

    type DenseChip<F> =
        NewVmChipWrapper<F, Rv32BranchEqualAir, Rv32BranchEqualStep, DenseRecordArena>;
    type SparseChip<F> =
        NewVmChipWrapper<F, Rv32BranchEqualAir, Rv32BranchEqualStep, MatrixRecordArena<F>>;

    #[test_case(BranchEqualOpcode::BEQ)]
    #[test_case(BranchEqualOpcode::BNE)]
    fn test_beq_opcode(opcode: BranchEqualOpcode) {
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));
        let mut rng = create_seeded_rng();

        let mut dense_chip = create_dense_chip(&tester);

        let mut gpu_chip = Rv32BranchEqualChipGpu::new(
            dense_chip.air.clone(),
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            None,
        );
        let mut cpu_chip = create_sparse_chip(&tester);

        const ABS_MAX_IMM: i32 = 1 << (IMM_BITS - 1);

        for _ in 0..100 {
            // Generate random register values following OpenVM pattern exactly
            let a: [u8; RV32_REGISTER_NUM_LIMBS] = array::from_fn(|_| rng.gen_range(0..=u8::MAX));
            let b: [u8; RV32_REGISTER_NUM_LIMBS] = if rng.gen_bool(0.5) {
                a
            } else {
                array::from_fn(|_| rng.gen_range(0..=u8::MAX))
            };

            let imm = rng.gen_range((-ABS_MAX_IMM)..ABS_MAX_IMM);
            let rs1 = gen_pointer(&mut rng, 4);
            let rs2 = gen_pointer(&mut rng, 4);

            tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, a.map(F::from_canonical_u8));
            tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, b.map(F::from_canonical_u8));

            let initial_pc = rng.gen_range(imm.unsigned_abs()..(1 << (PC_BITS - 1)));
            tester.execute_with_pc(
                &mut dense_chip,
                &Instruction::from_isize(
                    opcode.global_opcode(),
                    rs1 as isize,
                    rs2 as isize,
                    imm as isize,
                    1,
                    1,
                ),
                initial_pc,
            );
        }

        // Transfer records from dense to sparse chip
        type Record<'a> = (
            &'a mut Rv32BranchAdapterRecord,
            &'a mut BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        );
        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut cpu_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32BranchAdapterStep>::new(),
            );
        gpu_chip.arena = Some(&dense_chip.arena);

        // TODO[stephen]: Because memory is not implemented yet, this test fails
        // interaction constraints. Once memory is completed, we should make sure
        // that verification passes.
        tester
            .build()
            .load_and_compare(gpu_chip, cpu_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }

    fn create_dense_chip(tester: &GpuChipTestBuilder) -> DenseChip<F> {
        let mut chip = DenseChip::<F>::new(
            VmAirWrapper::new(
                Rv32BranchAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
            ),
            BranchEqualStep::new(
                Rv32BranchAdapterStep::new(),
                BranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_sparse_chip(tester: &GpuChipTestBuilder) -> SparseChip<F> {
        let mut chip = SparseChip::<F>::new(
            VmAirWrapper::new(
                Rv32BranchAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
            ),
            BranchEqualStep::new(
                Rv32BranchAdapterStep::new(),
                BranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }
}
