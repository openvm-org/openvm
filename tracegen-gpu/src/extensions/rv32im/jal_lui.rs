use std::{mem::size_of, sync::Arc};

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32RdWriteAdapterRecord, RV32_CELL_BITS},
    Rv32JalLuiAir, Rv32JalLuiStepRecord,
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

use super::cuda::jal_lui::tracegen;
use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};

pub struct Rv32JalLuiChipGpu<'a> {
    pub air: Rv32JalLuiAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<'a> Rv32JalLuiChipGpu<'a> {
    pub fn new(
        air: Rv32JalLuiAir,
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

impl ChipUsageGetter for Rv32JalLuiChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(Rv32RdWriteAdapterRecord, Rv32JalLuiStepRecord)>();
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

impl DeviceChip<SC, GpuBackend> for Rv32JalLuiChipGpu<'_> {
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
        testing::BITWISE_OP_LOOKUP_BUS, DenseRecordArena, EmptyAdapterCoreLayout,
        MatrixRecordArena, NewVmChipWrapper, VmAirWrapper,
    };
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{
            Rv32CondRdWriteAdapterAir, Rv32CondRdWriteAdapterStep, Rv32RdWriteAdapterAir,
            Rv32RdWriteAdapterRecord, Rv32RdWriteAdapterStep, RV32_CELL_BITS,
        },
        Rv32JalLuiAir, Rv32JalLuiCoreAir, Rv32JalLuiStep, Rv32JalLuiStepRecord,
        Rv32JalLuiStepWithAdapter,
    };
    use openvm_rv32im_transpiler::Rv32JalLuiOpcode;

    use openvm_stark_backend::verifier::VerificationError;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;
    use stark_backend_gpu::prelude::F;

    use crate::testing::GpuChipTestBuilder;

    const IMM_BITS: usize = 12;
    const MAX_INS_CAPACITY: usize = 128;

    type DenseChip<F> =
        NewVmChipWrapper<F, Rv32JalLuiAir, Rv32JalLuiStepWithAdapter, DenseRecordArena>;
    type SparseChip<F> =
        NewVmChipWrapper<F, Rv32JalLuiAir, Rv32JalLuiStepWithAdapter, MatrixRecordArena<F>>;

    #[test]
    fn rand_jal_tracegen_test() {
        test_jal_lui(Rv32JalLuiOpcode::JAL); // JAL
    }

    #[test]
    fn rand_lui_tracegen_test() {
        test_jal_lui(Rv32JalLuiOpcode::LUI); // LUI
    }

    fn test_jal_lui(opcode: Rv32JalLuiOpcode) {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(bitwise_bus);
        let mut rng = create_seeded_rng();

        let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let mut dense_chip = create_dense_chip(&tester, cpu_bitwise_chip.clone());

        let mut gpu_chip = Rv32JalLuiChipGpu::new(
            dense_chip.air.clone(),
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            None,
        );
        let mut cpu_chip = create_sparse_chip(&tester, cpu_bitwise_chip.clone());

        for _ in 0..100 {
            let imm: i32 = rng.gen_range(0..(1 << IMM_BITS));
            let imm = match opcode {
                Rv32JalLuiOpcode::JAL => ((imm >> 1) << 2) - (1 << IMM_BITS), // JAL
                Rv32JalLuiOpcode::LUI => imm,                                 // LUI
                _ => unreachable!(),
            };

            let a = rng.gen_range((opcode == Rv32JalLuiOpcode::LUI) as usize..32) << 2; // LUI starts from 1
            let needs_write = a != 0 || opcode == Rv32JalLuiOpcode::LUI; // LUI always writes

            tester.execute_with_pc(
                &mut dense_chip,
                &Instruction::large_from_isize(
                    opcode.global_opcode(),
                    a as isize,
                    0,
                    imm as isize,
                    1,
                    0,
                    needs_write as isize,
                    0,
                ),
                rng.gen_range(imm.unsigned_abs()..(1 << PC_BITS)),
            );
        }

        // TODO[stephen]: The GPU chip should probably own the arena and lend it to
        // the executor, but because currently in OpenVM the executor owns it we need
        // to do something like this. This should be updated once the Chip definition
        // is updated.
        type Record<'a> = (
            &'a mut Rv32RdWriteAdapterRecord,
            &'a mut Rv32JalLuiStepRecord,
        );
        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut cpu_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32CondRdWriteAdapterStep>::new(),
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

    fn create_dense_chip(
        tester: &GpuChipTestBuilder,
        bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> DenseChip<F> {
        let mut chip = DenseChip::<F>::new(
            VmAirWrapper::new(
                Rv32CondRdWriteAdapterAir::new(Rv32RdWriteAdapterAir::new(
                    tester.memory_bridge(),
                    tester.execution_bridge(),
                )),
                Rv32JalLuiCoreAir::new(bitwise.bus()),
            ),
            Rv32JalLuiStep::new(
                Rv32CondRdWriteAdapterStep::new(Rv32RdWriteAdapterStep::new()),
                bitwise,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_sparse_chip(
        tester: &GpuChipTestBuilder,
        bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> SparseChip<F> {
        let mut chip = SparseChip::<F>::new(
            VmAirWrapper::new(
                Rv32CondRdWriteAdapterAir::new(Rv32RdWriteAdapterAir::new(
                    tester.memory_bridge(),
                    tester.execution_bridge(),
                )),
                Rv32JalLuiCoreAir::new(bitwise.bus()),
            ),
            Rv32JalLuiStep::new(
                Rv32CondRdWriteAdapterStep::new(Rv32RdWriteAdapterStep::new()),
                bitwise,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }
}
