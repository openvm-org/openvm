use std::{mem::size_of, sync::Arc};

use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32BaseAluAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
    LessThanCoreRecord, Rv32LessThanAir,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use super::cuda::less_than::tracegen as rv32_less_than_tracegen;

pub struct Rv32LessThanChipGpu<'a> {
    pub air: Rv32LessThanAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<'a> Rv32LessThanChipGpu<'a> {
    pub fn new(
        air: Rv32LessThanAir,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    ) -> Self {
        Self {
            air,
            range_checker,
            bitwise_lookup,
            arena: None,
        }
    }
}

impl<'a> ChipUsageGetter for Rv32LessThanChipGpu<'a> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BaseAluAdapterRecord,
            LessThanCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let buf = self.arena.as_ref().unwrap().records_buffer.get_ref();
        let len = self.arena.as_ref().unwrap().records_buffer.position() as usize;
        let records_len = buf[..len].len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl<'a> DeviceChip<SC, GpuBackend> for Rv32LessThanChipGpu<'a> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let buf = self.arena.as_ref().unwrap().records_buffer.get_ref();
        let len = self.arena.as_ref().unwrap().records_buffer.position() as usize;
        let records_slice = &buf[..len];
        let d_records = records_slice.to_device().unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            rv32_less_than_tracegen(
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
mod test {
    use super::*;
    use crate::testing::GpuChipTestBuilder;
    use openvm_circuit::arch::{
        testing::BITWISE_OP_LOOKUP_BUS, EmptyAdapterCoreLayout, MatrixRecordArena,
        NewVmChipWrapper, VmAirWrapper,
    };
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_instructions::{
        instruction::Instruction, program::PC_BITS, riscv::RV32_REGISTER_AS, LocalOpcode,
    };
    use openvm_rv32im_circuit::adapters::{Rv32BaseAluAdapterAir, Rv32BaseAluAdapterStep};
    use openvm_rv32im_circuit::{LessThanCoreAir, LessThanStep};
    use openvm_rv32im_transpiler::LessThanOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;

    type DenseChip<F> = NewVmChipWrapper<
        F,
        Rv32LessThanAir,
        LessThanStep<
            Rv32BaseAluAdapterStep<RV32_CELL_BITS>,
            RV32_REGISTER_NUM_LIMBS,
            RV32_CELL_BITS,
        >,
        DenseRecordArena,
    >;

    const MAX_INS_CAPACITY: usize = 128;

    #[test]
    fn less_than_gpu_chip_test() {
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));
        let mut rng = create_seeded_rng();

        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let shared_bitwise =
            SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus.clone());
        let mut dense_chip = DenseChip::<F>::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_bus.clone(),
                ),
                LessThanCoreAir::new(bitwise_bus.clone(), LessThanOpcode::CLASS_OFFSET),
            ),
            LessThanStep::new(
                Rv32BaseAluAdapterStep::new(shared_bitwise.clone()),
                shared_bitwise.clone(),
                LessThanOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        dense_chip.set_trace_buffer_height(MAX_INS_CAPACITY);

        type CpuLtChip<F> = NewVmChipWrapper<
            F,
            Rv32LessThanAir,
            LessThanStep<
                Rv32BaseAluAdapterStep<RV32_CELL_BITS>,
                RV32_REGISTER_NUM_LIMBS,
                RV32_CELL_BITS,
            >,
            MatrixRecordArena<F>,
        >;
        let mut cpu_chip = CpuLtChip::<F>::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_bus.clone(),
                ),
                LessThanCoreAir::new(bitwise_bus.clone(), LessThanOpcode::CLASS_OFFSET),
            ),
            LessThanStep::new(
                Rv32BaseAluAdapterStep::new(shared_bitwise.clone()),
                shared_bitwise.clone(),
                LessThanOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        cpu_chip.set_trace_buffer_height(MAX_INS_CAPACITY);

        let num_tests = 10;
        let opcodes = [LessThanOpcode::SLT, LessThanOpcode::SLTU];
        for &opcode in &opcodes {
            for _ in 0..num_tests {
                let rs1_ptr = rng.gen_range(0..32) << RV32_CELL_BITS;
                let rs2_ptr = rng.gen_range(0..32) << RV32_CELL_BITS;
                let rd_ptr = rng.gen_range(0..32) << RV32_CELL_BITS;
                let pc = rng.gen_range(0..(1 << PC_BITS));

                let val1 = rng.gen::<u32>();
                let val2 = rng.gen::<u32>();
                let bytes1 = val1.to_le_bytes().map(F::from_canonical_u8);
                let bytes2 = val2.to_le_bytes().map(F::from_canonical_u8);
                tester.write(RV32_REGISTER_AS as usize, rs1_ptr, bytes1);
                tester.write(RV32_REGISTER_AS as usize, rs2_ptr, bytes2);

                let insn = Instruction::from_usize(
                    opcode.global_opcode(),
                    [rd_ptr, rs1_ptr, rs2_ptr, 1, 0],
                );
                tester.execute_with_pc(&mut dense_chip, &insn, pc);
            }
        }

        type Record<'a> = (
            &'a mut Rv32BaseAluAdapterRecord,
            &'a mut LessThanCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        );
        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut cpu_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32BaseAluAdapterStep<RV32_CELL_BITS>>::new(),
            );

        let mut gpu_chip = Rv32LessThanChipGpu::new(
            cpu_chip.air.clone(),
            tester.range_checker(),
            tester.bitwise_op_lookup(),
        );
        gpu_chip.arena = Some(&dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, cpu_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
