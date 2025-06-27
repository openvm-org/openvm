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
    Rv32ShiftAir, ShiftCoreRecord,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use super::cuda::shift::tracegen as rv32_shift_tracegen;

pub struct Rv32ShiftChipGpu<'a> {
    pub air: Rv32ShiftAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<'a> Rv32ShiftChipGpu<'a> {
    pub fn new(
        air: Rv32ShiftAir,
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

impl ChipUsageGetter for Rv32ShiftChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BaseAluAdapterRecord,
            ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
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

impl DeviceChip<SC, GpuBackend> for Rv32ShiftChipGpu<'_> {
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
            rv32_shift_tracegen(
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
    use openvm_rv32im_circuit::{
        adapters::{Rv32BaseAluAdapterAir, Rv32BaseAluAdapterRecord, Rv32BaseAluAdapterStep},
        Rv32ShiftAir, Rv32ShiftStep, ShiftCoreAir, ShiftCoreRecord,
    };
    use openvm_rv32im_transpiler::ShiftOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;
    use stark_backend_gpu::prelude::F;

    type DenseChip<F> = NewVmChipWrapper<F, Rv32ShiftAir, Rv32ShiftStep, DenseRecordArena>;
    const MAX_INS_CAPACITY: usize = 128;

    #[test]
    fn rand_shift_tracegen_test() {
        let mut rng = create_seeded_rng();
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(bitwise_bus.clone());

        let shared_bitwise =
            SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus.clone());

        let mut dense_chip = DenseChip::<F>::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_bus.clone(),
                ),
                ShiftCoreAir::new(
                    bitwise_bus.clone(),
                    tester.cpu_range_checker().bus(),
                    ShiftOpcode::CLASS_OFFSET,
                ),
            ),
            Rv32ShiftStep::new(
                Rv32BaseAluAdapterStep::new(shared_bitwise.clone()),
                shared_bitwise.clone(),
                tester.cpu_range_checker().clone(),
                ShiftOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        dense_chip.set_trace_buffer_height(MAX_INS_CAPACITY);

        type CpuShiftChip<F> =
            NewVmChipWrapper<F, Rv32ShiftAir, Rv32ShiftStep, MatrixRecordArena<F>>;
        let mut cpu_chip = CpuShiftChip::<F>::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_bus.clone(),
                ),
                ShiftCoreAir::new(
                    bitwise_bus.clone(),
                    tester.cpu_range_checker().bus(),
                    ShiftOpcode::CLASS_OFFSET,
                ),
            ),
            Rv32ShiftStep::new(
                Rv32BaseAluAdapterStep::new(shared_bitwise.clone()),
                shared_bitwise.clone(),
                tester.cpu_range_checker().clone(),
                ShiftOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        cpu_chip.set_trace_buffer_height(MAX_INS_CAPACITY);

        let num_execution_steps = 10;
        let opcodes = [ShiftOpcode::SLL, ShiftOpcode::SRL, ShiftOpcode::SRA];
        for &opcode in &opcodes {
            for _ in 0..num_execution_steps {
                let rd_ptr = (rng.gen_range(0..32) << RV32_CELL_BITS) as usize;
                let rs1_ptr = (rng.gen_range(0..32) << RV32_CELL_BITS) as usize;
                let pc = rng.gen_range(0..(1 << PC_BITS));

                let val_rs1 = rng.gen::<u32>();
                tester.write(
                    RV32_REGISTER_AS as usize,
                    rs1_ptr,
                    val_rs1.to_le_bytes().map(F::from_canonical_u8),
                );

                let is_imm = rng.gen_bool(0.5);
                let (rs2_field, e_flag) = if is_imm {
                    let imm = rng.gen_range(0..(1 << RV32_CELL_BITS)) as usize;
                    (imm, 0usize)
                } else {
                    let rs2_ptr = (rng.gen_range(0..32) << RV32_CELL_BITS) as usize;
                    let val_rs2 = rng.gen::<u32>();
                    tester.write(
                        RV32_REGISTER_AS as usize,
                        rs2_ptr,
                        val_rs2.to_le_bytes().map(F::from_canonical_u8),
                    );
                    (rs2_ptr, 1usize)
                };

                let insn = Instruction::from_usize(
                    opcode.global_opcode(),
                    [rd_ptr, rs1_ptr, rs2_field, 1, e_flag],
                );
                tester.execute_with_pc(&mut dense_chip, &insn, pc);
            }
        }

        type Record<'a> = (
            &'a mut Rv32BaseAluAdapterRecord,
            &'a mut ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        );
        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut cpu_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32BaseAluAdapterStep<RV32_CELL_BITS>>::new(),
            );

        let mut gpu_chip = Rv32ShiftChipGpu::new(
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
