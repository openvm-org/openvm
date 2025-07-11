use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32BaseAluAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
    BaseAluCoreRecord, Rv32BaseAluAir,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use super::cuda::alu::tracegen as rv32_alu_tracegen;
use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};

#[derive(new)]
pub struct Rv32AluChipGpu<'a> {
    pub air: Rv32BaseAluAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl ChipUsageGetter for Rv32AluChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BaseAluAdapterRecord,
            BaseAluCoreRecord<{ RV32_REGISTER_NUM_LIMBS }>,
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

impl DeviceChip<SC, GpuBackend> for Rv32AluChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let buf = &self.arena.unwrap().allocated();
        let d_records = buf.to_device().unwrap();
        let height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(height, self.trace_width());
        unsafe {
            rv32_alu_tracegen(
                trace.buffer(),
                height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
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
    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS},
        DenseRecordArena, EmptyAdapterCoreLayout, MatrixRecordArena, NewVmChipWrapper,
        VmAirWrapper,
    };
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{Rv32BaseAluAdapterAir, Rv32BaseAluAdapterStep},
        BaseAluCoreAir, BaseAluCoreRecord, Rv32BaseAluAir, Rv32BaseAluStep,
    };
    use openvm_rv32im_transpiler::BaseAluOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;
    use stark_backend_gpu::prelude::F;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const MAX_INS_CAPACITY: usize = 512;

    type DenseAluChip<F> = NewVmChipWrapper<F, Rv32BaseAluAir, Rv32BaseAluStep, DenseRecordArena>;
    type SparseAluChip<F> =
        NewVmChipWrapper<F, Rv32BaseAluAir, Rv32BaseAluStep, MatrixRecordArena<F>>;

    fn create_dense_alu_chip(
        tester: &GpuChipTestBuilder,
        bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> DenseAluChip<F> {
        let mut chip = DenseAluChip::<F>::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_chip.bus(),
                ),
                BaseAluCoreAir::new(bitwise_chip.bus(), BaseAluOpcode::CLASS_OFFSET),
            ),
            Rv32BaseAluStep::new(
                Rv32BaseAluAdapterStep::new(bitwise_chip.clone()),
                bitwise_chip.clone(),
                BaseAluOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_sparse_alu_chip(
        tester: &GpuChipTestBuilder,
        bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> SparseAluChip<F> {
        let mut chip = SparseAluChip::<F>::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    tester.execution_bridge(),
                    tester.memory_bridge(),
                    bitwise_chip.bus(),
                ),
                BaseAluCoreAir::new(bitwise_chip.bus(), BaseAluOpcode::CLASS_OFFSET),
            ),
            Rv32BaseAluStep::new(
                Rv32BaseAluAdapterStep::new(bitwise_chip.clone()),
                bitwise_chip.clone(),
                BaseAluOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[test]
    fn rand_alu_tracegen_test() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let shared_bitwise = SharedBitwiseOperationLookupChip::new(bitwise_bus);

        let mut dense_chip = create_dense_alu_chip(&tester, shared_bitwise.clone());
        let mut gpu_chip = Rv32AluChipGpu::new(
            dense_chip.air,
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            None,
        );
        let mut cpu_chip = create_sparse_alu_chip(&tester, shared_bitwise.clone());

        let opcodes = [
            BaseAluOpcode::ADD,
            BaseAluOpcode::SUB,
            BaseAluOpcode::XOR,
            BaseAluOpcode::OR,
            BaseAluOpcode::AND,
        ];
        for opcode in opcodes {
            for _ in 0..100 {
                let rd_ptr = gen_pointer(&mut rng, RV32_REGISTER_NUM_LIMBS);
                let rs1_ptr = gen_pointer(&mut rng, RV32_REGISTER_NUM_LIMBS);

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
                    let rs2_ptr = gen_pointer(&mut rng, RV32_REGISTER_NUM_LIMBS);
                    let val_rs2 = rng.gen::<u32>();
                    tester.write(
                        RV32_REGISTER_AS as usize,
                        rs2_ptr,
                        val_rs2.to_le_bytes().map(F::from_canonical_u8),
                    );
                    (rs2_ptr as usize, 1usize)
                };

                tester.execute(
                    &mut dense_chip,
                    &Instruction::from_usize(
                        opcode.global_opcode(),
                        [
                            rd_ptr as usize,
                            rs1_ptr as usize,
                            rs2_field,
                            RV32_REGISTER_AS as usize,
                            e_flag,
                        ],
                    ),
                );
            }
        }

        type Record<'a> = (
            &'a mut Rv32BaseAluAdapterRecord,
            &'a mut BaseAluCoreRecord<{ RV32_REGISTER_NUM_LIMBS }>,
        );
        dense_chip
            .arena
            .get_record_seeker::<Record<'_>, _>()
            .transfer_to_matrix_arena(
                &mut cpu_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32BaseAluAdapterStep<RV32_CELL_BITS>>::new(),
            );
        gpu_chip.arena = Some(&dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, cpu_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
