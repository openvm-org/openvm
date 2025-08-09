use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{
        Rv32BaseAluAdapterCols, Rv32BaseAluAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    ShiftCoreCols, ShiftCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use super::cuda::shift_cuda::tracegen as rv32_shift_tracegen;
use crate::{
    get_empty_air_proving_ctx,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(new)]
pub struct Rv32ShiftChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32ShiftChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BaseAluAdapterRecord,
            ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv32BaseAluAdapterCols::<F>::width()
            + ShiftCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);
        unsafe {
            rv32_shift_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod test {
    use openvm_circuit::arch::EmptyAdapterCoreLayout;
    use openvm_circuit_primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChip, var_range::VariableRangeCheckerChip,
    };
    use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{
            Rv32BaseAluAdapterAir, Rv32BaseAluAdapterExecutor, Rv32BaseAluAdapterFiller,
            Rv32BaseAluAdapterRecord,
        },
        Rv32ShiftAir, Rv32ShiftChip, Rv32ShiftExecutor, ShiftCoreAir, ShiftCoreRecord, ShiftFiller,
    };
    use openvm_rv32im_transpiler::ShiftOpcode;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use stark_backend_gpu::prelude::F;
    use test_case::test_case;

    use super::*;
    use crate::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, memory::gen_pointer,
        GpuChipTestBuilder, GpuTestChipHarness,
    };

    const MAX_INS_CAPACITY: usize = 128;

    type Harness =
        GpuTestChipHarness<F, Rv32ShiftExecutor, Rv32ShiftAir, Rv32ShiftChipGpu, Rv32ShiftChip<F>>;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus's from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        let range_bus = default_var_range_checker_bus();

        // creating a dummy chips for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = Rv32ShiftAir::new(
            Rv32BaseAluAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
            ),
            ShiftCoreAir::new(bitwise_bus, range_bus, ShiftOpcode::CLASS_OFFSET),
        );
        let executor =
            Rv32ShiftExecutor::new(Rv32BaseAluAdapterExecutor, ShiftOpcode::CLASS_OFFSET);
        let cpu_chip = Rv32ShiftChip::<F>::new(
            ShiftFiller::new(
                Rv32BaseAluAdapterFiller::new(dummy_bitwise_chip.clone()),
                dummy_bitwise_chip,
                dummy_range_checker,
                ShiftOpcode::CLASS_OFFSET,
            ),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = Rv32ShiftChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            tester.timestamp_max_bits(),
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut Harness,
        rng: &mut StdRng,
        opcode: ShiftOpcode,
    ) {
        let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

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
        tester.execute(&mut harness.executor, &mut harness.dense_arena, &insn);
    }

    #[test_case(ShiftOpcode::SLL, 100)]
    #[test_case(ShiftOpcode::SRL, 100)]
    #[test_case(ShiftOpcode::SRA, 100)]
    fn rand_shift_tracegen_test(opcode: ShiftOpcode, num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        type Record<'a> = (
            &'a mut Rv32BaseAluAdapterRecord,
            &'a mut ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32BaseAluAdapterExecutor<RV32_CELL_BITS>>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
