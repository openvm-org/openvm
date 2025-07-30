use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{
        Rv32BranchAdapterCols, Rv32BranchAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    BranchLessThanCoreCols, BranchLessThanCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use super::cuda::branch_lt_cuda::tracegen;
use crate::{
    get_empty_air_proving_ctx,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(new)]
pub struct Rv32BranchLessThanChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32BranchLessThanChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BranchAdapterRecord,
            BranchLessThanCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width =
            BranchLessThanCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width()
                + Rv32BranchAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            tracegen(
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
mod tests {
    use std::array;

    use openvm_circuit::arch::{testing::memory::gen_pointer, EmptyAdapterCoreLayout};
    use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
    use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{
            Rv32BranchAdapterAir, Rv32BranchAdapterFiller, Rv32BranchAdapterRecord,
            Rv32BranchAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
        },
        BranchLessThanCoreAir, BranchLessThanCoreRecord, BranchLessThanFiller,
        Rv32BranchLessThanAir, Rv32BranchLessThanChip, Rv32BranchLessThanStep,
    };
    use openvm_rv32im_transpiler::BranchLessThanOpcode;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use stark_backend_gpu::prelude::F;
    use test_case::test_case;

    use super::*;
    use crate::testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness};

    const IMM_BITS: usize = 12;
    const MAX_INS_CAPACITY: usize = 128;
    const ABS_MAX_IMM: i32 = 1 << (IMM_BITS - 1);

    type Harness = GpuTestChipHarness<
        F,
        Rv32BranchLessThanStep,
        Rv32BranchLessThanAir,
        Rv32BranchLessThanChipGpu,
        Rv32BranchLessThanChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = Rv32BranchLessThanAir::new(
            Rv32BranchAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            BranchLessThanCoreAir::new(bitwise_bus, BranchLessThanOpcode::CLASS_OFFSET),
        );
        let executor = Rv32BranchLessThanStep::new(
            Rv32BranchAdapterStep::new(),
            BranchLessThanOpcode::CLASS_OFFSET,
        );
        let cpu_chip = Rv32BranchLessThanChip::new(
            BranchLessThanFiller::new(
                Rv32BranchAdapterFiller,
                dummy_bitwise_chip,
                BranchLessThanOpcode::CLASS_OFFSET,
            ),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = Rv32BranchLessThanChipGpu::new(
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
        opcode: BranchLessThanOpcode,
    ) {
        let a: [u8; RV32_REGISTER_NUM_LIMBS] = array::from_fn(|_| rng.gen_range(0..=u8::MAX));
        let b: [u8; RV32_REGISTER_NUM_LIMBS] = if rng.gen_bool(0.5) {
            a
        } else {
            array::from_fn(|_| rng.gen_range(0..=u8::MAX))
        };

        let imm = rng.gen_range((-ABS_MAX_IMM)..ABS_MAX_IMM);
        let rs1 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rs2 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, a.map(F::from_canonical_u8));
        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, b.map(F::from_canonical_u8));

        // Use the same pattern as the original test
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_isize(
                opcode.global_opcode(),
                rs1 as isize,
                rs2 as isize,
                imm as isize,
                1,
                1,
            ),
            rng.gen_range(imm.unsigned_abs()..(1 << (PC_BITS - 1))),
        );
    }

    #[test_case(BranchLessThanOpcode::BLT, 100)]
    #[test_case(BranchLessThanOpcode::BLTU, 100)]
    #[test_case(BranchLessThanOpcode::BGE, 100)]
    #[test_case(BranchLessThanOpcode::BGEU, 100)]
    fn test_branch_opcode(opcode: BranchLessThanOpcode, num_ops: usize) {
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
        let mut rng = create_seeded_rng();
        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        // Transfer records from dense to sparse chip
        type Record<'a> = (
            &'a mut Rv32BranchAdapterRecord,
            &'a mut BranchLessThanCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32BranchAdapterStep>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
