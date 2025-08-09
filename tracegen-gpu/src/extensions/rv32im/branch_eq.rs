use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32BranchAdapterCols, Rv32BranchAdapterRecord, RV32_REGISTER_NUM_LIMBS},
    BranchEqualCoreCols, BranchEqualCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use super::cuda::beq_cuda::tracegen;
use crate::{get_empty_air_proving_ctx, primitives::var_range::VariableRangeCheckerChipGPU};

#[derive(new)]
pub struct Rv32BranchEqualChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32BranchEqualChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BranchAdapterRecord,
            BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = BranchEqualCoreCols::<F, RV32_REGISTER_NUM_LIMBS>::width()
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
    use openvm_instructions::{
        instruction::Instruction,
        program::{DEFAULT_PC_STEP, PC_BITS},
        LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        adapters::{
            Rv32BranchAdapterAir, Rv32BranchAdapterExecutor, Rv32BranchAdapterFiller,
            Rv32BranchAdapterRecord, RV32_REGISTER_NUM_LIMBS,
        },
        BranchEqualCoreAir, BranchEqualCoreRecord, BranchEqualFiller, Rv32BranchEqualAir,
        Rv32BranchEqualChip, Rv32BranchEqualExecutor,
    };
    use openvm_rv32im_transpiler::BranchEqualOpcode;
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
        Rv32BranchEqualExecutor,
        Rv32BranchEqualAir,
        Rv32BranchEqualChipGpu,
        Rv32BranchEqualChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        let air = Rv32BranchEqualAir::new(
            Rv32BranchAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        );
        let executor = Rv32BranchEqualExecutor::new(
            Rv32BranchAdapterExecutor,
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        );
        let cpu_chip = Rv32BranchEqualChip::new(
            BranchEqualFiller::new(
                Rv32BranchAdapterFiller,
                BranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            tester.dummy_memory_helper(),
        );

        let gpu_chip =
            Rv32BranchEqualChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut Harness,
        rng: &mut StdRng,
        opcode: BranchEqualOpcode,
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

        let initial_pc = rng.gen_range(imm.unsigned_abs()..(1 << (PC_BITS - 1)));
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
            initial_pc,
        );
    }
    #[test_case(BranchEqualOpcode::BEQ, 100)]
    #[test_case(BranchEqualOpcode::BNE, 100)]
    fn test_beq_opcode(opcode: BranchEqualOpcode, num_ops: usize) {
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
            &'a mut BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32BranchAdapterExecutor>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
