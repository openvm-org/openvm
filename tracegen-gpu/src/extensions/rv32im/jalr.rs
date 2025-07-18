use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32JalrAdapterCols, Rv32JalrAdapterRecord, RV32_CELL_BITS},
    Rv32JalrCoreCols, Rv32JalrCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use super::cuda::jalr::tracegen;
use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    testing::get_empty_air_proving_ctx,
};

#[derive(new)]
pub struct Rv32JalrChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32JalrChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(Rv32JalrAdapterRecord, Rv32JalrCoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv32JalrCoreCols::<F>::width() + Rv32JalrAdapterCols::<F>::width();
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
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::EmptyAdapterCoreLayout;
    use openvm_circuit_primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChip, var_range::VariableRangeCheckerChip,
    };
    use openvm_instructions::{
        instruction::Instruction, program::PC_BITS, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        adapters::{
            Rv32JalrAdapterAir, Rv32JalrAdapterFiller, Rv32JalrAdapterRecord, Rv32JalrAdapterStep,
            RV32_CELL_BITS,
        },
        Rv32JalrAir, Rv32JalrChip, Rv32JalrCoreAir, Rv32JalrCoreRecord, Rv32JalrFiller,
        Rv32JalrStep,
    };
    use openvm_rv32im_transpiler::Rv32JalrOpcode::{self, *};
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use stark_backend_gpu::prelude::F;

    use super::*;
    use crate::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, memory::gen_pointer,
        GpuChipTestBuilder, GpuTestChipHarness,
    };

    const IMM_BITS: usize = 12;
    const MAX_INS_CAPACITY: usize = 128;

    type Harness =
        GpuTestChipHarness<F, Rv32JalrStep, Rv32JalrAir, Rv32JalrChipGpu, Rv32JalrChip<F>>;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = Rv32JalrAir::new(
            Rv32JalrAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
            Rv32JalrCoreAir::new(bitwise_bus, range_bus),
        );
        let executor = Rv32JalrStep::new(Rv32JalrAdapterStep);
        let cpu_chip = Rv32JalrChip::<F>::new(
            Rv32JalrFiller::new(
                Rv32JalrAdapterFiller,
                dummy_bitwise_chip,
                dummy_range_checker_chip,
            ),
            tester.cpu_memory_helper(),
        );

        let gpu_chip = Rv32JalrChipGpu::new(tester.range_checker(), tester.bitwise_op_lookup());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut Harness,
        rng: &mut StdRng,
        opcode: Rv32JalrOpcode,
    ) {
        let imm = rng.gen_range(0..(1 << IMM_BITS)) as usize;
        let imm_sign = rng.gen_range(0..2) as usize; // 0 or 1
        let imm_ext = imm + (imm_sign * 0xffff0000);

        let a = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let b = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let initial_pc = rng.gen_range(0..(1 << PC_BITS));
        let to_pc = rng.gen_range(0..(1 << PC_BITS));

        let rs1_val = (to_pc as u32).wrapping_sub(imm_ext as u32);
        let rs1_bytes = rs1_val.to_le_bytes().map(F::from_canonical_u8);

        tester.write(1, b, rs1_bytes);

        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(opcode.global_opcode(), [a, b, imm, 1, 0, 1, imm_sign]),
            initial_pc,
        );
    }

    #[test]
    fn rand_jalr_tracegen_test() {
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
        let mut rng = create_seeded_rng();

        let mut harness = create_test_harness(&tester);
        let num_ops = 100;
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, JALR);
        }

        type Record<'a> = (&'a mut Rv32JalrAdapterRecord, &'a mut Rv32JalrCoreRecord);
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32JalrAdapterStep>::new(),
            );

        // TODO[INT-4481]: Because memory is not implemented yet, this test fails
        // interaction constraints. Once memory is completed, we should make sure
        // that verification passes.
        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
