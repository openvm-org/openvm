use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32CondRdWriteAdapterCols, Rv32RdWriteAdapterRecord, RV32_CELL_BITS},
    Rv32JalLuiCoreCols, Rv32JalLuiCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use super::cuda::jal_lui_cuda::tracegen;
use crate::{
    get_empty_air_proving_ctx,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(new)]
pub struct Rv32JalLuiChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32JalLuiChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(Rv32RdWriteAdapterRecord, Rv32JalLuiCoreRecord)>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width =
            Rv32JalLuiCoreCols::<F>::width() + Rv32CondRdWriteAdapterCols::<F>::width();
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
    use openvm_circuit::arch::EmptyAdapterCoreLayout;
    use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
    use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{
            Rv32CondRdWriteAdapterAir, Rv32CondRdWriteAdapterExecutor,
            Rv32CondRdWriteAdapterFiller, Rv32RdWriteAdapterAir, Rv32RdWriteAdapterExecutor,
            Rv32RdWriteAdapterFiller, Rv32RdWriteAdapterRecord, RV32_CELL_BITS,
        },
        Rv32JalLuiAir, Rv32JalLuiChip, Rv32JalLuiCoreAir, Rv32JalLuiExecutor, Rv32JalLuiFiller,
    };
    use openvm_rv32im_transpiler::Rv32JalLuiOpcode;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use stark_backend_gpu::prelude::F;
    use test_case::test_case;

    use super::*;
    use crate::testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness};

    const IMM_BITS: usize = 12;
    const MAX_INS_CAPACITY: usize = 128;

    type Harness = GpuTestChipHarness<
        F,
        Rv32JalLuiExecutor,
        Rv32JalLuiAir,
        Rv32JalLuiChipGpu,
        Rv32JalLuiChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = Rv32JalLuiAir::new(
            Rv32CondRdWriteAdapterAir::new(Rv32RdWriteAdapterAir::new(
                tester.memory_bridge(),
                tester.execution_bridge(),
            )),
            Rv32JalLuiCoreAir::new(bitwise_bus),
        );
        let executor = Rv32JalLuiExecutor::new(Rv32CondRdWriteAdapterExecutor::new(
            Rv32RdWriteAdapterExecutor,
        ));
        let cpu_chip = Rv32JalLuiChip::<F>::new(
            Rv32JalLuiFiller::new(
                Rv32CondRdWriteAdapterFiller::new(Rv32RdWriteAdapterFiller),
                dummy_bitwise_chip,
            ),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = Rv32JalLuiChipGpu::new(
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
        opcode: Rv32JalLuiOpcode,
    ) {
        let imm: i32 = rng.gen_range(0..(1 << IMM_BITS));
        let imm = match opcode {
            Rv32JalLuiOpcode::JAL => ((imm >> 1) << 2) - (1 << IMM_BITS),
            Rv32JalLuiOpcode::LUI => imm,
        };

        let a = rng.gen_range((opcode == Rv32JalLuiOpcode::LUI) as usize..32) << 2;
        let needs_write = a != 0 || opcode == Rv32JalLuiOpcode::LUI;

        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
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
    #[test_case(Rv32JalLuiOpcode::JAL, 100)]
    #[test_case(Rv32JalLuiOpcode::LUI, 100)]
    fn test_jal_lui(opcode: Rv32JalLuiOpcode, num_ops: usize) {
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
        let mut rng = create_seeded_rng();

        let mut harness = create_test_harness(&tester);
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        // Transfer records from dense to sparse chip
        type Record<'a> = (
            &'a mut Rv32RdWriteAdapterRecord,
            &'a mut Rv32JalLuiCoreRecord,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32CondRdWriteAdapterExecutor>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
