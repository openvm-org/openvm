use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{
        Rv32BaseAluAdapterCols, Rv32BaseAluAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    LessThanCoreCols, LessThanCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use super::cuda::less_than_cuda::tracegen as rv32_less_than_tracegen;
use crate::{
    get_empty_air_proving_ctx,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(new)]
pub struct Rv32LessThanChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32LessThanChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BaseAluAdapterRecord,
            LessThanCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv32BaseAluAdapterCols::<F>::width()
            + LessThanCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            rv32_less_than_tracegen(
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
    use std::array;

    use openvm_circuit::arch::EmptyAdapterCoreLayout;
    use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
    use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{
            Rv32BaseAluAdapterAir, Rv32BaseAluAdapterExecutor, Rv32BaseAluAdapterFiller,
            RV_IS_TYPE_IMM_BITS,
        },
        LessThanCoreAir, LessThanFiller, Rv32LessThanAir, Rv32LessThanChip, Rv32LessThanExecutor,
    };
    use openvm_rv32im_transpiler::LessThanOpcode;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::testing::{
        default_bitwise_lookup_bus, memory::gen_pointer, GpuChipTestBuilder, GpuTestChipHarness,
    };

    const MAX_INS_CAPACITY: usize = 128;

    type Harness = GpuTestChipHarness<
        F,
        Rv32LessThanExecutor,
        Rv32LessThanAir,
        Rv32LessThanChipGpu,
        Rv32LessThanChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let air = Rv32LessThanAir::new(
            Rv32BaseAluAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
            ),
            LessThanCoreAir::new(bitwise_bus, LessThanOpcode::CLASS_OFFSET),
        );
        let executor =
            Rv32LessThanExecutor::new(Rv32BaseAluAdapterExecutor, LessThanOpcode::CLASS_OFFSET);
        let cpu_chip = Rv32LessThanChip::<F>::new(
            LessThanFiller::new(
                Rv32BaseAluAdapterFiller::new(dummy_bitwise_chip.clone()),
                dummy_bitwise_chip,
                LessThanOpcode::CLASS_OFFSET,
            ),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = Rv32LessThanChipGpu::new(
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
        opcode: LessThanOpcode,
    ) {
        let b: [u8; RV32_REGISTER_NUM_LIMBS] = array::from_fn(|_| rng.gen_range(0..=u8::MAX));
        let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        tester.write(
            RV32_REGISTER_AS as usize,
            rs1_ptr,
            b.map(F::from_canonical_u8),
        );

        let is_imm = rng.gen_bool(0.5);
        let rs2_ptr = if is_imm {
            let mut imm: u32 = rng.gen_range(0..(1 << RV_IS_TYPE_IMM_BITS));
            if (imm & 0x800) != 0 {
                imm |= !0xFFF
            }
            (imm & 0xFFFFFF) as usize
        } else {
            let c: [u8; RV32_REGISTER_NUM_LIMBS] = array::from_fn(|_| rng.gen_range(0..=u8::MAX));
            let rs2_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
            tester.write(
                RV32_REGISTER_AS as usize,
                rs2_ptr,
                c.map(F::from_canonical_u8),
            );
            rs2_ptr
        };

        let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(
                opcode.global_opcode(),
                [rd_ptr, rs1_ptr, rs2_ptr, 1, (!is_imm) as usize],
            ),
        );
    }

    #[test_case(LessThanOpcode::SLT, 100)]
    #[test_case(LessThanOpcode::SLTU, 100)]
    fn less_than_gpu_chip_test(opcode: LessThanOpcode, num_ops: usize) {
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
        let mut rng = create_seeded_rng();

        let mut harness = create_test_harness(&tester);
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        type Record<'a> = (
            &'a mut Rv32BaseAluAdapterRecord,
            &'a mut LessThanCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
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
