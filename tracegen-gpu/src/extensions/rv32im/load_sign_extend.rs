use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_rv32im_circuit::{
    adapters::{Rv32LoadStoreAdapterCols, Rv32LoadStoreAdapterRecord},
    LoadSignExtendCoreCols, LoadSignExtendCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use crate::{
    extensions::rv32im::cuda::load_sign_extend_cuda,
    primitives::var_range::VariableRangeCheckerChipGPU, testing::get_empty_air_proving_ctx,
};

#[derive(new)]
pub struct Rv32LoadSignExtendChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32LoadSignExtendChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32LoadStoreAdapterRecord,
            LoadSignExtendCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv32LoadStoreAdapterCols::<F>::width()
            + LoadSignExtendCoreCols::<F, RV32_REGISTER_NUM_LIMBS>::width();
        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        unsafe {
            load_sign_extend_cuda::tracegen(
                d_trace.buffer(),
                padded_height,
                trace_width,
                &d_records,
                height,
                self.pointer_max_bits,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod test {

    use openvm_circuit::arch::{testing::memory::gen_pointer, EmptyAdapterCoreLayout};
    use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
    use openvm_instructions::{
        instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterFiller, Rv32LoadStoreAdapterStep},
        LoadSignExtendCoreAir, LoadSignExtendFiller, Rv32LoadSignExtendAir, Rv32LoadSignExtendChip,
        Rv32LoadSignExtendStep,
    };
    use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng, RngCore};
    use test_case::test_case;

    use super::*;
    use crate::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
        GpuTestChipHarness,
    };

    const IMM_BITS: usize = 16;
    const MAX_INS_CAPACITY: usize = 128;

    type Harness = GpuTestChipHarness<
        F,
        Rv32LoadSignExtendStep,
        Rv32LoadSignExtendAir,
        Rv32LoadSignExtendChipGpu,
        Rv32LoadSignExtendChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let air = Rv32LoadSignExtendAir::new(
            Rv32LoadStoreAdapterAir::new(
                tester.memory_bridge(),
                tester.execution_bridge(),
                range_bus,
                tester.address_bits(),
            ),
            LoadSignExtendCoreAir::new(range_bus),
        );
        let executor =
            Rv32LoadSignExtendStep::new(Rv32LoadStoreAdapterStep::new(tester.address_bits()));
        let cpu_chip = Rv32LoadSignExtendChip::<F>::new(
            LoadSignExtendFiller::new(
                Rv32LoadStoreAdapterFiller::new(
                    tester.address_bits(),
                    dummy_range_checker_chip.clone(),
                ),
                dummy_range_checker_chip,
            ),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = Rv32LoadSignExtendChipGpu::new(
            tester.range_checker(),
            tester.address_bits(),
            tester.timestamp_max_bits(),
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut Harness,
        rng: &mut StdRng,
        opcode: Rv32LoadStoreOpcode,
    ) {
        let imm = rng.gen_range(0..(1 << IMM_BITS));
        let imm_sign = rng.gen_range(0..2);
        let imm_ext = imm + imm_sign * (0xffff0000);

        let alignment = match opcode {
            LOADB => 0,
            LOADH => 1,
            _ => unreachable!(),
        };

        let ptr_val: u32 =
            rng.gen_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
        let rs1 = ptr_val.wrapping_sub(imm_ext).to_le_bytes();
        let a = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let b = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        let shift_amount = ptr_val % 4;
        tester.write(1, b, rs1.map(F::from_canonical_u8));

        let some_prev_data: [F; RV32_REGISTER_NUM_LIMBS] = if a != 0 {
            rng.next_u32().to_le_bytes().map(F::from_canonical_u8)
        } else {
            [F::ZERO; RV32_REGISTER_NUM_LIMBS]
        };

        let read_data: [u8; RV32_REGISTER_NUM_LIMBS] = rng.next_u32().to_le_bytes();

        tester.write(1, a, some_prev_data);

        let mem_as = 2;
        tester.write(
            mem_as,
            (ptr_val - shift_amount) as usize,
            read_data.map(F::from_canonical_u8),
        );

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(
                opcode.global_opcode(),
                [
                    a,
                    b,
                    imm as usize,
                    1,
                    mem_as,
                    (a != 0) as usize,
                    imm_sign as usize,
                ],
            ),
        );
    }

    #[test_case(LOADB, 100)]
    #[test_case(LOADH, 100)]
    fn test_load_sign_extend_tracegen(opcode: Rv32LoadStoreOpcode, num_ops: usize) {
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
        let mut rng = create_seeded_rng();

        let mut harness = create_test_harness(&tester);
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        type Record<'a> = (
            &'a mut Rv32LoadStoreAdapterRecord,
            &'a mut LoadSignExtendCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        );

        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32LoadStoreAdapterStep>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
