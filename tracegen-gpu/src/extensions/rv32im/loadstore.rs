use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_rv32im_circuit::{
    adapters::{Rv32LoadStoreAdapterCols, Rv32LoadStoreAdapterRecord},
    LoadStoreCoreCols, LoadStoreCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use super::cuda::loadstore_cuda;
use crate::{
    primitives::var_range::VariableRangeCheckerChipGPU, testing::get_empty_air_proving_ctx,
};

#[derive(new)]
pub struct Rv32LoadStoreChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32LoadStoreChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32LoadStoreAdapterRecord,
            LoadStoreCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv32LoadStoreAdapterCols::<F>::width()
            + LoadStoreCoreCols::<F, RV32_REGISTER_NUM_LIMBS>::width();
        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        unsafe {
            loadstore_cuda::tracegen(
                d_trace.buffer(),
                padded_height,
                trace_width,
                &d_records,
                height,
                self.pointer_max_bits,
                &self.range_checker.count,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod test {
    use std::array;

    use openvm_circuit::{
        arch::{testing::memory::gen_pointer, EmptyAdapterCoreLayout, MemoryConfig},
        system::memory::merkle::public_values::PUBLIC_VALUES_AS,
    };
    use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
        LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterFiller, Rv32LoadStoreAdapterStep},
        LoadStoreCoreAir, LoadStoreFiller, Rv32LoadStoreAir, Rv32LoadStoreChip, Rv32LoadStoreStep,
    };
    use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::testing::{default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness};

    const IMM_BITS: usize = 16;
    const MAX_INS_CAPACITY: usize = 128;
    type Harness = GpuTestChipHarness<
        F,
        Rv32LoadStoreStep,
        Rv32LoadStoreAir,
        Rv32LoadStoreChipGpu,
        Rv32LoadStoreChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let air = Rv32LoadStoreAir::new(
            Rv32LoadStoreAdapterAir::new(
                tester.memory_bridge(),
                tester.execution_bridge(),
                range_bus,
                tester.address_bits(),
            ),
            LoadStoreCoreAir::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        let executor = Rv32LoadStoreStep::new(
            Rv32LoadStoreAdapterStep::new(tester.address_bits()),
            Rv32LoadStoreOpcode::CLASS_OFFSET,
        );
        let cpu_chip = Rv32LoadStoreChip::<F>::new(
            LoadStoreFiller::new(
                Rv32LoadStoreAdapterFiller::new(tester.address_bits(), dummy_range_checker_chip),
                Rv32LoadStoreOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );

        let gpu_chip = Rv32LoadStoreChipGpu::new(tester.range_checker(), tester.address_bits());

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
        let imm_ext = imm + imm_sign * 0xffff0000;

        let alignment = match opcode {
            LOADW | STOREW => 2,
            LOADHU | STOREH => 1,
            LOADBU | STOREB => 0,
            _ => unreachable!(),
        };

        let is_load = [LOADW, LOADHU, LOADBU].contains(&opcode);
        let mem_as = if is_load { 2 } else { rng.gen_range(2..=4) };

        let ptr_val: u32 =
            rng.gen_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
        let rs1 = ptr_val.wrapping_sub(imm_ext).to_le_bytes();

        let a = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let b = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        let shift_amount = ptr_val % 4;
        tester.write(1, b, rs1.map(F::from_canonical_u8));

        let mut some_prev_data: [F; RV32_REGISTER_NUM_LIMBS] =
            array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));
        let mut read_data: [F; RV32_REGISTER_NUM_LIMBS] =
            array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));

        if is_load {
            if a == 0 {
                some_prev_data = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
            }
            tester.write(1, a, some_prev_data);
            if mem_as == 1 && ptr_val - shift_amount == 0 {
                read_data = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
            }
            tester.write(mem_as, (ptr_val - shift_amount) as usize, read_data);
        } else {
            if mem_as == 4 {
                some_prev_data = array::from_fn(|_| rng.gen());
            }
            if a == 0 {
                read_data = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
            }

            tester.write(mem_as, (ptr_val - shift_amount) as usize, some_prev_data);
            tester.write(1, a, read_data);
        }

        let enabled_write = !(is_load & (a == 0));

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
                    enabled_write as usize,
                    imm_sign as usize,
                ],
            ),
        );
    }

    #[test_case(LOADW, 100)]
    #[test_case(LOADBU, 100)]
    #[test_case(LOADHU, 100)]
    #[test_case(STOREW, 100)]
    #[test_case(STOREB, 100)]
    #[test_case(STOREH, 100)]
    fn test_load_store_tracegen(opcode: Rv32LoadStoreOpcode, num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut mem_config = MemoryConfig::default();
        if [STOREW, STOREB, STOREH].contains(&opcode) {
            mem_config.addr_space_sizes[PUBLIC_VALUES_AS as usize] = 1 << 29;
        }
        let mut tester = GpuChipTestBuilder::volatile(mem_config)
            .with_variable_range_checker(default_var_range_checker_bus());

        let mut harness = create_test_harness(&tester);
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        type Record<'a> = (
            &'a mut Rv32LoadStoreAdapterRecord,
            &'a mut LoadStoreCoreRecord<RV32_REGISTER_NUM_LIMBS>,
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
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
