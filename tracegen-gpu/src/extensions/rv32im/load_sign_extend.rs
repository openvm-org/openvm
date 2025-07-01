use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_rv32im_circuit::{
    adapters::Rv32LoadStoreAdapterRecord, LoadSignExtendCoreRecord, Rv32LoadSignExtendAir,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    extensions::rv32im::cuda::load_sign_extend_cuda,
    primitives::var_range::VariableRangeCheckerChipGPU, DeviceChip,
};

#[derive(new)]
pub struct Rv32LoadSignExtendChipGpu {
    pub air: Rv32LoadSignExtendAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for Rv32LoadSignExtendChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            Rv32LoadStoreAdapterRecord,
            LoadSignExtendCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Rv32LoadSignExtendChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.allocated().to_device().unwrap();
        let height = self.current_trace_height();
        let padded_height = next_power_of_two_or_zero(height);
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, self.trace_width());
        unsafe {
            load_sign_extend_cuda::tracegen(
                trace.buffer(),
                padded_height,
                self.trace_width(),
                &d_records,
                height,
                self.pointer_max_bits,
                &self.range_checker.count,
            )
            .unwrap();
        }
        trace
    }
}

#[cfg(test)]
mod test {
    use std::array;

    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS},
        EmptyAdapterCoreLayout, MemoryConfig, NewVmChipWrapper,
    };
    use openvm_circuit_primitives::{
        bitwise_op_lookup::BitwiseOperationLookupBus, var_range::VariableRangeCheckerBus,
    };
    use openvm_instructions::{
        instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterStep},
        LoadSignExtendCoreAir, LoadSignExtendStep, Rv32LoadSignExtendChip, Rv32LoadSignExtendStep,
    };
    use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng, RngCore};
    use test_case::test_case;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const IMM_BITS: usize = 16;
    const MAX_INS_CAPACITY: usize = 128;
    type Rv32LoadSignExtendDenseChip<F> =
        NewVmChipWrapper<F, Rv32LoadSignExtendAir, Rv32LoadSignExtendStep, DenseRecordArena>;

    // Returns write_data
    #[inline(always)]
    fn run_write_data_sign_extend<const NUM_CELLS: usize>(
        opcode: Rv32LoadStoreOpcode,
        read_data: [u8; NUM_CELLS],
        shift: usize,
    ) -> [u8; NUM_CELLS] {
        match (opcode, shift) {
        (LOADH, 0) | (LOADH, 2) => {
            let ext = (read_data[NUM_CELLS / 2 - 1 + shift] >> 7) * u8::MAX;
            array::from_fn(|i| {
                if i < NUM_CELLS / 2 {
                    read_data[i + shift]
                } else {
                    ext
                }
            })
        }
        (LOADB, 0) | (LOADB, 1) | (LOADB, 2) | (LOADB, 3) => {
            let ext = (read_data[shift] >> 7) * u8::MAX;
            array::from_fn(|i| {
                if i == 0 {
                    read_data[i + shift]
                } else {
                    ext
                }
            })
        }
        _ => unreachable!(
            "unaligned memory access not supported by this execution environment: {opcode:?}, shift: {shift}"
        ),
    }
    }

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> Rv32LoadSignExtendDenseChip<F> {
        let range_checker_chip = tester.memory_controller().range_checker.clone();
        let mut chip = Rv32LoadSignExtendDenseChip::new(
            Rv32LoadSignExtendAir::new(
                Rv32LoadStoreAdapterAir::new(
                    tester.memory_bridge(),
                    tester.execution_bridge(),
                    range_checker_chip.bus(),
                    tester.address_bits(),
                ),
                LoadSignExtendCoreAir::new(range_checker_chip.bus()),
            ),
            LoadSignExtendStep::new(
                Rv32LoadStoreAdapterStep::new(tester.address_bits(), range_checker_chip.clone()),
                range_checker_chip.clone(),
            ),
            tester.cpu_memory_helper(),
        );

        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(tester: &mut GpuChipTestBuilder) -> Rv32LoadSignExtendChip<F> {
        let range_checker_chip = tester.memory_controller().range_checker.clone();
        let mut chip = Rv32LoadSignExtendChip::new(
            Rv32LoadSignExtendAir::new(
                Rv32LoadStoreAdapterAir::new(
                    tester.memory_bridge(),
                    tester.execution_bridge(),
                    range_checker_chip.bus(),
                    tester.address_bits(),
                ),
                LoadSignExtendCoreAir::new(range_checker_chip.bus()),
            ),
            LoadSignExtendStep::new(
                Rv32LoadStoreAdapterStep::new(tester.address_bits(), range_checker_chip.clone()),
                range_checker_chip.clone(),
            ),
            tester.cpu_memory_helper(),
        );

        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        chip: &mut Rv32LoadSignExtendDenseChip<F>,
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
        let a = gen_pointer(rng, 4);
        let b = gen_pointer(rng, 4);

        let shift_amount = ptr_val % 4;
        tester.write(1, b, rs1.map(F::from_canonical_u8));

        let some_prev_data: [F; RV32_REGISTER_NUM_LIMBS] = if a != 0 {
            rng.next_u32().to_le_bytes().map(F::from_canonical_u8)
        } else {
            [F::ZERO; RV32_REGISTER_NUM_LIMBS]
        };

        let read_data: [u8; RV32_REGISTER_NUM_LIMBS] = rng.next_u32().to_le_bytes();

        tester.write(1, a, some_prev_data);

        let mem_as = rng.gen_range(1..3);
        tester.write(
            mem_as,
            (ptr_val - shift_amount) as usize,
            read_data.map(F::from_canonical_u8),
        );

        tester.execute(
            chip,
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

        let write_data = run_write_data_sign_extend(opcode, read_data, shift_amount as usize);
        if a != 0 {
            assert_eq!(write_data.map(F::from_canonical_u8), tester.read::<4>(1, a));
        } else {
            assert_eq!([F::ZERO; 4], tester.read::<4>(1, a));
        }
    }

    #[test_case(LOADB, 100)]
    #[test_case(LOADH, 100)]
    fn test_load_sign_extend_tracegen(opcode: Rv32LoadStoreOpcode, num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));
        // CPU execution
        let mut dense_chip = create_test_dense_chip(&tester);
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut dense_chip, &mut rng, opcode);
        }

        let mut sparse_chip = create_test_sparse_chip(&mut tester);

        type Record<'a> = (
            &'a mut Rv32LoadStoreAdapterRecord,
            &'a mut LoadSignExtendCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        );

        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut sparse_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32LoadStoreAdapterStep>::new(),
            );

        // GPU tracegen
        let mem_config = MemoryConfig::default();
        let var_range_gpu_chip = Arc::new(VariableRangeCheckerChipGPU::new(
            VariableRangeCheckerBus::new(4, mem_config.decomp),
        ));
        let gpu_chip = Rv32LoadSignExtendChipGpu::new(
            sparse_chip.air.clone(),
            var_range_gpu_chip,
            tester.address_bits(),
            dense_chip.arena,
        );

        // `gpu_chip` does GPU tracegen, `sparse_chip` does CPU tracegen. Must check that they are the same
        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
