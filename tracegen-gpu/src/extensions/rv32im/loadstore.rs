use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_rv32im_circuit::{
    adapters::Rv32LoadStoreAdapterRecord, LoadStoreCoreRecord, Rv32LoadStoreAir,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use super::cuda::loadstore_cuda;
use crate::{primitives::var_range::VariableRangeCheckerChipGPU, DeviceChip};

#[derive(new)]
pub struct Rv32LoadStoreChipGpu {
    pub air: Rv32LoadStoreAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for Rv32LoadStoreChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            Rv32LoadStoreAdapterRecord,
            LoadStoreCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Rv32LoadStoreChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.allocated().to_device().unwrap();
        let height = self.current_trace_height();
        let padded_height = next_power_of_two_or_zero(height);
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, self.trace_width());
        unsafe {
            loadstore_cuda::tracegen(
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

    use openvm_circuit::{
        arch::{
            testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS},
            EmptyAdapterCoreLayout, MemoryConfig, NewVmChipWrapper,
        },
        system::memory::merkle::public_values::PUBLIC_VALUES_AS,
    };
    use openvm_circuit_primitives::{
        bitwise_op_lookup::BitwiseOperationLookupBus, var_range::VariableRangeCheckerBus,
    };
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
        LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterStep},
        LoadStoreCoreAir, LoadStoreStep, Rv32LoadStoreChip, Rv32LoadStoreStep,
    };
    use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
    use openvm_stark_backend::{
        p3_field::{FieldAlgebra, PrimeField32},
        verifier::VerificationError,
    };
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const IMM_BITS: usize = 16;
    const MAX_INS_CAPACITY: usize = 128;
    type Rv32LoadStoreDenseChip<F> =
        NewVmChipWrapper<F, Rv32LoadStoreAir, Rv32LoadStoreStep, DenseRecordArena>;

    // Returns write_data
    #[inline(always)]
    fn run_write_data<const NUM_CELLS: usize>(
        opcode: Rv32LoadStoreOpcode,
        read_data: [u8; NUM_CELLS],
        prev_data: [u32; NUM_CELLS],
        shift: usize,
    ) -> [u32; NUM_CELLS] {
        match (opcode, shift) {
            (LOADW, 0) => {
                read_data.map(|x| x as u32)
            },
            (LOADBU, 0) | (LOADBU, 1) | (LOADBU, 2) | (LOADBU, 3) => {
               let mut wrie_data = [0; NUM_CELLS];
               wrie_data[0] = read_data[shift] as u32;
               wrie_data
            }
            (LOADHU, 0) | (LOADHU, 2) => {
                let mut write_data = [0; NUM_CELLS];
                for (i, cell) in write_data.iter_mut().take(NUM_CELLS / 2).enumerate() {
                    *cell = read_data[i + shift] as u32;
                }
                write_data
            }
            (STOREW, 0) => {
                read_data.map(|x| x as u32)
            },
            (STOREB, 0) | (STOREB, 1) | (STOREB, 2) | (STOREB, 3) => {
                let mut write_data = prev_data;
                write_data[shift] = read_data[0] as u32;
                write_data
            }
            (STOREH, 0) | (STOREH, 2) => {
                array::from_fn(|i| {
                    if i >= shift && i < (NUM_CELLS / 2 + shift){
                        read_data[i - shift] as u32
                    } else {
                        prev_data[i]
                    }
                })
            }
            // Currently the adapter AIR requires `ptr_val` to be aligned to the data size in bytes.
            // The circuit requires that `shift = ptr_val % 4` so that `ptr_val - shift` is a multiple of 4.
            // This requirement is non-trivial to remove, because we use it to ensure that `ptr_val - shift + 4 <= 2^pointer_max_bits`.
            _ => unreachable!(
                "unaligned memory access not supported by this execution environment: {opcode:?}, shift: {shift}"
            ),
        }
    }

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> Rv32LoadStoreDenseChip<F> {
        let range_checker_chip = tester.memory_controller().range_checker.clone();
        let mut chip = Rv32LoadStoreDenseChip::new(
            Rv32LoadStoreAir::new(
                Rv32LoadStoreAdapterAir::new(
                    tester.memory_bridge(),
                    tester.execution_bridge(),
                    range_checker_chip.bus(),
                    tester.address_bits(),
                ),
                LoadStoreCoreAir::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
            ),
            LoadStoreStep::new(
                Rv32LoadStoreAdapterStep::new(tester.address_bits(), range_checker_chip.clone()),
                Rv32LoadStoreOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );

        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(tester: &mut GpuChipTestBuilder) -> Rv32LoadStoreChip<F> {
        let range_checker_chip = tester.memory_controller().range_checker.clone();
        let mut chip = Rv32LoadStoreChip::new(
            Rv32LoadStoreAir::new(
                Rv32LoadStoreAdapterAir::new(
                    tester.memory_bridge(),
                    tester.execution_bridge(),
                    range_checker_chip.bus(),
                    tester.address_bits(),
                ),
                LoadStoreCoreAir::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
            ),
            LoadStoreStep::new(
                Rv32LoadStoreAdapterStep::new(tester.address_bits(), range_checker_chip.clone()),
                Rv32LoadStoreOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );

        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        chip: &mut Rv32LoadStoreDenseChip<F>,
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
        let mem_as = if is_load {
            rng.gen_range(1..=2)
        } else {
            rng.gen_range(2..=4)
        };

        let ptr_val: u32 =
            rng.gen_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
        let rs1 = ptr_val.wrapping_sub(imm_ext).to_le_bytes();

        let a = gen_pointer(rng, 4);
        let b = gen_pointer(rng, 4);

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
            chip,
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

        let write_data = run_write_data(
            opcode,
            read_data.map(|x| x.as_canonical_u32() as u8),
            some_prev_data.map(|x| x.as_canonical_u32()),
            shift_amount as usize,
        )
        .map(F::from_canonical_u32);
        if is_load {
            if enabled_write {
                assert_eq!(write_data, tester.read::<4>(1, a));
            } else {
                assert_eq!([F::ZERO; RV32_REGISTER_NUM_LIMBS], tester.read::<4>(1, a));
            }
        } else {
            assert_eq!(
                write_data,
                tester.read::<4>(mem_as, (ptr_val - shift_amount) as usize)
            );
        }
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
            &'a mut LoadStoreCoreRecord<RV32_REGISTER_NUM_LIMBS>,
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
        let gpu_chip = Rv32LoadStoreChipGpu::new(
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
