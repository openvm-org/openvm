use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};

use super::cuda::hintstore::tracegen;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_rv32im_circuit::{
    Rv32HintStoreAir, Rv32HintStoreCols, Rv32HintStoreLayout, Rv32HintStoreRecordMut,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::CudaError},
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};

#[derive(new)]
pub struct Rv32HintStoreChipGpu {
    pub air: Rv32HintStoreAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for Rv32HintStoreChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        // TEMP[arayi]: This is temporary we probably need to get rid of `current_trace_height` or add a counter to `arena`
        self.arena.allocated().len()
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

// This is the info needed by each row to do parallel tracegen
#[repr(C)]
#[derive(new)]
pub struct OffsetInfo {
    pub record_offset: u32,
    pub local_idx: u32,
}

impl DeviceChip<SC, GpuBackend> for Rv32HintStoreChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let width = Rv32HintStoreCols::<u8>::width();
        let records = self.arena.allocated();

        // TEMP[arayi]: Temporary hack to get mut access to `records`, should have `self` or `&mut self` as a parameter
        // **SAFATY**: `records` should be non-empty at this point
        let records =
            unsafe { std::slice::from_raw_parts_mut(records.as_ptr() as *mut u8, records.len()) };

        let mut offsets = Vec::<OffsetInfo>::new();
        let mut offset = 0;

        while offset < records.len() {
            let prev_offset = offset;
            let record = RecordSeeker::<
                DenseRecordArena,
                Rv32HintStoreRecordMut,
                Rv32HintStoreLayout,
            >::get_record_at(&mut offset, records);
            for idx in 0..record.inner.num_words {
                offsets.push(OffsetInfo::new(prev_offset as u32, idx));
            }
        }

        let d_records = records.to_device().unwrap();
        let d_record_offsets = offsets.to_device().unwrap();

        let trace_height = next_power_of_two_or_zero(offsets.len());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, width);

        unsafe {
            tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                offsets.len() as u32,
                &d_record_offsets,
                self.pointer_max_bits,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();
        }

        trace
    }
}

#[cfg(test)]
mod test {
    use crate::testing::GpuChipTestBuilder;

    use super::*;
    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS, RANGE_CHECKER_BUS},
        ExecutionBridge, InstructionExecutor, MemoryConfig, NewVmChipWrapper,
    };
    use openvm_circuit_primitives::{
        bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
        var_range::VariableRangeCheckerBus,
    };
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
        VmOpcode,
    };
    use openvm_rv32im_circuit::{Rv32HintStoreChip, Rv32HintStoreStep};
    use openvm_rv32im_transpiler::Rv32HintStoreOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng, RngCore};
    use Rv32HintStoreOpcode::*;

    const MAX_INS_CAPACITY: usize = 1024;
    type Rv32HintStoreDenseChip<F> =
        NewVmChipWrapper<F, Rv32HintStoreAir, Rv32HintStoreStep, DenseRecordArena>;

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> Rv32HintStoreDenseChip<F> {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

        let mut chip = Rv32HintStoreDenseChip::new(
            Rv32HintStoreAir::new(
                ExecutionBridge::new(tester.execution_bus(), tester.program_bus()),
                tester.memory_bridge(),
                bitwise_chip.bus(),
                0,
                tester.address_bits(),
            ),
            Rv32HintStoreStep::new(bitwise_chip.clone(), tester.address_bits(), 0),
            tester.cpu_memory_helper(),
        );

        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(
        tester: &mut GpuChipTestBuilder,
    ) -> (
        Rv32HintStoreChip<F>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

        let mut chip = Rv32HintStoreChip::new(
            Rv32HintStoreAir::new(
                ExecutionBridge::new(tester.execution_bus(), tester.program_bus()),
                tester.memory_bridge(),
                bitwise_chip.bus(),
                0,
                tester.address_bits(),
            ),
            Rv32HintStoreStep::new(bitwise_chip.clone(), tester.address_bits(), 0),
            tester.cpu_memory_helper(),
        );

        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        (chip, bitwise_chip)
    }

    fn set_and_execute<E: InstructionExecutor<F>>(
        tester: &mut GpuChipTestBuilder,
        chip: &mut E,
        rng: &mut StdRng,
        opcode: Rv32HintStoreOpcode,
        num_words: Option<usize>,
    ) {
        let num_words = num_words.unwrap_or(match opcode {
            HINT_STOREW => 1,
            HINT_BUFFER => rng.gen_range(1..28),
        }) as u32;

        let a = if opcode == HINT_BUFFER {
            let a = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
            tester.write(
                RV32_REGISTER_AS as usize,
                a,
                num_words.to_le_bytes().map(F::from_canonical_u8),
            );
            a
        } else {
            0
        };

        let mem_ptr = gen_pointer(rng, 4) as u32;
        let b = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        tester.write(1, b, mem_ptr.to_le_bytes().map(F::from_canonical_u8));

        let mut input = Vec::with_capacity(num_words as usize * 4);
        for _ in 0..num_words {
            let data = rng.next_u32().to_le_bytes().map(F::from_canonical_u8);
            input.extend(data);
            tester.streams.hint_stream.extend(data);
        }

        tester.execute(
            chip,
            &Instruction::from_usize(
                VmOpcode::from_usize(opcode as usize),
                [a, b, 0, RV32_REGISTER_AS as usize, RV32_MEMORY_AS as usize],
            ),
        );

        for idx in 0..num_words as usize {
            let data = tester.read::<4>(RV32_MEMORY_AS as usize, mem_ptr as usize + idx * 4);

            let expected: [F; 4] = input[idx * 4..(idx + 1) * 4].try_into().unwrap();
            assert_eq!(data, expected);
        }
    }

    #[test]
    fn test_hintstore_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));
        // CPU execution
        let mut dense_chip = create_test_dense_chip(&tester);
        let num_ops = 50;
        for _ in 0..num_ops {
            let opcode = if rng.gen_bool(0.5) {
                HINT_STOREW
            } else {
                HINT_BUFFER
            };
            set_and_execute(&mut tester, &mut dense_chip, &mut rng, opcode, None);
        }

        let (mut sparse_chip, sparse_bitwise_chip) = create_test_sparse_chip(&mut tester);
        dense_chip
            .arena
            .get_record_seeker::<Rv32HintStoreRecordMut, _>()
            .transfer_to_matrix_arena(&mut sparse_chip.arena);

        // GPU tracegen
        let bitwise_gpu_chip = Arc::new(BitwiseOperationLookupChipGPU::<RV32_CELL_BITS>::new(
            sparse_bitwise_chip.bus(),
        ));
        let mem_config = MemoryConfig::default();
        let var_range_gpu_chip = Arc::new(VariableRangeCheckerChipGPU::new(
            VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, mem_config.decomp),
        ));
        let gpu_chip = Rv32HintStoreChipGpu::new(
            sparse_chip.air,
            var_range_gpu_chip,
            bitwise_gpu_chip,
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
