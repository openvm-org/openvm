use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_rv32im_circuit::{Rv32HintStoreCols, Rv32HintStoreLayout, Rv32HintStoreRecordMut};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use super::cuda::hintstore::tracegen;
use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    testing::get_empty_air_proving_ctx,
};

#[derive(new)]
pub struct Rv32HintStoreChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
}

// This is the info needed by each row to do parallel tracegen
#[repr(C)]
#[derive(new)]
pub struct OffsetInfo {
    pub record_offset: u32,
    pub local_idx: u32,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32HintStoreChipGpu {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let width = Rv32HintStoreCols::<u8>::width();
        let records = arena.allocated_mut();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

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
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, width);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                offsets.len() as u32,
                &d_record_offsets,
                self.pointer_max_bits as u32,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod test {
    use openvm_circuit::arch::testing::memory::gen_pointer;
    use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
        LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        Rv32HintStoreAir, Rv32HintStoreChip, Rv32HintStoreFiller, Rv32HintStoreStep,
    };
    use openvm_rv32im_transpiler::Rv32HintStoreOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng, RngCore};
    use Rv32HintStoreOpcode::*;

    use super::*;
    use crate::testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness};

    const MAX_INS_CAPACITY: usize = 1024;
    type Harness = GpuTestChipHarness<
        F,
        Rv32HintStoreStep,
        Rv32HintStoreAir,
        Rv32HintStoreChipGpu,
        Rv32HintStoreChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = Rv32HintStoreAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            Rv32HintStoreOpcode::CLASS_OFFSET,
            tester.address_bits(),
        );
        let executor =
            Rv32HintStoreStep::new(tester.address_bits(), Rv32HintStoreOpcode::CLASS_OFFSET);
        let cpu_chip = Rv32HintStoreChip::<F>::new(
            Rv32HintStoreFiller::new(tester.address_bits(), dummy_bitwise_chip),
            tester.cpu_memory_helper(),
        );

        let gpu_chip = Rv32HintStoreChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            tester.address_bits(),
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut Harness,
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
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(
                opcode.global_opcode(),
                [a, b, 0, RV32_REGISTER_AS as usize, RV32_MEMORY_AS as usize],
            ),
        );
    }

    #[test]
    fn test_hintstore_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let mut harness = create_test_harness(&tester);
        let num_ops = 50;
        for _ in 0..num_ops {
            let opcode = if rng.gen_bool(0.5) {
                HINT_STOREW
            } else {
                HINT_BUFFER
            };
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode, None);
        }

        harness
            .dense_arena
            .get_record_seeker::<_, Rv32HintStoreLayout>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
