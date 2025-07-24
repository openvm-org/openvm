use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{
        Rv32MultAdapterCols, Rv32MultAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    MultiplicationCoreCols, MultiplicationCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use super::cuda::mul_cuda::tracegen as mul_tracegen;
use crate::{
    get_empty_air_proving_ctx,
    primitives::{range_tuple::RangeTupleCheckerChipGPU, var_range::VariableRangeCheckerChipGPU},
    UInt2,
};

#[derive(new)]
pub struct Rv32MultiplicationChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32MultiplicationChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32MultAdapterRecord,
            MultiplicationCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width =
            MultiplicationCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width()
                + Rv32MultAdapterCols::<F>::width();

        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            mul_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
                &self.range_tuple_checker.count,
                tuple_checker_sizes,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, RANGE_TUPLE_CHECKER_BUS},
        EmptyAdapterCoreLayout,
    };
    use openvm_circuit_primitives::range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip};
    use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{Rv32MultAdapterAir, Rv32MultAdapterFiller, Rv32MultAdapterStep},
        MultiplicationCoreAir, MultiplicationCoreRecord, MultiplicationFiller,
        Rv32MultiplicationAir, Rv32MultiplicationChip, Rv32MultiplicationStep,
    };
    use openvm_rv32im_transpiler::MulOpcode;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};

    use super::*;
    use crate::testing::{GpuChipTestBuilder, GpuTestChipHarness};

    const MAX_INS_CAPACITY: usize = 128;
    const TUPLE_CHECKER_SIZES: [u32; 2] = [
        (1 << RV32_CELL_BITS) as u32,
        (8 * (1 << RV32_CELL_BITS)) as u32,
    ];

    type Harness = GpuTestChipHarness<
        F,
        Rv32MultiplicationStep,
        Rv32MultiplicationAir,
        Rv32MultiplicationChipGpu,
        Rv32MultiplicationChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_tuple_bus =
            RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_tuple_chip = Arc::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

        let air = Rv32MultiplicationAir::new(
            Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            MultiplicationCoreAir::new(range_tuple_bus, MulOpcode::CLASS_OFFSET),
        );
        let executor = Rv32MultiplicationStep::new(Rv32MultAdapterStep, MulOpcode::CLASS_OFFSET);
        let cpu_chip = Rv32MultiplicationChip::<F>::new(
            MultiplicationFiller::new(
                Rv32MultAdapterFiller,
                dummy_range_tuple_chip,
                MulOpcode::CLASS_OFFSET,
            ),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = Rv32MultiplicationChipGpu::new(
            tester.range_checker(),
            tester.range_tuple_checker(),
            tester.timestamp_max_bits(),
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut Harness,
        rng: &mut StdRng,
        opcode: MulOpcode,
    ) {
        let rd = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let ptr1 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let ptr2 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        let val1 = rng.gen::<u32>();
        let val2 = rng.gen::<u32>();

        tester.write(
            RV32_REGISTER_AS as usize,
            ptr1,
            val1.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write(
            RV32_REGISTER_AS as usize,
            ptr2,
            val2.to_le_bytes().map(F::from_canonical_u8),
        );

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(
                opcode.global_opcode(),
                [rd, ptr1, ptr2, RV32_REGISTER_AS as usize, 0],
            ),
        );
    }

    #[test]
    fn rand_mul_tracegen_test() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default().with_range_tuple_checker(
            RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES),
        );

        let mut harness = create_test_harness(&tester);
        let num_ops = 100;
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, MulOpcode::MUL);
        }

        type Record<'a> = (
            &'a mut Rv32MultAdapterRecord,
            &'a mut MultiplicationCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record<'_>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32MultAdapterStep>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
