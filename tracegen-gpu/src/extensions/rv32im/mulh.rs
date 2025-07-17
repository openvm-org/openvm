use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{
        Rv32MultAdapterCols, Rv32MultAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    MulHCoreCols, MulHCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use super::cuda::mulh::tracegen;
use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
        var_range::VariableRangeCheckerChipGPU,
    },
    testing::get_empty_air_proving_ctx,
    UInt2,
};

#[derive(new)]
pub struct Rv32MulHChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32MulHChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32MultAdapterRecord,
            MulHCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = MulHCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32MultAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);

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
                &self.range_tuple_checker.count,
                tuple_checker_sizes,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::{
        arch::{
            testing::{memory::gen_pointer, RANGE_TUPLE_CHECKER_BUS},
            EmptyAdapterCoreLayout,
        },
        utils::generate_long_number,
    };
    use openvm_circuit_primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChip,
        range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip},
    };
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_rv32im_circuit::{
        adapters::{Rv32MultAdapterAir, Rv32MultAdapterFiller, Rv32MultAdapterStep},
        MulHCoreAir, MulHFiller, Rv32MulHAir, Rv32MulHChip, Rv32MulHStep,
    };
    use openvm_rv32im_transpiler::MulHOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::rngs::StdRng;
    use test_case::test_case;

    use super::*;
    use crate::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
        GpuTestChipHarness,
    };

    const MAX_INS_CAPACITY: usize = 128;
    const TUPLE_CHECKER_SIZES: [u32; 2] = [
        (1 << RV32_CELL_BITS) as u32,
        (8 * (1 << RV32_CELL_BITS)) as u32,
    ];
    type Harness =
        GpuTestChipHarness<F, Rv32MulHStep, Rv32MulHAir, Rv32MulHChipGpu, Rv32MulHChip<F>>;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus's from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        let range_tuple_bus =
            RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

        // creating a dummy chips for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let dummy_range_tuple_chip =
            SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

        let air = Rv32MulHAir::new(
            Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            MulHCoreAir::new(bitwise_bus, range_tuple_bus),
        );
        let executor = Rv32MulHStep::new(Rv32MultAdapterStep, MulHOpcode::CLASS_OFFSET);
        let cpu_chip = Rv32MulHChip::<F>::new(
            MulHFiller::new(
                Rv32MultAdapterFiller,
                dummy_bitwise_chip,
                dummy_range_tuple_chip,
            ),
            tester.cpu_memory_helper(),
        );
        let gpu_chip = Rv32MulHChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            tester.range_tuple_checker(),
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        harness: &mut Harness,
        rng: &mut StdRng,
        opcode: MulHOpcode,
    ) {
        let b = generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(rng);
        let c = generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(rng);

        let rs1 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rs2 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rd = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 0]),
        );
    }

    #[test_case(MulHOpcode::MULH, 100)]
    #[test_case(MulHOpcode::MULHSU, 100)]
    #[test_case(MulHOpcode::MULHU, 100)]
    fn rand_mulh_tracegen_test(opcode: MulHOpcode, num_ops: usize) {
        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker(default_var_range_checker_bus())
            .with_bitwise_op_lookup(default_bitwise_lookup_bus())
            .with_range_tuple_checker(RangeTupleCheckerBus::new(
                RANGE_TUPLE_CHECKER_BUS,
                TUPLE_CHECKER_SIZES,
            ));
        let mut rng = create_seeded_rng();

        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode);
        }

        type Record<'a> = (
            &'a mut Rv32MultAdapterRecord,
            &'a mut MulHCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        );

        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32MultAdapterStep>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
