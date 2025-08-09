use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_rv32im_circuit::{
    adapters::{Rv32MultAdapterCols, Rv32MultAdapterRecord},
    DivRemCoreCols, DivRemCoreRecord,
};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use crate::{
    extensions::rv32im::cuda::divrem_cuda,
    get_empty_air_proving_ctx,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
        var_range::VariableRangeCheckerChipGPU,
    },
    UInt2,
};

#[derive(new)]
pub struct Rv32DivRemChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32DivRemChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32MultAdapterRecord,
            DivRemCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = DivRemCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32MultAdapterCols::<F>::width();
        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);
        unsafe {
            divrem_cuda::tracegen(
                d_trace.buffer(),
                padded_height as u32,
                trace_width as u32,
                &d_records,
                height as u32,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
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
mod test {
    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, RANGE_TUPLE_CHECKER_BUS},
        EmptyAdapterCoreLayout,
    };
    use openvm_circuit_primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChip,
        range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip},
    };
    use openvm_instructions::{
        instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        adapters::{Rv32MultAdapterAir, Rv32MultAdapterExecutor, Rv32MultAdapterFiller},
        DivRemCoreAir, DivRemFiller, Rv32DivRemAir, Rv32DivRemChip, Rv32DivRemExecutor,
    };
    use openvm_rv32im_transpiler::DivRemOpcode::{self, *};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness};

    // the max number of limbs we currently support DIVREM for is 32 (i.e. for U256s)
    const MAX_NUM_LIMBS: u32 = 32;
    const TUPLE_CHECKER_SIZES: [u32; 2] = [
        (1 << RV32_CELL_BITS) as u32,
        (MAX_NUM_LIMBS * (1 << RV32_CELL_BITS)),
    ];
    const MAX_INS_CAPACITY: usize = 128;

    type Harness = GpuTestChipHarness<
        F,
        Rv32DivRemExecutor,
        Rv32DivRemAir,
        Rv32DivRemChipGpu,
        Rv32DivRemChip<F>,
    >;

    fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
        // getting bus's from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        let range_tuple_bus =
            RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

        // creating dummy chips for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let dummy_range_tuple_chip =
            SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

        let air = Rv32DivRemAir::new(
            Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            DivRemCoreAir::new(bitwise_bus, range_tuple_bus, DivRemOpcode::CLASS_OFFSET),
        );
        let executor = Rv32DivRemExecutor::new(Rv32MultAdapterExecutor, DivRemOpcode::CLASS_OFFSET);
        let cpu_chip = Rv32DivRemChip::<F>::new(
            DivRemFiller::new(
                Rv32MultAdapterFiller,
                dummy_bitwise_chip,
                dummy_range_tuple_chip,
                DivRemOpcode::CLASS_OFFSET,
            ),
            tester.dummy_memory_helper(),
        );

        let gpu_chip = Rv32DivRemChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            tester.range_tuple_checker(),
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
        opcode: DivRemOpcode,
        b: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
        c: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    ) {
        let b: [u8; RV32_REGISTER_NUM_LIMBS] = b.unwrap_or_else(|| rng.gen());
        let c: [u8; RV32_REGISTER_NUM_LIMBS] = c.unwrap_or_else(|| rng.gen());

        let rs1 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rs2 = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rd = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u8));
        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u8));

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 0]),
        );
    }

    #[test_case(DIV, 100)]
    #[test_case(DIVU, 100)]
    #[test_case(REM, 100)]
    #[test_case(REMU, 100)]
    fn test_divrem_tracegen(opcode: DivRemOpcode, num_ops: usize) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default()
            .with_bitwise_op_lookup(default_bitwise_lookup_bus())
            .with_range_tuple_checker(RangeTupleCheckerBus::new(
                RANGE_TUPLE_CHECKER_BUS,
                TUPLE_CHECKER_SIZES,
            ));

        let mut harness = create_test_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut harness, &mut rng, opcode, None, None);
        }
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode,
            Some([98, 188, 163, 127]),
            Some([0, 0, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode,
            Some([98, 188, 163, 229]),
            Some([0, 0, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode,
            Some([0, 0, 0, 128]),
            Some([0, 1, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode,
            Some([0, 0, 0, 127]),
            Some([0, 1, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode,
            Some([0, 0, 0, 0]),
            Some([0, 0, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode,
            Some([0, 0, 0, 0]),
            Some([0, 0, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode,
            Some([0, 0, 0, 128]),
            Some([255, 255, 255, 255]),
        );

        type Record<'a> = (
            &'a mut Rv32MultAdapterRecord,
            &'a mut DivRemCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        );

        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv32MultAdapterExecutor>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
