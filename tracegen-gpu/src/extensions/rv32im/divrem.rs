use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_rv32im_circuit::{adapters::Rv32MultAdapterRecord, DivRemCoreRecords, Rv32DivRemAir};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    extensions::rv32im::cuda::divrem_cuda,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
        var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip, UInt2,
};

#[derive(new)]
pub struct Rv32DivRemChipGpu {
    pub air: Rv32DivRemAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub pointer_max_bits: usize,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for Rv32DivRemChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(
            Rv32MultAdapterRecord,
            DivRemCoreRecords<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Rv32DivRemChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.allocated().to_device().unwrap();
        let height = self.current_trace_height();
        let padded_height = next_power_of_two_or_zero(height);
        let trace = DeviceMatrix::<F>::with_capacity(padded_height, self.trace_width());

        let tuple_checker_sizes = self.range_tuple_checker.air.bus.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);
        unsafe {
            divrem_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                self.trace_width() as u32,
                &d_records,
                height as u32,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                &self.range_tuple_checker.count,
                tuple_checker_sizes,
            )
            .unwrap();
        }

        trace
    }
}

#[cfg(test)]
mod test {
    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS, RANGE_TUPLE_CHECKER_BUS},
        EmptyAdapterCoreLayout, NewVmChipWrapper,
    };
    use openvm_circuit_primitives::{
        bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
        range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
    };
    use openvm_instructions::{
        instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode,
    };
    use openvm_rv32im_circuit::{
        adapters::{Rv32MultAdapterAir, Rv32MultAdapterStep},
        DivRemCoreAir, DivRemStep, Rv32DivRemChip, Rv32DivRemStep,
    };
    use openvm_rv32im_transpiler::DivRemOpcode::{self, *};
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const TUPLE_CHECKER_SIZES: [u32; 2] = [
        (1 << RV32_CELL_BITS) as u32,
        (32 * (1 << RV32_CELL_BITS)) as u32,
    ];
    const MAX_INS_CAPACITY: usize = 128;
    type Rv32DivRemDenseChip<F> =
        NewVmChipWrapper<F, Rv32DivRemAir, Rv32DivRemStep, DenseRecordArena>;

    fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> Rv32DivRemDenseChip<F> {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let range_tuple_bus =
            RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let range_tuple_chip = SharedRangeTupleCheckerChip::new(range_tuple_bus);

        let mut chip = Rv32DivRemDenseChip::<F>::new(
            Rv32DivRemAir::new(
                Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                DivRemCoreAir::new(bitwise_bus, range_tuple_bus, DivRemOpcode::CLASS_OFFSET),
            ),
            DivRemStep::new(
                Rv32MultAdapterStep::new(),
                bitwise_chip.clone(),
                range_tuple_chip.clone(),
                DivRemOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_test_sparse_chip(
        tester: &GpuChipTestBuilder,
        dense_chip: &Rv32DivRemDenseChip<F>,
    ) -> Rv32DivRemChip<F> {
        let mut chip = Rv32DivRemChip::<F>::new(
            dense_chip.air,
            DivRemStep::new(
                Rv32MultAdapterStep::new(),
                dense_chip.step.bitwise_lookup_chip.clone(),
                dense_chip.step.range_tuple_chip.clone(),
                DivRemOpcode::CLASS_OFFSET,
            ),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[allow(clippy::too_many_arguments)]
    fn set_and_execute(
        tester: &mut GpuChipTestBuilder,
        chip: &mut Rv32DivRemDenseChip<F>,
        rng: &mut StdRng,
        opcode: DivRemOpcode,
        b: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
        c: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    ) {
        let b: [u8; RV32_REGISTER_NUM_LIMBS] = b.unwrap_or_else(|| rng.gen());
        let c: [u8; RV32_REGISTER_NUM_LIMBS] = c.unwrap_or_else(|| rng.gen());

        let rs1 = gen_pointer(rng, 4);
        let rs2 = gen_pointer(rng, 4);
        let rd = gen_pointer(rng, 4);

        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u8));
        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u8));

        tester.execute(
            chip,
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
            .with_variable_range_checker()
            .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS))
            .with_range_tuple_checker(RangeTupleCheckerBus::new(
                RANGE_TUPLE_CHECKER_BUS,
                TUPLE_CHECKER_SIZES,
            ));

        // CPU execution
        let mut dense_chip = create_test_dense_chip(&tester);

        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut dense_chip, &mut rng, opcode, None, None);
        }
        set_and_execute(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode,
            Some([98, 188, 163, 127]),
            Some([0, 0, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode,
            Some([98, 188, 163, 229]),
            Some([0, 0, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode,
            Some([0, 0, 0, 128]),
            Some([0, 1, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode,
            Some([0, 0, 0, 127]),
            Some([0, 1, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode,
            Some([0, 0, 0, 0]),
            Some([0, 0, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode,
            Some([0, 0, 0, 0]),
            Some([0, 0, 0, 0]),
        );
        set_and_execute(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode,
            Some([0, 0, 0, 128]),
            Some([255, 255, 255, 255]),
        );

        let mut sparse_chip = create_test_sparse_chip(&tester, &dense_chip);

        type Record<'a> = (
            &'a mut Rv32MultAdapterRecord,
            &'a mut DivRemCoreRecords<RV32_REGISTER_NUM_LIMBS>,
        );

        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut sparse_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32MultAdapterStep>::new(),
            );

        // GPU tracegen
        let gpu_chip = Rv32DivRemChipGpu::new(
            dense_chip.air,
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            tester.range_tuple_checker(),
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
