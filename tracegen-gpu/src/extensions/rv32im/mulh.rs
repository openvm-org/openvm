use std::{mem::size_of, sync::Arc};

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::adapters::{
    Rv32MultAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use super::cuda::mulh::tracegen;
use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
        var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};

type MulHCoreRecord =
    openvm_rv32im_circuit::MulHCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
type Rv32MulHRecord = (Rv32MultAdapterRecord, MulHCoreRecord);

pub struct Rv32MulHChipGpu<'a> {
    pub air: openvm_rv32im_circuit::Rv32MulHAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<'a> Rv32MulHChipGpu<'a> {
    pub fn new(
        air: openvm_rv32im_circuit::Rv32MulHAir,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
        range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
        arena: Option<&'a DenseRecordArena>,
    ) -> Self {
        Self {
            air,
            range_checker,
            bitwise_lookup,
            range_tuple_checker,
            arena,
        }
    }

    fn get_arena(&self) -> &DenseRecordArena {
        self.arena.as_ref().unwrap()
    }
}

impl ChipUsageGetter for Rv32MulHChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<Rv32MulHRecord>();
        let arena = self.get_arena();
        let records_len =
            arena.records_buffer.get_ref()[..arena.records_buffer.position() as usize].len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Rv32MulHChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let arena = self.get_arena();
        let d_records = arena.records_buffer.get_ref()[..arena.records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());

        // Get range tuple sizes from the range tuple checker
        let range_tuple_size0 = self.range_tuple_checker.air.bus.sizes[0];
        let range_tuple_size1 = self.range_tuple_checker.air.bus.sizes[1];

        unsafe {
            tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                range_tuple_size0,
                range_tuple_size1,
            )
            .unwrap();
        }
        trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openvm_circuit::arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS},
        EmptyAdapterCoreLayout, MatrixRecordArena, NewVmChipWrapper, VmAirWrapper,
    };
    use openvm_circuit::utils::generate_long_number;
    use openvm_circuit_primitives::{
        bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
        range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
    };
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_rv32im_circuit::adapters::{Rv32MultAdapterAir, Rv32MultAdapterStep};
    use openvm_rv32im_transpiler::MulHOpcode;
    use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::Rng;

    use crate::testing::GpuChipTestBuilder;

    const MAX_INS_CAPACITY: usize = 128;

    type DenseChip<F> = NewVmChipWrapper<
        F,
        openvm_rv32im_circuit::Rv32MulHAir,
        openvm_rv32im_circuit::Rv32MulHStep,
        DenseRecordArena,
    >;
    type SparseChip<F> = NewVmChipWrapper<
        F,
        openvm_rv32im_circuit::Rv32MulHAir,
        openvm_rv32im_circuit::Rv32MulHStep,
        MatrixRecordArena<F>,
    >;

    #[test]
    fn rand_mulh_tracegen_test() {
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);

        // Range tuple sizes for MulH (8-bit limbs)
        let sizes = [
            1u32 << RV32_CELL_BITS,
            ((1u32 << RV32_CELL_BITS) * 2 * RV32_REGISTER_NUM_LIMBS as u32),
        ];
        let range_tuple_bus = RangeTupleCheckerBus::new(0, sizes);

        let mut tester = GpuChipTestBuilder::default()
            .with_variable_range_checker()
            .with_bitwise_op_lookup(bitwise_bus)
            .with_range_tuple_checker(range_tuple_bus);
        let mut rng = create_seeded_rng();
        let cpu_range_tuple_chip = SharedRangeTupleCheckerChip::new(range_tuple_bus.clone());
        let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

        let mut dense_chip = {
            let mut chip = DenseChip::<F>::new(
                VmAirWrapper::new(
                    Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                    openvm_rv32im_circuit::MulHCoreAir::new(
                        cpu_bitwise_chip.bus(),
                        *cpu_range_tuple_chip.bus(),
                    ),
                ),
                openvm_rv32im_circuit::MulHStep::new(
                    Rv32MultAdapterStep::new(),
                    cpu_bitwise_chip.clone(),
                    cpu_range_tuple_chip.clone(),
                ),
                tester.cpu_memory_helper(),
            );
            chip.set_trace_buffer_height(MAX_INS_CAPACITY);
            chip
        };

        let mut cpu_chip = {
            let mut chip = SparseChip::<F>::new(
                VmAirWrapper::new(
                    Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
                    openvm_rv32im_circuit::MulHCoreAir::new(
                        cpu_bitwise_chip.bus(),
                        *cpu_range_tuple_chip.bus(),
                    ),
                ),
                openvm_rv32im_circuit::MulHStep::new(
                    Rv32MultAdapterStep::new(),
                    cpu_bitwise_chip,
                    cpu_range_tuple_chip,
                ),
                tester.cpu_memory_helper(),
            );
            chip.set_trace_buffer_height(MAX_INS_CAPACITY);
            chip
        };
        let mut gpu_chip = Rv32MulHChipGpu::new(
            dense_chip.air.clone(),
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            tester.range_tuple_checker(),
            None,
        );

        // Test all three MulH opcodes
        let opcodes = [MulHOpcode::MULH, MulHOpcode::MULHSU, MulHOpcode::MULHU];

        for _ in 0..100 {
            let opcode = opcodes[rng.gen_range(0..opcodes.len())];
            let b = generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);
            let c = generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);

            let rs1 = gen_pointer(&mut rng, 4);
            let rs2 = gen_pointer(&mut rng, 4);
            let rd = gen_pointer(&mut rng, 4);

            tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
            tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));

            tester.execute(
                &mut dense_chip,
                &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 0]),
            );
        }

        type Record<'a> = (&'a mut Rv32MultAdapterRecord, &'a mut MulHCoreRecord);
        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut cpu_chip.arena,
                EmptyAdapterCoreLayout::<F, Rv32MultAdapterStep>::new(),
            );
        gpu_chip.arena = Some(&dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, cpu_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
