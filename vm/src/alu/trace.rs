use std::{array, borrow::BorrowMut};

use afs_stark_backend::{
    config::StarkGenericConfig,
    rap::{get_air_name, AnyRap},
};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Domain;

use super::{
    columns::{ArithmeticLogicAuxCols, ArithmeticLogicCols, ArithmeticLogicIoCols},
    ArithmeticLogicChip, ArithmeticLogicRecord, WriteRecord,
};
use crate::{
    arch::{
        instructions::{U256Opcode, UsizeOpcode},
        MachineChip,
    },
    memory::offline_checker::MemoryWriteAuxCols,
    uint_multiplication::MemoryData,
};

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> MachineChip<F>
    for ArithmeticLogicChip<F, NUM_LIMBS, LIMB_BITS>
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

        let width = self.trace_width();
        let height = self.data.len();
        let padded_height = height.next_power_of_two();
        let mut rows = vec![F::zero(); width * padded_height];

        for (row, operation) in rows.chunks_mut(width).zip(self.data) {
            let ArithmeticLogicRecord::<F, NUM_LIMBS, LIMB_BITS> {
                from_state,
                instruction,
                x_ptr_read,
                y_ptr_read,
                z_ptr_read,
                x_read,
                y_read,
                z_write,
                x_sign,
                y_sign,
                cmp_buffer,
            } = operation;

            let row: &mut ArithmeticLogicCols<F, NUM_LIMBS, LIMB_BITS> = row.borrow_mut();

            row.io = ArithmeticLogicIoCols {
                from_state: from_state.map(F::from_canonical_usize),
                x: MemoryData::<F, NUM_LIMBS, LIMB_BITS> {
                    data: x_read.data,
                    address: x_read.pointer,
                    ptr_to_address: x_ptr_read.pointer,
                },
                y: MemoryData::<F, NUM_LIMBS, LIMB_BITS> {
                    data: y_read.data,
                    address: y_read.pointer,
                    ptr_to_address: y_ptr_read.pointer,
                },
                z: match &z_write {
                    WriteRecord::Long(z) => MemoryData {
                        data: z.data,
                        address: z.pointer,
                        ptr_to_address: z_ptr_read.pointer,
                    },
                    WriteRecord::Bool(z) => MemoryData {
                        data: array::from_fn(|i| cmp_buffer[i]),
                        address: z.pointer,
                        ptr_to_address: z_ptr_read.pointer,
                    },
                },
                cmp_result: match &z_write {
                    WriteRecord::Long(_) => F::zero(),
                    WriteRecord::Bool(z) => z.data[0],
                },
                ptr_as: instruction.d,
                address_as: instruction.e,
            };

            let opcode = U256Opcode::from_usize(instruction.opcode);
            row.aux = ArithmeticLogicAuxCols {
                is_valid: F::one(),
                x_sign,
                y_sign,
                opcode_add_flag: F::from_bool(opcode == U256Opcode::ADD),
                opcode_sub_flag: F::from_bool(opcode == U256Opcode::SUB),
                opcode_sltu_flag: F::from_bool(opcode == U256Opcode::LT),
                opcode_eq_flag: F::from_bool(opcode == U256Opcode::EQ),
                opcode_xor_flag: F::from_bool(opcode == U256Opcode::XOR),
                opcode_and_flag: F::from_bool(opcode == U256Opcode::AND),
                opcode_or_flag: F::from_bool(opcode == U256Opcode::OR),
                opcode_slt_flag: F::from_bool(opcode == U256Opcode::SLT),
                read_ptr_aux_cols: [z_ptr_read, x_ptr_read, y_ptr_read]
                    .map(|read| aux_cols_factory.make_read_aux_cols(read.clone())),
                read_x_aux_cols: aux_cols_factory.make_read_aux_cols(x_read.clone()),
                read_y_aux_cols: aux_cols_factory.make_read_aux_cols(y_read.clone()),
                write_z_aux_cols: match &z_write {
                    WriteRecord::Long(z) => aux_cols_factory.make_write_aux_cols(z.clone()),
                    WriteRecord::Bool(_) => MemoryWriteAuxCols::disabled(),
                },
                write_cmp_aux_cols: match &z_write {
                    WriteRecord::Long(_) => MemoryWriteAuxCols::disabled(),
                    WriteRecord::Bool(z) => aux_cols_factory.make_write_aux_cols(z.clone()),
                },
            };
        }
        RowMajorMatrix::new(rows, width)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Arc<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        ArithmeticLogicCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
