use std::{array, borrow::BorrowMut};

use afs_stark_backend::{config::StarkGenericConfig, rap::AnyRap};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Domain;

use super::{
    columns::{UintArithmeticAuxCols, UintArithmeticCols, UintArithmeticIoCols},
    UintArithmeticChip, UintArithmeticRecord, WriteRecord,
};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode},
    memory::offline_checker::MemoryWriteAuxCols,
    uint_multiplication::MemoryData,
};

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> MachineChip<F>
    for UintArithmeticChip<F, NUM_LIMBS, LIMB_BITS>
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

        let width = self.trace_width();
        let height = self.data.len();
        let padded_height = height.next_power_of_two();
        let mut rows = vec![F::zero(); width * padded_height];

        for (row, operation) in rows.chunks_mut(width).zip(self.data) {
            let UintArithmeticRecord::<F, NUM_LIMBS, LIMB_BITS> {
                from_state,
                instruction,
                x_ptr_read,
                y_ptr_read,
                z_ptr_read,
                x_read,
                y_read,
                z_write,
                cmp_buffer,
            } = operation;

            let row: &mut UintArithmeticCols<F, NUM_LIMBS, LIMB_BITS> = row.borrow_mut();

            row.io = UintArithmeticIoCols {
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
                    WriteRecord::Uint(z) => MemoryData {
                        data: z.data,
                        address: z.pointer,
                        ptr_to_address: z_ptr_read.pointer,
                    },
                    WriteRecord::Short(z) => MemoryData {
                        data: array::from_fn(|i| cmp_buffer[i]),
                        address: z.pointer,
                        ptr_to_address: z_ptr_read.pointer,
                    },
                },
                cmp_result: match &z_write {
                    WriteRecord::Uint(_) => F::zero(),
                    WriteRecord::Short(z) => z.data[0],
                },
                ptr_as: instruction.d,
                address_as: instruction.e,
            };

            row.aux = UintArithmeticAuxCols {
                is_valid: F::one(),
                is_x_neg: F::zero(), // TODO: SLT impl
                is_y_neg: F::zero(),
                opcode_add_flag: F::from_bool(instruction.opcode == Opcode::ADD256),
                opcode_sub_flag: F::from_bool(instruction.opcode == Opcode::SUB256),
                opcode_lt_flag: F::from_bool(instruction.opcode == Opcode::LT256),
                opcode_eq_flag: F::from_bool(instruction.opcode == Opcode::EQ256),
                opcode_xor_flag: F::from_bool(instruction.opcode == Opcode::XOR256),
                opcode_and_flag: F::from_bool(instruction.opcode == Opcode::AND256),
                opcode_or_flag: F::from_bool(instruction.opcode == Opcode::OR256),
                opcode_slt_flag: F::from_bool(instruction.opcode == Opcode::SLT256),
                read_ptr_aux_cols: [z_ptr_read, x_ptr_read, y_ptr_read]
                    .map(|read| aux_cols_factory.make_read_aux_cols(read.clone())),
                read_x_aux_cols: aux_cols_factory.make_read_aux_cols(x_read.clone()),
                read_y_aux_cols: aux_cols_factory.make_read_aux_cols(y_read.clone()),
                write_z_aux_cols: match &z_write {
                    WriteRecord::Uint(z) => aux_cols_factory.make_write_aux_cols(z.clone()),
                    WriteRecord::Short(_) => MemoryWriteAuxCols::disabled(),
                },
                write_cmp_aux_cols: match &z_write {
                    WriteRecord::Uint(_) => MemoryWriteAuxCols::disabled(),
                    WriteRecord::Short(z) => aux_cols_factory.make_write_aux_cols(z.clone()),
                },
            };
        }
        RowMajorMatrix::new(rows, width)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        UintArithmeticCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
