use std::array::from_fn;

use afs_stark_backend::{config::StarkGenericConfig, rap::AnyRap};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Domain;

use super::{
    columns::{MemoryData, UintArithmeticAuxCols, UintArithmeticCols, UintArithmeticIoCols},
    num_limbs, UintArithmeticChip, WriteRecord,
};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode},
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, F: PrimeField32> MachineChip<F>
    for UintArithmeticChip<ARG_SIZE, LIMB_SIZE, F>
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();
        let rows = self
            .data
            .iter()
            .map(|operation| {
                {
                    let super::UintArithmeticRecord::<ARG_SIZE, LIMB_SIZE, F> {
                        from_state,
                        instruction,
                        x_ptr_read,
                        y_ptr_read,
                        z_ptr_read,
                        x_read,
                        y_read,
                        z_write,
                        result,
                        buffer,
                    } = operation;

                    UintArithmeticCols {
                        io: UintArithmeticIoCols {
                            from_state: from_state.map(F::from_canonical_usize),
                            x: MemoryData::<ARG_SIZE, LIMB_SIZE, F> {
                                data: x_read.data.to_vec(),
                                address_space: x_read.address_space,
                                address: x_read.pointer,
                                ptr: x_ptr_read.pointer,
                            },
                            y: MemoryData {
                                data: y_read.data.to_vec(),
                                address_space: y_read.address_space,
                                address: y_read.pointer,
                                ptr: y_ptr_read.pointer,
                            },
                            z: match &z_write {
                                WriteRecord::Uint(z) => MemoryData {
                                    data: z.data.to_vec(),
                                    address_space: z.address_space,
                                    address: z.pointer,
                                    ptr: z_ptr_read.pointer,
                                },
                                WriteRecord::Short(z) => MemoryData {
                                    data: result
                                        .iter()
                                        .cloned()
                                        .chain(std::iter::repeat(F::zero()))
                                        .take(num_limbs)
                                        .collect(),
                                    address_space: z.address_space,
                                    address: z.pointer,
                                    ptr: z_ptr_read.pointer,
                                },
                            },
                            d: instruction.d,
                            cmp_result: match &z_write {
                                WriteRecord::Uint(_) => F::zero(),
                                WriteRecord::Short(z) => z.data[0],
                            },
                        },
                        aux: UintArithmeticAuxCols {
                            is_valid: F::one(),
                            opcode_add_flag: F::from_bool(instruction.opcode == Opcode::ADD256),
                            opcode_sub_flag: F::from_bool(instruction.opcode == Opcode::SUB256),
                            opcode_lt_flag: F::from_bool(instruction.opcode == Opcode::LT256),
                            opcode_eq_flag: F::from_bool(instruction.opcode == Opcode::EQ256),
                            buffer: buffer.clone(),
                            read_ptr_aux_cols: [z_ptr_read, x_ptr_read, y_ptr_read]
                                .map(|read| aux_cols_factory.make_read_aux_cols(read.clone())),
                            read_x_aux_cols: aux_cols_factory.make_read_aux_cols(x_read.clone()),
                            read_y_aux_cols: aux_cols_factory.make_read_aux_cols(y_read.clone()),
                            write_z_aux_cols: match &z_write {
                                WriteRecord::Uint(z) => {
                                    aux_cols_factory.make_write_aux_cols(z.clone())
                                }
                                WriteRecord::Short(_) => MemoryWriteAuxCols::disabled(),
                            },
                            write_cmp_aux_cols: match &z_write {
                                WriteRecord::Uint(_) => MemoryWriteAuxCols::disabled(),
                                WriteRecord::Short(z) => {
                                    aux_cols_factory.make_write_aux_cols(z.clone())
                                }
                            },
                        },
                    }
                }
                .flatten()
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();

        let blank_row = UintArithmeticCols::<ARG_SIZE, LIMB_SIZE, F> {
            io: Default::default(),
            aux: UintArithmeticAuxCols {
                is_valid: Default::default(),
                opcode_add_flag: Default::default(),
                opcode_sub_flag: Default::default(),
                opcode_lt_flag: Default::default(),
                opcode_eq_flag: Default::default(),
                buffer: vec![Default::default(); num_limbs],
                read_ptr_aux_cols: from_fn(|_| MemoryReadAuxCols::disabled()),
                read_x_aux_cols: MemoryReadAuxCols::disabled(),
                read_y_aux_cols: MemoryReadAuxCols::disabled(),
                write_z_aux_cols: MemoryWriteAuxCols::disabled(),
                write_cmp_aux_cols: MemoryWriteAuxCols::disabled(),
            },
        }
        .flatten();
        let width = blank_row.len();

        let mut padded_rows = rows;

        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
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
        UintArithmeticCols::<ARG_SIZE, LIMB_SIZE, F>::width()
    }
}
