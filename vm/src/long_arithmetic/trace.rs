use afs_stark_backend::{config::StarkGenericConfig, rap::AnyRap};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Domain;

use super::{
    columns::{LongArithmeticAuxCols, LongArithmeticCols, LongArithmeticIoCols, MemoryData},
    num_limbs, LongArithmeticChip, WriteRecord,
};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode},
    memory::offline_checker::columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, F: PrimeField32> MachineChip<F>
    for LongArithmeticChip<ARG_SIZE, LIMB_SIZE, F>
{
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
        LongArithmeticCols::<ARG_SIZE, LIMB_SIZE, F>::get_width(&self.air)
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let memory_chip = self.memory_chip.borrow();
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();
        let rows = self
            .data
            .iter()
            .map(|operation| {
                {
                    LongArithmeticCols {
                        io: LongArithmeticIoCols {
                            from_state: operation.record.from_state.map(F::from_canonical_usize),
                            x: MemoryData::<ARG_SIZE, LIMB_SIZE, F> {
                                data: operation.record.x_read.data.to_vec(),
                                address_space: operation.record.x_read.address_space,
                                address: operation.record.x_read.pointer,
                            },
                            y: MemoryData {
                                data: operation.record.y_read.data.to_vec(),
                                address_space: operation.record.y_read.address_space,
                                address: operation.record.y_read.pointer,
                            },
                            z: match &operation.record.z_write {
                                WriteRecord::Long(z) => MemoryData {
                                    data: z.data.to_vec(),
                                    address_space: z.address_space,
                                    address: z.pointer,
                                },
                                WriteRecord::Short(z) => MemoryData {
                                    data: operation
                                        .result
                                        .iter()
                                        .cloned()
                                        .chain(std::iter::repeat(F::zero()))
                                        .take(num_limbs)
                                        .collect(),
                                    address_space: z.address_space,
                                    address: z.pointer,
                                },
                            },
                            cmp_result: match &operation.record.z_write {
                                WriteRecord::Long(_) => F::zero(),
                                WriteRecord::Short(z) => z.data[0],
                            },
                        },
                        aux: LongArithmeticAuxCols {
                            is_valid: F::one(),
                            opcode_add_flag: F::from_bool(
                                operation.record.instruction.opcode == Opcode::ADD256,
                            ),
                            opcode_sub_flag: F::from_bool(
                                operation.record.instruction.opcode == Opcode::SUB256,
                            ),
                            opcode_lt_flag: F::from_bool(
                                operation.record.instruction.opcode == Opcode::LT256,
                            ),
                            opcode_eq_flag: F::from_bool(
                                operation.record.instruction.opcode == Opcode::EQ256,
                            ),
                            buffer: operation.buffer.clone(),
                            read_x_aux_cols: memory_chip
                                .make_read_aux_cols(operation.record.x_read.clone()),
                            read_y_aux_cols: memory_chip
                                .make_read_aux_cols(operation.record.y_read.clone()),
                            write_z_aux_cols: match &operation.record.z_write {
                                WriteRecord::Long(z) => memory_chip.make_write_aux_cols(z.clone()),
                                WriteRecord::Short(_) => memory_chip.make_disabled_write_aux_cols(),
                            },
                            write_cmp_aux_cols: match &operation.record.z_write {
                                WriteRecord::Long(_) => memory_chip.make_disabled_write_aux_cols(),
                                WriteRecord::Short(z) => memory_chip.make_write_aux_cols(z.clone()),
                            },
                        },
                    }
                }
                .flatten()
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();

        let blank_row = LongArithmeticCols::<ARG_SIZE, LIMB_SIZE, F> {
            io: Default::default(),
            aux: LongArithmeticAuxCols {
                is_valid: Default::default(),
                opcode_add_flag: Default::default(),
                opcode_sub_flag: Default::default(),
                opcode_lt_flag: Default::default(),
                opcode_eq_flag: Default::default(),
                buffer: vec![Default::default(); num_limbs],
                read_x_aux_cols: MemoryReadAuxCols::disabled(self.air.mem_oc),
                read_y_aux_cols: MemoryReadAuxCols::disabled(self.air.mem_oc),
                write_z_aux_cols: MemoryWriteAuxCols::disabled(self.air.mem_oc),
                write_cmp_aux_cols: MemoryWriteAuxCols::disabled(self.air.mem_oc),
            },
        }
        .flatten();
        let width = blank_row.len();

        let mut padded_rows = rows;

        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
