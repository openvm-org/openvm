use afs_primitives::sub_chip::LocalTraceInstructions;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{
        MemoryData, ModularArithmeticAuxCols, ModularArithmeticCols, ModularArithmeticIoCols,
    },
    ModularArithmeticChip, ModularArithmeticRecord,
};
use crate::{
    arch::chips::MachineChip,
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

impl<F: PrimeField32> MachineChip<F> for ModularArithmeticChip<F> {
    fn air<SC: p3_uni_stark::StarkGenericConfig>(
        &self,
    ) -> Box<dyn afs_stark_backend::rap::AnyRap<SC>>
    where
        p3_uni_stark::Domain<SC>: p3_commit::PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        ModularArithmeticCols::<F>::width(&self.air)
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let memory_chip = self.memory_chip.borrow();

        let rows = self
            .data
            .iter()
            .map(|record| {
                let ModularArithmeticRecord {
                    from_state,
                    instruction: _instruction, // FIXME: use opcode
                    x_read,
                    y_read,
                    z_write,
                    x_address_read,
                    y_address_read,
                    z_address_read,
                } = record;
                let io = ModularArithmeticIoCols {
                    from_state: from_state.map(F::from_canonical_usize),
                    x: MemoryData {
                        data: x_read.data.to_vec(),
                        address_space: x_read.address_space,
                        address: x_read.pointer,
                    },
                    y: MemoryData {
                        data: y_read.data.to_vec(),
                        address_space: y_read.address_space,
                        address: y_read.pointer,
                    },
                    z: MemoryData {
                        data: z_write.data.to_vec(),
                        address_space: z_write.address_space,
                        address: z_write.pointer,
                    },
                    x_address: MemoryData {
                        data: x_address_read.data.to_vec(),
                        address_space: x_address_read.address_space,
                        address: x_address_read.pointer,
                    },
                    y_address: MemoryData {
                        data: y_address_read.data.to_vec(),
                        address_space: y_address_read.address_space,
                        address: y_address_read.pointer,
                    },
                    z_address: MemoryData {
                        data: z_address_read.data.to_vec(),
                        address_space: z_address_read.address_space,
                        address: z_address_read.pointer,
                    },
                };
                let x_limbs = x_read
                    .data
                    .iter()
                    .map(|x| x.as_canonical_u32())
                    .collect::<Vec<_>>();
                let x_biguint = super::limbs_to_biguint(&x_limbs);
                let y_limbs = y_read
                    .data
                    .iter()
                    .map(|x| x.as_canonical_u32())
                    .collect::<Vec<_>>();
                let y_biguint = super::limbs_to_biguint(&y_limbs);
                let primitive_row = self.air.air.generate_trace_row((
                    x_biguint,
                    y_biguint,
                    self.range_checker_chip.clone(),
                ));

                let aux = ModularArithmeticAuxCols {
                    is_valid: F::one(),
                    read_x_aux_cols: memory_chip.make_read_aux_cols(x_read.clone()),
                    read_y_aux_cols: memory_chip.make_read_aux_cols(y_read.clone()),
                    write_z_aux_cols: memory_chip.make_write_aux_cols(z_write.clone()),
                    x_address_aux_cols: memory_chip.make_read_aux_cols(x_address_read.clone()),
                    y_address_aux_cols: memory_chip.make_read_aux_cols(y_address_read.clone()),
                    z_address_aux_cols: memory_chip.make_read_aux_cols(z_address_read.clone()),
                    carries: primitive_row.carries,
                    q: primitive_row.q,
                };
                ModularArithmeticCols { io, aux }.flatten()
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();

        let blank_row = ModularArithmeticCols {
            io: Default::default(),
            aux: ModularArithmeticAuxCols {
                is_valid: Default::default(),
                read_x_aux_cols: MemoryReadAuxCols::disabled(),
                read_y_aux_cols: MemoryReadAuxCols::disabled(),
                write_z_aux_cols: MemoryWriteAuxCols::disabled(),
                x_address_aux_cols: MemoryReadAuxCols::disabled(),
                y_address_aux_cols: MemoryReadAuxCols::disabled(),
                z_address_aux_cols: MemoryReadAuxCols::disabled(),
                carries: vec![F::zero(); self.air.carry_limbs],
                q: vec![F::zero(); self.air.q_limbs],
            },
        }
        .flatten();
        let width = blank_row.len();

        let mut padded_rows = rows;
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
