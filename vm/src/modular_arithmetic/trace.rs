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
    memory::offline_checker::columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
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
                    instruction,
                    x_read,
                    y_read,
                    z_write,
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
                };
                let aux = ModularArithmeticAuxCols {
                    read_x_aux_cols: memory_chip.make_read_aux_cols(x_read.clone()),
                    read_y_aux_cols: memory_chip.make_read_aux_cols(y_read.clone()),
                    write_z_aux_cols: memory_chip.make_write_aux_cols(z_write.clone()),
                };
                ModularArithmeticCols { io, aux }.flatten()
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();

        let blank_row = ModularArithmeticCols {
            io: Default::default(),
            aux: ModularArithmeticAuxCols {
                read_x_aux_cols: MemoryReadAuxCols::disabled(self.air.mem_oc),
                read_y_aux_cols: MemoryReadAuxCols::disabled(self.air.mem_oc),
                write_z_aux_cols: MemoryWriteAuxCols::disabled(self.air.mem_oc),
            },
        }
        .flatten();
        let width = blank_row.len();

        let mut padded_rows = rows;
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
