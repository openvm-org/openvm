use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{ModularArithmeticAuxCols, ModularArithmeticCols, ModularArithmeticIoCols},
    ModularArithmeticAirVariant, ModularArithmeticChip, ModularArithmeticRecord, NUM_LIMBS,
};
use crate::{
    arch::{chips::MachineChip, columns::ExecutionState},
    memory::{
        offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
        MemoryDataIoCols, MemoryHeapDataIoCols,
    },
};

impl<F: PrimeField32> MachineChip<F> for ModularArithmeticChip<F, ModularArithmeticAirVariant> {
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
        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

        let rows = self
            .data
            .iter()
            .map(|record| {
                let ModularArithmeticRecord {
                    from_state,
                    instruction: _instruction, // FIXME: use opcode
                    x_array_read,
                    y_array_read,
                    z_array_write,
                } = record;
                let io = ModularArithmeticIoCols {
                    from_state: from_state.map(F::from_canonical_usize),
                    x: MemoryHeapDataIoCols::<F, NUM_LIMBS>::from(x_array_read.clone()),
                    y: MemoryHeapDataIoCols::<F, NUM_LIMBS>::from(y_array_read.clone()),
                    z: MemoryHeapDataIoCols::<F, NUM_LIMBS>::from(z_array_write.clone()),
                };
                let x_limbs = x_array_read
                    .data_read
                    .data
                    .iter()
                    .map(|x| x.as_canonical_u32())
                    .collect::<Vec<_>>();
                let x_biguint = super::limbs_to_biguint(&x_limbs);
                let y_limbs = y_array_read
                    .data_read
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
                    read_x_aux_cols: aux_cols_factory.make_heap_read_aux_cols(x_array_read.clone()),
                    read_y_aux_cols: aux_cols_factory.make_heap_read_aux_cols(y_array_read.clone()),
                    write_z_aux_cols: aux_cols_factory
                        .make_heap_write_aux_cols(z_array_write.clone()),
                    carries: primitive_row.carries,
                    q: primitive_row.q,
                    opcode: F::from_canonical_u8(record.instruction.opcode as u8),
                };
                ModularArithmeticCols { io, aux }.flatten()
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();

        let dummy_mem_data = MemoryDataIoCols {
            data: [F::zero(); NUM_LIMBS],
            address_space: F::zero(),
            address: F::zero(),
        };
        let dummy_mem_addr = MemoryDataIoCols {
            data: [F::zero()],
            address_space: F::zero(),
            address: F::zero(),
        };
        let dummy_mem_heap_data = MemoryHeapDataIoCols {
            address: dummy_mem_addr.clone(),
            data: dummy_mem_data.clone(),
        };
        let blank_row = ModularArithmeticCols {
            io: ModularArithmeticIoCols {
                from_state: ExecutionState::default(),
                x: dummy_mem_heap_data.clone(),
                y: dummy_mem_heap_data.clone(),
                z: dummy_mem_heap_data.clone(),
            },
            aux: ModularArithmeticAuxCols {
                is_valid: Default::default(),
                read_x_aux_cols: MemoryHeapReadAuxCols::disabled(),
                read_y_aux_cols: MemoryHeapReadAuxCols::disabled(),
                write_z_aux_cols: MemoryHeapWriteAuxCols::disabled(),
                carries: vec![F::zero(); self.air.carry_limbs],
                q: vec![F::zero(); self.air.q_limbs],
                opcode: F::zero(),
            },
        }
        .flatten();
        let width = blank_row.len();

        let mut padded_rows = rows;
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
