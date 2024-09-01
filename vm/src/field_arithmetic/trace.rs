use afs_stark_backend::rap::AnyRap;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::{
    columns::{FieldArithmeticAuxCols, FieldArithmeticCols, FieldArithmeticIoCols},
    FieldArithmeticChip, FieldArithmeticRecord, Operand,
};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode},
    memory::offline_checker::columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

impl<F: PrimeField32> FieldArithmeticChip<F> {
    fn make_blank_row(&self) -> FieldArithmeticCols<F> {
        FieldArithmeticCols {
            io: Default::default(),
            aux: FieldArithmeticAuxCols {
                is_valid: F::zero(),
                is_add: F::zero(),
                is_sub: F::zero(),
                is_mul: F::zero(),
                is_div: F::zero(),
                divisor_inv: F::zero(),
                read_x_aux_cols: MemoryReadAuxCols::disabled(self.air.mem_oc),
                read_y_aux_cols: MemoryReadAuxCols::disabled(self.air.mem_oc),
                write_z_aux_cols: MemoryWriteAuxCols::disabled(self.air.mem_oc),
            },
        }
    }

    fn record_to_cols(&self, record: FieldArithmeticRecord<F>) -> FieldArithmeticCols<F> {
        let FieldArithmeticRecord {
            opcode,
            from_state,
            x_read,
            y_read,
            z_write,
        } = record;

        let x = x_read.value();
        let y = y_read.value();
        let z = z_write.value();

        let is_add = F::from_bool(opcode == Opcode::FADD);
        let is_sub = F::from_bool(opcode == Opcode::FSUB);
        let is_div = F::from_bool(opcode == Opcode::FDIV);
        let is_mul = F::from_bool(opcode == Opcode::FMUL);
        let divisor_inv = if opcode == Opcode::FDIV {
            y.inverse()
        } else {
            F::zero()
        };

        let memory_chip = self.memory_chip.borrow();

        FieldArithmeticCols {
            io: FieldArithmeticIoCols {
                from_state: from_state.map(F::from_canonical_usize),
                x: Operand::new(x_read.address_space, x_read.pointer, x),
                y: Operand::new(y_read.address_space, y_read.pointer, y),
                z: Operand::new(z_write.address_space, z_write.pointer, z),
            },
            aux: FieldArithmeticAuxCols {
                is_valid: F::one(),
                is_add,
                is_sub,
                is_mul,
                is_div,
                divisor_inv,
                read_x_aux_cols: memory_chip.make_read_aux_cols(x_read),
                read_y_aux_cols: memory_chip.make_read_aux_cols(y_read),
                write_z_aux_cols: memory_chip.make_write_aux_cols(z_write),
            },
        }
    }
}

impl<F: PrimeField32> MachineChip<F> for FieldArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let mut trace: Vec<F> = self
            .records
            .iter()
            .cloned()
            .flat_map(|op| self.record_to_cols(op).flatten())
            .collect();

        let curr_height = self.records.len();
        let correct_height = curr_height.next_power_of_two();
        // TODO[jpw]: it is better to allocate the full 1d trace matrix first and then mutate buffers
        let blank_row = self.make_blank_row().flatten();
        let dummy_rows_flattened =
            (0..correct_height - curr_height).flat_map(|_| blank_row.clone());
        trace.extend(dummy_rows_flattened);

        RowMajorMatrix::new(trace, self.trace_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        FieldArithmeticCols::<F>::get_width(&self.air)
    }
}
