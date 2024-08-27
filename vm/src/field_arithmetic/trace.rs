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
    arch::{chips::MachineChip, columns::ExecutionState, instructions::Opcode},
    memory::manager::MemoryAccess,
};

impl<F: PrimeField32> FieldArithmeticChip<F> {
    fn make_blank_row(&self) -> FieldArithmeticCols<F> {
        let timestamp = self.memory_chip.borrow_mut().timestamp();

        self.generate_row(
            FieldArithmeticRecord {
                opcode: Opcode::FADD,
                from_state: ExecutionState {
                    pc: 0,
                    timestamp: timestamp.as_canonical_u32() as usize,
                },
                x_read: MemoryAccess::disabled_read(timestamp, F::one()),
                y_read: MemoryAccess::disabled_read(timestamp, F::one()),
                z_write: MemoryAccess::disabled_write(timestamp, F::one()),
            },
            false,
        )
    }

    fn generate_row(
        &self,
        record: FieldArithmeticRecord<F>,
        is_valid: bool,
    ) -> FieldArithmeticCols<F> {
        let FieldArithmeticRecord {
            opcode,
            from_state,
            x_read,
            y_read,
            z_write,
        } = record;

        let [x] = x_read.op.cell.data;
        let [y] = y_read.op.cell.data;
        let [z] = z_write.op.cell.data;

        let is_add = F::from_bool(opcode == Opcode::FADD);
        let is_sub = F::from_bool(opcode == Opcode::FSUB);
        let is_div = F::from_bool(opcode == Opcode::FDIV);
        let is_mul = F::from_bool(opcode == Opcode::FMUL);
        let divisor_inv = if opcode == Opcode::FDIV {
            y.inverse()
        } else {
            F::zero()
        };

        FieldArithmeticCols {
            io: FieldArithmeticIoCols {
                opcode: F::from_canonical_u32(opcode as u32),
                from_state: from_state.map(F::from_canonical_usize),
                x: Operand::new(x_read.op.addr_space, x_read.op.pointer, x),
                y: Operand::new(y_read.op.addr_space, y_read.op.pointer, y),
                z: Operand::new(z_write.op.addr_space, z_write.op.pointer, z),
            },
            aux: FieldArithmeticAuxCols {
                is_valid: F::from_bool(is_valid),
                is_add,
                is_sub,
                is_mul,
                is_div,
                divisor_inv,
                read_x_aux_cols: self.memory_chip.borrow().make_access_cols(x_read),
                read_y_aux_cols: self.memory_chip.borrow().make_access_cols(y_read),
                write_z_aux_cols: self.memory_chip.borrow().make_access_cols(z_write),
            },
        }
    }
}

impl<F: PrimeField32> MachineChip<F> for FieldArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let mut trace: Vec<F> = self
            .records
            .iter()
            .cloned()
            .flat_map(|op| self.generate_row(op, true).flatten())
            .collect();

        let curr_height = self.records.len();
        let correct_height = curr_height.next_power_of_two();
        // WARNING: do not clone below because timestamps are different per row
        let dummy_rows_flattened =
            (0..correct_height - curr_height).flat_map(|_| self.make_blank_row().flatten());
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
