use std::sync::Arc;

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    rap::{get_air_name, AnyRap},
    Chip,
};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{FieldArithmeticAuxCols, FieldArithmeticCols, FieldArithmeticIoCols},
    FieldArithmeticChip, FieldArithmeticRecord, Operand,
};
use crate::{
    arch::{
        instructions::{FieldArithmeticOpcode, UsizeOpcode},
        VmChip,
    },
    system::memory::offline_checker::{MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
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
                read_x_aux_cols: MemoryReadOrImmediateAuxCols::disabled(),
                read_y_aux_cols: MemoryReadOrImmediateAuxCols::disabled(),
                write_z_aux_cols: MemoryWriteAuxCols::disabled(),
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
        let opcode = FieldArithmeticOpcode::from_usize(opcode);

        let x = x_read.value();
        let y = y_read.value();
        let z = z_write.value();

        let is_add = F::from_bool(opcode == FieldArithmeticOpcode::ADD);
        let is_sub = F::from_bool(opcode == FieldArithmeticOpcode::SUB);
        let is_div = F::from_bool(opcode == FieldArithmeticOpcode::DIV);
        let is_mul = F::from_bool(opcode == FieldArithmeticOpcode::MUL);
        let divisor_inv = if opcode == FieldArithmeticOpcode::DIV {
            y.inverse()
        } else {
            F::zero()
        };

        let aux_cols_factory = self.memory_controller.borrow().aux_cols_factory();

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
                read_x_aux_cols: aux_cols_factory.make_read_or_immediate_aux_cols(x_read),
                read_y_aux_cols: aux_cols_factory.make_read_or_immediate_aux_cols(y_read),
                write_z_aux_cols: aux_cols_factory.make_write_aux_cols(z_write),
            },
        }
    }
}

impl<F: PrimeField32> VmChip<F> for FieldArithmeticChip<F> {
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

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        FieldArithmeticCols::<F>::get_width()
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for FieldArithmeticChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }
}
