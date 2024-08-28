use std::array;

use afs_stark_backend::rap::AnyRap;
use itertools::Itertools;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::columns::{
    FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols, FieldExtensionArithmeticIoCols,
};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode},
    field_extension::chip::{
        FieldExtensionArithmetic, FieldExtensionArithmeticChip, FieldExtensionArithmeticRecord,
        EXTENSION_DEGREE,
    },
    memory::{
        manager::{MemoryRead, MemoryWrite},
        OpType,
    },
};

impl<F: PrimeField32> MachineChip<F> for FieldExtensionArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    ///
    /// NOTE: may only be called once on a chip. TODO: make consume self or change behavior.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let curr_height = self.records.len();
        let correct_height = curr_height.next_power_of_two();

        let width = FieldExtensionArithmeticCols::<F>::get_width(&self.air);
        let dummy_rows_flattened = (0..correct_height - curr_height)
            .flat_map(|_| self.make_blank_row().flatten())
            .collect_vec();

        let records = std::mem::take(&mut self.records);

        let mut flattened_trace: Vec<F> = records
            .into_iter()
            .flat_map(|record| self.cols_from_record(record).flatten())
            .collect();

        flattened_trace.extend(dummy_rows_flattened);

        RowMajorMatrix::new(flattened_trace, width)
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
        BaseAir::<F>::width(&self.air)
    }
}

impl<F: PrimeField32> FieldExtensionArithmeticChip<F> {
    /// Constructs a new set of columns (including auxiliary columns) given inputs.
    fn cols_from_record(
        &self,
        record: FieldExtensionArithmeticRecord<F>,
    ) -> FieldExtensionArithmeticCols<F> {
        let is_add = F::from_bool(record.instruction.opcode == Opcode::FE4ADD);
        let is_sub = F::from_bool(record.instruction.opcode == Opcode::FE4SUB);
        let is_mul = F::from_bool(record.instruction.opcode == Opcode::BBE4MUL);
        let is_div = F::from_bool(record.instruction.opcode == Opcode::BBE4DIV);

        let FieldExtensionArithmeticRecord { x, y, z, .. } = record;

        let divisor_inv = if record.instruction.opcode == Opcode::BBE4DIV {
            FieldExtensionArithmetic::invert(record.y)
        } else {
            [F::zero(); EXTENSION_DEGREE]
        };

        let memory = self.memory_chip.borrow();

        FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIoCols {
                opcode: F::from_canonical_usize(record.instruction.opcode as usize),
                pc: F::from_canonical_usize(record.pc),
                timestamp: F::from_canonical_usize(record.timestamp),
                op_a: record.instruction.op_a,
                op_b: record.instruction.op_b,
                op_c: record.instruction.op_c,
                d: record.instruction.d,
                e: record.instruction.e,
                x,
                y,
                z,
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: F::one(),
                is_add,
                is_sub,
                is_mul,
                is_div,
                divisor_inv,
                read_x_aux_cols: record.x_reads.map(|read| memory.make_read_aux_cols(read)),
                read_y_aux_cols: record.y_reads.map(|read| memory.make_read_aux_cols(read)),
                write_aux_cols: record
                    .z_writes
                    .map(|write| memory.make_write_aux_cols(write)),
            },
        }
    }

    fn make_blank_row(&self) -> FieldExtensionArithmeticCols<F> {
        let timestamp = self.memory_chip.borrow().timestamp();

        let make_aux_col = |op_type| match op_type {
            OpType::Read => self
                .memory_chip
                .borrow()
                .make_read_aux_cols(MemoryRead::disabled(timestamp, F::one())),
            OpType::Write => self
                .memory_chip
                .borrow()
                .make_write_aux_cols(MemoryWrite::disabled(timestamp, F::one())),
        };

        FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIoCols {
                timestamp,
                opcode: F::from_canonical_u32(Opcode::FE4ADD as u32),
                pc: F::zero(),
                op_a: F::zero(),
                op_b: F::zero(),
                op_c: F::zero(),
                d: F::one(),
                e: F::one(),
                x: [F::zero(); EXTENSION_DEGREE],
                y: [F::zero(); EXTENSION_DEGREE],
                z: [F::zero(); EXTENSION_DEGREE],
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: F::zero(),
                is_add: F::one(),
                is_sub: F::zero(),
                is_mul: F::zero(),
                is_div: F::zero(),
                divisor_inv: [F::zero(); EXTENSION_DEGREE],
                read_x_aux_cols: array::from_fn(|_| make_aux_col(OpType::Read)),
                read_y_aux_cols: array::from_fn(|_| make_aux_col(OpType::Read)),
                write_aux_cols: array::from_fn(|_| make_aux_col(OpType::Write)),
            },
        }
    }
}
