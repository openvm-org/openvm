use std::array;

use itertools::{all, Itertools};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::columns::{
    FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols, FieldExtensionArithmeticIoCols,
};
use crate::{
    cpu::OpCode,
    field_extension::chip::{
        FieldExtensionArithmetic, FieldExtensionArithmeticChip, FieldExtensionArithmeticRecord,
        EXTENSION_DEGREE,
    },
    memory::{
        manager::{trace_builder::MemoryTraceBuilder, MemoryAccess},
        OpType,
    },
};

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    FieldExtensionArithmeticChip<NUM_WORDS, WORD_SIZE, F>
{
    /// Generates trace for field arithmetic chip.
    ///
    /// NOTE: may only be called once on a chip. TODO: make consume self or change behavior.
    pub fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let curr_height = self.records.len();
        let correct_height = curr_height.next_power_of_two();

        let width = FieldExtensionArithmeticCols::<WORD_SIZE, F>::get_width(&self.air);
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

    /// Constructs a new set of columns (including auxiliary columns) given inputs.
    fn cols_from_record(
        &self,
        record: FieldExtensionArithmeticRecord<WORD_SIZE, F>,
    ) -> FieldExtensionArithmeticCols<WORD_SIZE, F> {
        let is_add = F::from_bool(record.opcode == OpCode::FE4ADD);
        let is_sub = F::from_bool(record.opcode == OpCode::FE4SUB);
        let is_mul = F::from_bool(record.opcode == OpCode::BBE4MUL);
        let is_inv = F::from_bool(record.opcode == OpCode::BBE4INV);

        let FieldExtensionArithmeticRecord { x, y, z, .. } = record;

        let inv = if all(x, |xi| xi == F::zero()) {
            x
        } else {
            FieldExtensionArithmetic::solve(OpCode::BBE4INV, x, y).unwrap()
        };

        let access_to_aux = |access| {
            MemoryTraceBuilder::<NUM_WORDS, WORD_SIZE, F>::memory_access_to_checker_aux_cols(
                &self.air.mem_oc,
                self.range_checker.clone(),
                &access,
            )
        };

        FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIoCols {
                clk: F::from_canonical_usize(record.clk),
                opcode: F::from_canonical_usize(record.opcode as usize),
                op_a: record.op_a,
                op_b: record.op_b,
                op_c: record.op_c,
                d: record.d,
                e: record.e,
                x,
                y,
                z,
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: F::from_bool(record.is_valid),
                valid_y_read: if record.is_valid {
                    F::one() - is_inv
                } else {
                    F::zero()
                },
                is_add,
                is_sub,
                is_mul,
                is_inv,
                inv,
                read_x_aux_cols: record.x_reads.map(access_to_aux),
                read_y_aux_cols: record.y_reads.map(access_to_aux),
                write_aux_cols: record.z_writes.map(access_to_aux),
            },
        }
    }

    fn make_blank_row(&self) -> FieldExtensionArithmeticCols<WORD_SIZE, F> {
        let clk = self.memory.borrow().get_clk();

        let make_aux_col = |op_type| {
            let access = MemoryAccess::disabled_op(clk, F::zero(), op_type);
            MemoryTraceBuilder::<NUM_WORDS, WORD_SIZE, F>::memory_access_to_checker_aux_cols(
                &self.air.mem_oc,
                self.range_checker.clone(),
                &access,
            )
        };

        FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIoCols {
                clk,
                opcode: F::from_canonical_u32(OpCode::FE4ADD as u32),
                op_a: F::zero(),
                op_b: F::zero(),
                op_c: F::zero(),
                d: F::zero(),
                e: F::zero(),
                x: [F::zero(); EXTENSION_DEGREE],
                y: [F::zero(); EXTENSION_DEGREE],
                z: [F::zero(); EXTENSION_DEGREE],
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: F::zero(),
                valid_y_read: F::zero(),
                is_add: F::one(),
                is_sub: F::zero(),
                is_mul: F::zero(),
                is_inv: F::zero(),
                inv: [F::zero(); EXTENSION_DEGREE],
                read_x_aux_cols: array::from_fn(|_| make_aux_col(OpType::Read)),
                read_y_aux_cols: array::from_fn(|_| make_aux_col(OpType::Read)),
                write_aux_cols: array::from_fn(|_| make_aux_col(OpType::Write)),
            },
        }
    }
}
