use afs_stark_backend::rap::{get_air_name, AnyRap};
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::columns::{
    FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols, FieldExtensionArithmeticIoCols,
};
use crate::{
    arch::{
        instructions::{FieldExtensionOpcode, UsizeOpcode},
        MachineChip,
    },
    field_extension::chip::{
        FieldExtensionArithmetic, FieldExtensionArithmeticChip, FieldExtensionArithmeticRecord,
        EXT_DEG,
    },
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

impl<F: PrimeField32> MachineChip<F> for FieldExtensionArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    fn generate_trace(mut self) -> RowMajorMatrix<F> {
        let curr_height = self.records.len();
        let correct_height = curr_height.next_power_of_two();

        let width = FieldExtensionArithmeticCols::<F>::get_width();
        // TODO[jpw] better to create entire 1d trace matrix first and then mutate buffers
        let blank_row = self.make_blank_row().flatten();
        let dummy_rows_flattened =
            (0..correct_height - curr_height).flat_map(|_| blank_row.clone());

        let records = std::mem::take(&mut self.records);

        let flattened_trace: Vec<F> = records
            .into_iter()
            .flat_map(|record| self.cols_from_record(record).flatten())
            .chain(dummy_rows_flattened)
            .collect();

        RowMajorMatrix::new(flattened_trace, width)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
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
        let opcode = FieldExtensionOpcode::from_usize(record.instruction.opcode);
        let is_add = F::from_bool(opcode == FieldExtensionOpcode::FE4ADD);
        let is_sub = F::from_bool(opcode == FieldExtensionOpcode::FE4SUB);
        let is_mul = F::from_bool(opcode == FieldExtensionOpcode::BBE4MUL);
        let is_div = F::from_bool(opcode == FieldExtensionOpcode::BBE4DIV);

        let FieldExtensionArithmeticRecord { x, y, z, .. } = record;

        let divisor_inv = if opcode == FieldExtensionOpcode::BBE4DIV {
            FieldExtensionArithmetic::invert(record.y)
        } else {
            [F::zero(); EXT_DEG]
        };

        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

        FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIoCols {
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
                read_x_aux_cols: aux_cols_factory.make_read_aux_cols(record.x_read),
                read_y_aux_cols: aux_cols_factory.make_read_aux_cols(record.y_read),
                write_aux_cols: aux_cols_factory.make_write_aux_cols(record.z_write),
            },
        }
    }

    fn make_blank_row(&self) -> FieldExtensionArithmeticCols<F> {
        FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIoCols::default(),
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: F::zero(),
                is_add: F::zero(),
                is_sub: F::zero(),
                is_mul: F::zero(),
                is_div: F::zero(),
                divisor_inv: [F::zero(); EXT_DEG],
                read_x_aux_cols: MemoryReadAuxCols::disabled(),
                read_y_aux_cols: MemoryReadAuxCols::disabled(),
                write_aux_cols: MemoryWriteAuxCols::disabled(),
            },
        }
    }
}
