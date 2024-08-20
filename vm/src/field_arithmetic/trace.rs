use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{FieldArithmeticAuxCols, FieldArithmeticCols, FieldArithmeticIoCols},
    FieldArithmeticChip,
};
use crate::cpu::OpCode;

/// Constructs a new set of columns (including auxiliary columns) given inputs.
fn generate_cols<T: Field>(op: OpCode, x: T, y: T) -> FieldArithmeticCols<T> {
    let is_add = T::from_bool(op == OpCode::FADD);
    let is_sub = T::from_bool(op == OpCode::FSUB);
    let is_div = T::from_bool(op == OpCode::FDIV);
    let is_mul = T::from_bool(op == OpCode::FMUL);
    let divisor_inv = if op == OpCode::FDIV {
        y.inverse()
    } else {
        T::zero()
    };

    let z = match op {
        OpCode::FADD => x + y,
        OpCode::FSUB => x - y,
        OpCode::FMUL => x * y,
        OpCode::FDIV => x * divisor_inv,
        _ => panic!("unexpected opcode {}", op),
    };

    FieldArithmeticCols {
        io: FieldArithmeticIoCols {
            rcv_count: T::one(),
            opcode: T::from_canonical_u32(op as u32),
            x,
            y,
            z,
        },
        aux: FieldArithmeticAuxCols {
            is_add,
            is_sub,
            is_mul,
            is_div,
            divisor_inv,
        },
    }
}

impl<F: Field> FieldArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        let mut trace: Vec<F> = self
            .operations
            .iter()
            .flat_map(|op| {
                let cols = generate_cols(op.opcode, op.operand1, op.operand2);
                cols.flatten()
            })
            .collect();

        let empty_row: Vec<F> = FieldArithmeticCols::blank_row().flatten();
        let curr_height = self.operations.len();
        let correct_height = curr_height.next_power_of_two();
        trace.extend(
            empty_row
                .iter()
                .cloned()
                .cycle()
                .take((correct_height - curr_height) * FieldArithmeticCols::<F>::get_width()),
        );

        RowMajorMatrix::new(trace, FieldArithmeticCols::<F>::get_width())
    }
}
