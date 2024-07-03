use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{FieldArithmeticAuxCols, FieldArithmeticCols, FieldArithmeticIOCols},
    FieldArithmeticChip,
};
use crate::cpu::OpCode;

use super::FieldArithmeticAir;

/// Constructs a new set of columns (including auxiliary columns) given inputs.
fn generate_cols<T: Field>(op: OpCode, x: T, y: T) -> FieldArithmeticCols<T> {
    let opcode = op as u32;
    let opcode_value = opcode - FieldArithmeticAir::BASE_OP as u32;
    let opcode_lo_u32 = opcode_value % 2;
    let opcode_hi_u32 = opcode_value / 2;
    let opcode_lo = T::from_canonical_u32(opcode_lo_u32);
    let opcode_hi = T::from_canonical_u32(opcode_hi_u32);
    let is_div = T::from_bool(op == OpCode::FDIV);
    let is_mul = T::from_bool(op == OpCode::FMUL);
    let sum_or_diff = x + y - T::two() * opcode_lo * y;
    let product = x * y;
    let quotient = if y == T::zero() || op != OpCode::FDIV {
        T::zero()
    } else {
        x * y.inverse()
    };
    let divisor_inv = if op != OpCode::FDIV {
        T::zero()
    } else {
        y.inverse()
    };
    let z = is_mul * product + is_div * quotient + (T::one() - opcode_hi) * sum_or_diff;

    FieldArithmeticCols {
        io: FieldArithmeticIOCols {
            rcv_count: T::one(),
            opcode: T::from_canonical_u32(opcode),
            x,
            y,
            z,
        },
        aux: FieldArithmeticAuxCols {
            opcode_lo,
            opcode_hi,
            is_mul,
            is_div,
            sum_or_diff,
            product,
            quotient,
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
                .take((correct_height - curr_height) * FieldArithmeticCols::<F>::NUM_COLS),
        );

        RowMajorMatrix::new(trace, FieldArithmeticCols::<F>::NUM_COLS)
    }
}
