use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::columns::{FieldArithmeticAuxCols, FieldArithmeticCols, FieldArithmeticIOCols};
use crate::cpu::{trace::ProgramExecution, OpCode};

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
    let quotient = if y == T::zero() {
        T::zero()
    } else {
        x * y.inverse()
    };
    let z = is_mul * product + is_div * quotient + (T::one() - opcode_hi) * sum_or_diff;

    FieldArithmeticCols {
        io: FieldArithmeticIOCols {
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
        },
    }
}

impl FieldArithmeticAir {
    /// Generates trace for field arithmetic chip.
    pub fn generate_trace<T: Field>(&self, prog_exec: &ProgramExecution<T>) -> RowMajorMatrix<T> {
        let trace = prog_exec
            .arithmetic_ops
            .iter()
            .flat_map(|op| {
                let cols = generate_cols(op.opcode, op.operand1, op.operand2);
                cols.flatten()
            })
            .collect();

        RowMajorMatrix::new(trace, FieldArithmeticCols::<T>::NUM_COLS)
    }
}
