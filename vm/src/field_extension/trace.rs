use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::cpu::trace::{FieldExtensionOperation, ProgramExecution};
use crate::cpu::OpCode;

use crate::field_extension::BETA;

use super::columns::{
    FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols, FieldExtensionArithmeticIOCols,
};
use super::FieldExtensionArithmeticAir;

impl FieldExtensionArithmeticAir {
    /// Generates trace for field extension add/sub chip.
    pub fn generate_trace<T: Field>(
        &self,
        prog_exec: &ProgramExecution<1, T>,
    ) -> RowMajorMatrix<T> {
        let trace: Vec<T> = prog_exec
            .field_extension_ops
            .iter()
            .flat_map(|op| Self::generate_trace_row(*op))
            .collect();

        RowMajorMatrix::new(trace, FieldExtensionArithmeticCols::<T>::NUM_COLS)
    }

    fn generate_trace_row<T: Field>(op: FieldExtensionOperation<T>) -> Vec<T> {
        let opcode_value = op.opcode as u32 - FieldExtensionArithmeticAir::BASE_OP as u32;
        let opcode_lo_u32 = opcode_value % 2;
        let opcode_hi_u32 = opcode_value / 2;
        let opcode_lo = T::from_canonical_u32(opcode_lo_u32);
        let opcode_hi = T::from_canonical_u32(opcode_hi_u32);
        let is_mul = T::from_bool(op.opcode == OpCode::BBE4MUL);
        let is_inv = T::from_bool(op.opcode == OpCode::BBE4INV);

        let x = op.operand1;
        let y = op.operand2;

        let add_sub_coeff = T::one() - T::two() * opcode_lo;

        let sum_or_diff = [
            x[0] + add_sub_coeff * y[0],
            x[1] + add_sub_coeff * y[1],
            x[2] + add_sub_coeff * y[2],
            x[3] + add_sub_coeff * y[3],
        ];

        let product = Self::solve(OpCode::BBE4MUL, (x, y)).unwrap();

        let b0 =
            x[0] * x[0] - T::from_canonical_usize(BETA) * (T::two() * x[1] * x[3] - x[2] * x[2]);
        let b2 = T::two() * x[0] * x[2] - x[1] * x[1] - T::from_canonical_usize(BETA) * x[3] * x[3];
        let c = b0 * b0 - T::from_canonical_usize(BETA) * b2 * b2;
        let inv_c = c.inverse();

        let inv = Self::solve(OpCode::BBE4INV, (x, y)).unwrap();

        let cols = FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIOCols {
                opcode: T::from_canonical_usize(op.opcode as usize),
                x,
                y,
                z: op.result,
            },
            aux: FieldExtensionArithmeticAuxCols {
                opcode_lo,
                opcode_hi,
                is_mul,
                is_inv,
                sum_or_diff,
                product,
                inv_c,
                inv,
            },
        };

        cols.flatten()
    }
}
