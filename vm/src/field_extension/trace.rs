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
    pub fn generate_trace<T: Field>(&self, prog_exec: &ProgramExecution<T>) -> RowMajorMatrix<T> {
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
        let is_mul = T::from_bool(op.opcode == OpCode::FEMUL);

        let x = op.operand1;
        let y = op.operand2;

        let add_sub_coeff = T::one() - T::two() * opcode_lo;

        let sum_or_diff = [
            x[0] + add_sub_coeff * y[0],
            x[1] + add_sub_coeff * y[1],
            x[2] + add_sub_coeff * y[2],
            x[3] + add_sub_coeff * y[3],
        ];

        let product = [
            x[0] * y[0] + T::from_canonical_usize(BETA) * (x[1] * y[3] + x[2] * y[2] + x[3] * y[1]),
            x[0] * y[1] + x[1] * y[0] + T::from_canonical_usize(BETA) * (x[2] * y[3] + x[3] * y[2]),
            x[0] * y[2] + x[1] * y[1] + x[2] * y[0] + T::from_canonical_usize(BETA) * (x[3] * y[3]),
            x[0] * y[3] + x[1] * y[2] + x[2] * y[1] + x[3] * y[0],
        ];

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
                sum_or_diff,
                product,
            },
        };

        cols.flatten()
    }
}
