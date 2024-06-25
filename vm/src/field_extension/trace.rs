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
        let is_inv = T::from_bool(op.opcode == OpCode::FEINV);

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

        // Let x be the vector we are taking the inverse of ([x[0], x[1], x[2], x[3]]), and define
        // x' = [x[0], -x[1], x[2], -x[3]]. We want to compute 1 / x = x' / (x * x'). Let the
        // denominator x * x' = y. By construction, y will have the degree 1 and degree 3 coefficients
        // equal to 0. Let the degree 0 coefficient be b0 and the degree 2 coefficient be b2. Now,
        // define y' as y but with the b2 negated. Note that y * y' = b0^2 - 11 * b2^2, which is an
        // element of the original field, which we can call c. We can invert c as usual and find that
        // 1 / x = x' / (x * x') = x' * y' / c = x' * y' * c^(-1). We multiply out as usual to obtain
        // the answer.
        let mut b0 =
            x[0] * x[0] - T::from_canonical_usize(BETA) * (T::two() * x[1] * x[3] - x[2] * x[2]);
        let mut b2 =
            T::two() * x[0] * x[2] - x[1] * x[1] - T::from_canonical_usize(BETA) * x[3] * x[3];
        let c = b0 * b0 - T::from_canonical_usize(BETA) * b2 * b2;
        let inv_c = c.inverse();

        b0 *= inv_c;
        b2 *= inv_c;

        let inv = [
            x[0] * b0 - T::from_canonical_usize(BETA) * x[2] * b2,
            -x[1] * b0 + T::from_canonical_usize(BETA) * x[3] * b2,
            -x[0] * b2 + x[2] * b0,
            x[1] * b2 - x[3] * b0,
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
