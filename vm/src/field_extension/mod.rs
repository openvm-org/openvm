use p3_field::Field;

use crate::cpu::{trace::FieldExtensionOperation, OpCode};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub const BETA: usize = 11;
pub const EXTENSION_DEGREE: usize = 4;

/// Field extension arithmetic chip. The irreducible polynomial is x^4 - 11.
#[derive(Default, Clone, Copy)]
pub struct FieldExtensionArithmeticAir {}

impl FieldExtensionArithmeticAir {
    pub const BASE_OP: u8 = OpCode::FE4ADD as u8;
    pub const BUS_INDEX: usize = 2;

    pub fn new() -> Self {
        Self {}
    }

    /// Converts vectorized opcodes and operands into vectorized FieldExtensionOperations.
    pub fn request<T: Field>(
        ops: Vec<OpCode>,
        operands: Vec<([T; EXTENSION_DEGREE], [T; EXTENSION_DEGREE])>,
    ) -> Vec<FieldExtensionOperation<T>> {
        ops.iter()
            .zip(operands.iter())
            .map(|(op, operand)| FieldExtensionOperation {
                opcode: *op,
                operand1: operand.0,
                operand2: operand.1,
                result: Self::solve::<T>(*op, *operand).unwrap(),
            })
            .collect()
    }

    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for non field extension add/sub operations.
    pub fn solve<T: Field>(
        op: OpCode,
        operands: ([T; EXTENSION_DEGREE], [T; EXTENSION_DEGREE]),
    ) -> Option<[T; EXTENSION_DEGREE]> {
        match op {
            OpCode::FE4ADD => Some([
                operands.0[0] + operands.1[0],
                operands.0[1] + operands.1[1],
                operands.0[2] + operands.1[2],
                operands.0[3] + operands.1[3],
            ]),
            OpCode::FE4SUB => Some([
                operands.0[0] - operands.1[0],
                operands.0[1] - operands.1[1],
                operands.0[2] - operands.1[2],
                operands.0[3] - operands.1[3],
            ]),
            OpCode::BBE4MUL => Some([
                operands.0[0] * operands.1[0]
                    + T::from_canonical_usize(BETA)
                        * (operands.0[1] * operands.1[3]
                            + operands.0[2] * operands.1[2]
                            + operands.0[3] * operands.1[1]),
                operands.0[0] * operands.1[1]
                    + operands.0[1] * operands.1[0]
                    + T::from_canonical_usize(BETA)
                        * (operands.0[2] * operands.1[3] + operands.0[3] * operands.1[2]),
                operands.0[0] * operands.1[2]
                    + operands.0[1] * operands.1[1]
                    + operands.0[2] * operands.1[0]
                    + T::from_canonical_usize(BETA) * operands.0[3] * operands.1[3],
                operands.0[0] * operands.1[3]
                    + operands.0[1] * operands.1[2]
                    + operands.0[2] * operands.1[1]
                    + operands.0[3] * operands.1[0],
            ]),
            // Let x be the vector we are taking the inverse of ([x[0], x[1], x[2], x[3]]), and define
            // x' = [x[0], -x[1], x[2], -x[3]]. We want to compute 1 / x = x' / (x * x'). Let the
            // denominator x * x' = y. By construction, y will have the degree 1 and degree 3 coefficients
            // equal to 0. Let the degree 0 coefficient be b0 and the degree 2 coefficient be b2. Now,
            // define y' as y but with the b2 negated. Note that y * y' = b0^2 - 11 * b2^2, which is an
            // element of the original field, which we can call c. We can invert c as usual and find that
            // 1 / x = x' / (x * x') = x' * y' / c = x' * y' * c^(-1). We multiply out as usual to obtain
            // the answer.
            OpCode::BBE4INV => {
                let mut b0 = operands.0[0] * operands.0[0]
                    - T::from_canonical_usize(BETA)
                        * (T::two() * operands.0[1] * operands.0[3]
                            - operands.0[2] * operands.0[2]);
                let mut b2 = T::two() * operands.0[0] * operands.0[2]
                    - operands.0[1] * operands.0[1]
                    - T::from_canonical_usize(BETA) * operands.0[3] * operands.0[3];

                let c = b0 * b0 - T::from_canonical_usize(BETA) * b2 * b2;
                let inv_c = c.inverse();

                b0 *= inv_c;
                b2 *= inv_c;

                let result = [
                    operands.0[0] * b0 - T::from_canonical_usize(BETA) * operands.0[2] * b2,
                    -operands.0[1] * b0 + T::from_canonical_usize(BETA) * operands.0[3] * b2,
                    -operands.0[0] * b2 + operands.0[2] * b0,
                    operands.0[1] * b2 - operands.0[3] * b0,
                ];
                Some(result)
            }
            _ => None,
        }
    }

    /// Vectorized solve<>
    pub fn solve_all<T: Field>(
        ops: Vec<OpCode>,
        operands: Vec<([T; EXTENSION_DEGREE], [T; EXTENSION_DEGREE])>,
    ) -> Vec<[T; EXTENSION_DEGREE]> {
        ops.iter()
            .zip(operands.iter())
            .filter_map(|(op, operand)| Self::solve::<T>(*op, *operand))
            .collect()
    }
}
