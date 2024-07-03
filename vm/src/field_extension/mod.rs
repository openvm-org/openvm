use p3_field::{Field, PrimeField32};

use crate::{
    cpu::{compose, decompose, trace::isize_to_field, OpCode},
    vm::VirtualMachine,
};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub const BETA: usize = 11;
pub const EXTENSION_DEGREE: usize = 4;
pub const TIMESTAMP_FACTOR: usize = 20;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FieldExtensionArithmeticOperation<F> {
    pub opcode: OpCode,
    pub operand1: [F; EXTENSION_DEGREE],
    pub operand2: [F; EXTENSION_DEGREE],
    pub result: [F; EXTENSION_DEGREE],
}

impl<F: Field> FieldExtensionArithmeticOperation<F> {
    pub fn from_isize(
        opcode: OpCode,
        operand1: [isize; EXTENSION_DEGREE],
        operand2: [isize; EXTENSION_DEGREE],
        result: [isize; EXTENSION_DEGREE],
    ) -> Self {
        Self {
            opcode,
            operand1: [
                isize_to_field(operand1[0]),
                isize_to_field(operand1[1]),
                isize_to_field(operand1[2]),
                isize_to_field(operand1[3]),
            ],
            operand2: [
                isize_to_field(operand2[0]),
                isize_to_field(operand2[1]),
                isize_to_field(operand2[2]),
                isize_to_field(operand2[3]),
            ],
            result: [
                isize_to_field(result[0]),
                isize_to_field(result[1]),
                isize_to_field(result[2]),
                isize_to_field(result[3]),
            ],
        }
    }

    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![F::from_canonical_usize(self.opcode as usize)];
        result.extend(self.operand1.iter());
        result.extend(self.operand2.iter());
        result.extend(self.result.iter());
        result
    }
}

/// Field extension arithmetic chip. The irreducible polynomial is x^4 - 11.
#[derive(Default, Clone, Copy)]
pub struct FieldExtensionArithmeticAir<F: Field> {
    pub clock_cycle: usize,
    pub op_a: F,
    pub op_b: F,
    pub op_c: F,
    pub d: F,
    pub e: F,
}

impl<F: Field> FieldExtensionArithmeticAir<F> {
    pub const BASE_OP: u8 = OpCode::FE4ADD as u8;
    pub const BUS_INDEX: usize = 3;

    pub fn new() -> Self {
        Self {
            clock_cycle: 0,
            op_a: F::zero(),
            op_b: F::zero(),
            op_c: F::zero(),
            d: F::zero(),
            e: F::zero(),
        }
    }

    /// Converts vectorized opcodes and operands into vectorized FieldExtensionOperations.
    pub fn request<T: Field>(
        ops: Vec<OpCode>,
        operands: Vec<([T; EXTENSION_DEGREE], [T; EXTENSION_DEGREE])>,
    ) -> Vec<FieldExtensionArithmeticOperation<T>> {
        ops.iter()
            .zip(operands.iter())
            .map(|(op, operand)| FieldExtensionArithmeticOperation {
                opcode: *op,
                operand1: operand.0,
                operand2: operand.1,
                result: Self::solve::<T>(*op, operand.0, operand.1).unwrap(),
            })
            .collect()
    }

    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for non field extension add/sub operations.
    pub fn solve<T: Field>(
        op: OpCode,
        operand1: [T; EXTENSION_DEGREE],
        operand2: [T; EXTENSION_DEGREE],
    ) -> Option<[T; EXTENSION_DEGREE]> {
        let a0 = operand1[0];
        let a1 = operand1[1];
        let a2 = operand1[2];
        let a3 = operand1[3];

        let b0 = operand2[0];
        let b1 = operand2[1];
        let b2 = operand2[2];
        let b3 = operand2[3];

        let beta_f = T::from_canonical_usize(BETA);

        match op {
            OpCode::FE4ADD => Some([a0 + b0, a1 + b1, a2 + b2, a3 + b3]),
            OpCode::FE4SUB => Some([a0 - b0, a1 - b1, a2 - b2, a3 - b3]),
            OpCode::BBE4MUL => Some([
                a0 * b0 + beta_f * (a1 * b3 + a2 * b2 + a3 * b1),
                a0 * b1 + a1 * b0 + beta_f * (a2 * b3 + a3 * b2),
                a0 * b2 + a1 * b1 + a2 * b0 + beta_f * a3 * b3,
                a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0,
            ]),
            // Let x be the vector we are taking the inverse of ([x[0], x[1], x[2], x[3]]), and define
            // x' = [x[0], -x[1], x[2], -x[3]]. We want to compute 1 / x = x' / (x * x'). Let the
            // denominator x * x' = y. By construction, y will have the degree 1 and degree 3 coefficients
            // equal to 0. Let the degree 0 coefficient be n and the degree 2 coefficient be m. Now,
            // define y' as y but with the m negated. Note that y * y' = n^2 - 11 * m^2, which is an
            // element of the original field, which we can call c. We can invert c as usual and find that
            // 1 / x = x' / (x * x') = x' * y' / c = x' * y' * c^(-1). We multiply out as usual to obtain
            // the answer.
            OpCode::BBE4INV => {
                let mut n = a0 * a0 - beta_f * (T::two() * a1 * a3 - a2 * a2);
                let mut m = T::two() * a0 * a2 - a1 * a1 - beta_f * a3 * a3;

                let c = n * n - beta_f * m * m;
                let inv_c = c.inverse();

                n *= inv_c;
                m *= inv_c;

                let result = [
                    a0 * n - beta_f * a2 * m,
                    -a1 * n + beta_f * a3 * m,
                    -a0 * m + a2 * n,
                    a1 * m - a3 * n,
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
        let mut result = Vec::<[T; EXTENSION_DEGREE]>::new();

        for i in 0..ops.len() {
            match Self::solve::<T>(ops[i], operands[i].0, operands[i].1) {
                Some(res) => result.push(res),
                None => {
                    panic!("FieldExtensionArithmeticAir::solve_all: non-field extension opcode")
                }
            }
        }

        result
    }
}

pub struct FieldExtensionArithmeticChip<const WORD_SIZE: usize, F: PrimeField32> {
    pub air: FieldExtensionArithmeticAir<F>,
    pub operations: Vec<FieldExtensionArithmeticOperation<F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> FieldExtensionArithmeticChip<WORD_SIZE, F> {
    pub fn new() -> Self {
        Self {
            air: FieldExtensionArithmeticAir::new(),
            operations: vec![],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn calculate(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
        clk: usize,
        op: OpCode,
        op_a: F,
        op_b: F,
        op_c: F,
        d: F,
        e: F,
    ) -> [F; EXTENSION_DEGREE] {
        let timestamp = clk * TIMESTAMP_FACTOR;
        let operand1 = FieldExtensionArithmeticChip::read_extension_element(vm, timestamp, d, op_b);
        let operand2 = if op == OpCode::BBE4INV {
            [[F::zero(); WORD_SIZE]; EXTENSION_DEGREE]
        } else {
            FieldExtensionArithmeticChip::read_extension_element(vm, timestamp + 5, e, op_c)
        };

        let operand1_comp = [
            compose(operand1[0]),
            compose(operand1[1]),
            compose(operand1[2]),
            compose(operand1[3]),
        ];
        let operand2_comp = [
            compose(operand2[0]),
            compose(operand2[1]),
            compose(operand2[2]),
            compose(operand2[3]),
        ];

        let result =
            FieldExtensionArithmeticAir::<F>::solve::<F>(op, operand1_comp, operand2_comp).unwrap();
        let result_decomp = [
            decompose(result[0]),
            decompose(result[1]),
            decompose(result[2]),
            decompose(result[3]),
        ];

        FieldExtensionArithmeticChip::write_extension_element(
            vm,
            timestamp + 9,
            d,
            op_a,
            result_decomp,
        );

        vm.field_extension_chip
            .operations
            .push(FieldExtensionArithmeticOperation {
                opcode: op,
                operand1: operand1_comp,
                operand2: operand2_comp,
                result,
            });

        result
    }

    fn read_extension_element(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
        timestamp: usize,
        address_space: F,
        address: F,
    ) -> [[F; WORD_SIZE]; EXTENSION_DEGREE] {
        assert!(address_space != F::zero());

        let mut result = [[F::zero(); WORD_SIZE]; EXTENSION_DEGREE];

        for (i, result_row) in result.iter_mut().enumerate() {
            let data = vm
                .memory_chip
                .read_word(timestamp + i, address_space, address);

            *result_row = data;
        }

        result
    }

    fn write_extension_element(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
        timestamp: usize,
        address_space: F,
        address: F,
        result: [[F; WORD_SIZE]; EXTENSION_DEGREE],
    ) {
        assert!(address_space != F::zero());

        for (i, row) in result.iter().enumerate() {
            vm.memory_chip
                .write_word(timestamp + i, address_space, address, *row);
        }
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> Default
    for FieldExtensionArithmeticChip<WORD_SIZE, F>
{
    fn default() -> Self {
        Self::new()
    }
}
