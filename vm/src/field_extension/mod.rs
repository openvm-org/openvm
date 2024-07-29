use p3_field::{Field, PrimeField32};

use crate::cpu::trace::Instruction;
use crate::cpu::FIELD_EXTENSION_INSTRUCTIONS;
use crate::{cpu::OpCode, vm::VirtualMachine};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub const BETA: usize = 11;
pub const EXTENSION_DEGREE: usize = 4;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FieldExtensionArithmeticOperation<F> {
    pub start_timestamp: usize,
    pub opcode: OpCode,
    pub op_a: F,
    pub op_b: F,
    pub op_c: F,
    pub d: F,
    pub e: F,
    pub operand1: [F; EXTENSION_DEGREE],
    pub operand2: [F; EXTENSION_DEGREE],
    pub result: [F; EXTENSION_DEGREE],
}

/// Field extension arithmetic chip. The irreducible polynomial is x^4 - 11.
#[derive(Default, Clone, Copy)]
pub struct FieldExtensionArithmeticAir {}

impl FieldExtensionArithmeticAir {
    pub const BASE_OP: u8 = OpCode::FE4ADD as u8;

    pub fn max_accesses_per_instruction(opcode: OpCode) -> usize {
        assert!(FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode));
        12
    }

    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for opcodes not in cpu::FIELD_EXTENSION_INSTRUCTIONS.
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
}

pub struct FieldExtensionArithmeticChip<const WORD_SIZE: usize, F: PrimeField32> {
    pub air: FieldExtensionArithmeticAir,
    pub operations: Vec<FieldExtensionArithmeticOperation<F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> FieldExtensionArithmeticChip<WORD_SIZE, F> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            air: FieldExtensionArithmeticAir {},
            operations: vec![],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn calculate(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
        start_timestamp: usize,
        instruction: Instruction<F>,
    ) -> [F; EXTENSION_DEGREE] {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            debug: _debug,
        } = instruction;
        assert!(FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode));

        let operand1 =
            FieldExtensionArithmeticChip::read_extension_element(vm, start_timestamp, d, op_b);
        let operand2 = if opcode == OpCode::BBE4INV {
            [F::zero(); EXTENSION_DEGREE]
        } else {
            FieldExtensionArithmeticChip::read_extension_element(vm, start_timestamp + 4, e, op_c)
        };

        let result = FieldExtensionArithmeticAir::solve::<F>(opcode, operand1, operand2).unwrap();

        FieldExtensionArithmeticChip::write_extension_element(
            vm,
            start_timestamp + 8,
            d,
            op_a,
            result,
        );

        vm.field_extension_chip
            .operations
            .push(FieldExtensionArithmeticOperation {
                start_timestamp,
                opcode,
                op_a,
                op_b,
                op_c,
                d,
                e,
                operand1,
                operand2,
                result,
            });

        result
    }

    fn read_extension_element(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
        timestamp: usize,
        address_space: F,
        address: F,
    ) -> [F; EXTENSION_DEGREE] {
        assert!(address_space != F::zero());

        let mut result = [F::zero(); EXTENSION_DEGREE];

        for (i, result_elem) in result.iter_mut().enumerate() {
            let data = vm.memory_chip.read_elem(
                timestamp + i,
                address_space,
                address + F::from_canonical_usize(i * WORD_SIZE),
            );

            *result_elem = data;
        }

        result
    }

    fn write_extension_element(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
        timestamp: usize,
        address_space: F,
        address: F,
        result: [F; EXTENSION_DEGREE],
    ) {
        assert!(address_space != F::zero());

        for (i, row) in result.iter().enumerate() {
            vm.memory_chip.write_elem(
                timestamp + i,
                address_space,
                address + F::from_canonical_usize(i * WORD_SIZE),
                *row,
            );
        }
    }
}
