use std::{array, cell::RefCell, rc::Rc};

use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::OpCodeExecutor,
        columns::ExecutionState,
        instructions::{FIELD_EXTENSION_INSTRUCTIONS, OpCode},
    },
    cpu::trace::Instruction,
    field_extension::columns::{
        FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols,
        FieldExtensionArithmeticIoCols,
    },
    memory::{
        manager::{MemoryManager, trace_builder::MemoryTraceBuilder},
        offline_checker::bridge::MemoryOfflineChecker,
        OpType,
    }
    ,
};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub const BETA: usize = 11;
pub const EXTENSION_DEGREE: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldExtensionArithmeticOperation<F> {
    pub pc: usize,
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
#[derive(Clone, Copy)]
pub struct FieldExtensionArithmeticAir {
    execution_bus: ExecutionBus,
    mem_oc: MemoryOfflineChecker,
}

impl FieldExtensionArithmeticAir {
    pub fn timestamp_delta() -> usize {
        3 * EXTENSION_DEGREE
    }
}

pub struct FieldExtensionArithmetic;

impl FieldExtensionArithmetic {
    pub const BASE_OP: u8 = OpCode::FE4ADD as u8;

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

pub struct FieldExtensionArithmeticChip<F: PrimeField32> {
    pub air: FieldExtensionArithmeticAir,
    pub operations: Vec<FieldExtensionArithmeticOperation<F>>,

    pub memory_manager: Rc<RefCell<MemoryManager<F>>>,
    pub memory: MemoryTraceBuilder<F>,
}

impl<F: PrimeField32> FieldExtensionArithmeticChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new(execution_bus: ExecutionBus, memory_manager: Rc<RefCell<MemoryManager<F>>>) -> Self {
        let air = FieldExtensionArithmeticAir {
            execution_bus,
            mem_oc: MemoryManager::make_offline_checker(memory_manager.clone()),
        };
        let memory = MemoryManager::make_trace_builder(memory_manager.clone());
        Self {
            air,
            operations: vec![],
            memory_manager,
            memory,
        }
    }

    fn read_extension_element(&mut self, address_space: F, address: F) -> [F; EXTENSION_DEGREE] {
        assert_ne!(address_space, F::zero());

        let mut result = [F::zero(); EXTENSION_DEGREE];

        for (i, result_elem) in result.iter_mut().enumerate() {
            let data = self
                .memory
                .read_elem(address_space, address + F::from_canonical_usize(i));

            *result_elem = data;
        }

        result
    }

    fn write_extension_element(
        &mut self,
        address_space: F,
        address: F,
        result: [F; EXTENSION_DEGREE],
    ) {
        assert_ne!(address_space, F::zero());

        for (i, row) in result.iter().enumerate() {
            self.memory
                .write_elem(address_space, address + F::from_canonical_usize(i), *row);
        }
    }

    pub fn make_blank_row(&self) -> FieldExtensionArithmeticCols<F> {
        let mut trace_builder = MemoryManager::make_trace_builder(self.memory_manager.clone());

        let timestamp = self.memory_manager.borrow().timestamp();

        for _ in 0..8 {
            trace_builder.disabled_op(F::zero(), OpType::Read);
        }
        for _ in 0..4 {
            trace_builder.disabled_op(F::zero(), OpType::Write);
        }
        let mut mem_oc_aux_iter = trace_builder.take_accesses_buffer().into_iter();

        FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIoCols {
                pc: F::zero(),
                opcode: F::from_canonical_u32(OpCode::FE4ADD as u32),
                timestamp: timestamp,
                x: [F::zero(); EXTENSION_DEGREE],
                y: [F::zero(); EXTENSION_DEGREE],
                z: [F::zero(); EXTENSION_DEGREE],
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: F::zero(),
                valid_y_read: F::zero(),
                op_a: F::zero(),
                op_b: F::zero(),
                op_c: F::zero(),
                d: F::zero(),
                e: F::zero(),
                is_add: F::one(),
                is_sub: F::zero(),
                is_mul: F::zero(),
                is_inv: F::zero(),
                inv: [F::zero(); EXTENSION_DEGREE],
                mem_oc_aux_cols: array::from_fn(|_| mem_oc_aux_iter.next().unwrap()),
            },
        }
    }
}

impl<F: PrimeField32> OpCodeExecutor<F> for FieldExtensionArithmeticChip<F> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            op_f: _f,
            op_g: _g,
            debug: _debug,
        } = instruction.clone();
        assert!(FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode));

        let operand1 = self.read_extension_element(d, op_b);
        let operand2 = if opcode == OpCode::BBE4INV {
            // 4 disabled reads
            for _ in 0..4 {
                self.memory.disabled_op(e, OpType::Read);
            }
            [F::zero(); EXTENSION_DEGREE]
        } else {
            self.read_extension_element(e, op_c)
        };

        let result = FieldExtensionArithmetic::solve(opcode, operand1, operand2).unwrap();

        self.write_extension_element(d, op_a, result);

        self.operations.push(FieldExtensionArithmeticOperation {
            pc: from_state.pc,
            start_timestamp: from_state.timestamp,
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

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: from_state.timestamp + FieldExtensionArithmeticAir::timestamp_delta(),
        }
    }
}
