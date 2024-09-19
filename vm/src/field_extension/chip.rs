use std::{
    array,
    ops::{Add, Mul, Sub},
};

use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, FIELD_EXTENSION_INSTRUCTIONS},
    },
    field_extension::air::FieldExtensionArithmeticAir,
    memory::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
    program::{bridge::ProgramBus, ExecutionError, Instruction},
};

pub const BETA: usize = 11;
pub const EXT_DEG: usize = 4;

/// Records an arithmetic operation that happened at run-time.
#[derive(Clone, Debug)]
pub(crate) struct FieldExtensionArithmeticRecord<F> {
    /// Program counter
    pub(crate) pc: usize,
    /// Timestamp at start of instruction
    pub(crate) timestamp: usize,
    pub(crate) instruction: Instruction<F>,
    pub(crate) x: [F; EXT_DEG],
    pub(crate) y: [F; EXT_DEG],
    pub(crate) z: [F; EXT_DEG],
    /// Memory accesses for reading `x`.
    pub(crate) x_read: MemoryReadRecord<F, EXT_DEG>,
    /// Memory accesses for reading `y`.
    pub(crate) y_read: MemoryReadRecord<F, EXT_DEG>,
    /// Memory accesses for writing `z`.
    pub(crate) z_write: MemoryWriteRecord<F, EXT_DEG>,
}

/// A chip for performing arithmetic operations over the field extension
/// defined by the irreducible polynomial x^4 - 11.
#[derive(Clone, Debug)]
pub struct FieldExtensionArithmeticChip<F: PrimeField32> {
    pub air: FieldExtensionArithmeticAir,
    pub(crate) memory_chip: MemoryChipRef<F>,
    pub(crate) records: Vec<FieldExtensionArithmeticRecord<F>>,
}

impl<F: PrimeField32> InstructionExecutor<F> for FieldExtensionArithmeticChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            ..
        } = instruction;

        assert!(FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode));

        assert_ne!(d, F::zero());
        assert_ne!(e, F::zero());

        let mut memory_chip = self.memory_chip.borrow_mut();

        let x_read = memory_chip.read(d, op_b);
        let x: [F; EXT_DEG] = x_read.data;

        let y_read = memory_chip.read(e, op_c);
        let y: [F; EXT_DEG] = y_read.data;

        let z = FieldExtensionArithmetic::solve(opcode, x, y).unwrap();
        let z_write = memory_chip.write(d, op_a, z);

        self.records.push(FieldExtensionArithmeticRecord {
            timestamp: from_state.timestamp,
            pc: from_state.pc,
            instruction,
            x,
            y,
            z,
            x_read,
            y_read,
            z_write,
        });

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        })
    }
}

impl<F: PrimeField32> FieldExtensionArithmeticChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory: MemoryChipRef<F>,
    ) -> Self {
        let air = FieldExtensionArithmeticAir::new(
            execution_bus,
            program_bus,
            memory.borrow().memory_bridge(),
        );
        Self {
            air,
            records: vec![],
            memory_chip: memory,
        }
    }

    pub fn current_height(&self) -> usize {
        self.records.len()
    }
}

pub struct FieldExtensionArithmetic;

impl FieldExtensionArithmetic {
    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for opcodes not in core::FIELD_EXTENSION_INSTRUCTIONS.
    pub(crate) fn solve<T: Field>(
        op: Opcode,
        x: [T; EXT_DEG],
        y: [T; EXT_DEG],
    ) -> Option<[T; EXT_DEG]> {
        match op {
            Opcode::FE4ADD => Some(Self::add(x, y)),
            Opcode::FE4SUB => Some(Self::subtract(x, y)),
            Opcode::BBE4MUL => Some(Self::multiply(x, y)),
            Opcode::BBE4DIV => Some(Self::divide(x, y)),
            _ => None,
        }
    }

    pub(crate) fn add<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        V: Copy,
        V: Add<V, Output = E>,
    {
        array::from_fn(|i| x[i] + y[i])
    }

    pub(crate) fn subtract<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        V: Copy,
        V: Sub<V, Output = E>,
    {
        array::from_fn(|i| x[i] - y[i])
    }

    pub(crate) fn multiply<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        E: AbstractField,
        V: Copy,
        V: Mul<V, Output = E>,
        E: Mul<V, Output = E>,
        V: Add<V, Output = E>,
        E: Add<V, Output = E>,
    {
        let [x0, x1, x2, x3] = x;
        let [y0, y1, y2, y3] = y;
        [
            x0 * y0 + (x1 * y3 + x2 * y2 + x3 * y1) * E::from_canonical_usize(BETA),
            x0 * y1 + x1 * y0 + (x2 * y3 + x3 * y2) * E::from_canonical_usize(BETA),
            x0 * y2 + x1 * y1 + x2 * y0 + (x3 * y3) * E::from_canonical_usize(BETA),
            x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0,
        ]
    }

    pub(crate) fn divide<F: Field>(x: [F; EXT_DEG], y: [F; EXT_DEG]) -> [F; EXT_DEG] {
        Self::multiply(x, Self::invert(y))
    }

    pub(crate) fn invert<T: Field>(a: [T; EXT_DEG]) -> [T; EXT_DEG] {
        // Let a = (a0, a1, a2, a3) represent the element we want to invert.
        // Define a' = (a0, -a1, a2, -a3).  By construction, the product b = a * a' will have zero
        // degree-1 and degree-3 coefficients.
        // Let b = (b0, 0, b2, 0) and define b' = (b0, 0, -b2, 0).
        // Note that c = b * b' = b0^2 - BETA * b2^2, which is an element of the base field.
        // Therefore, the inverse of a is 1 / a = a' / (a * a') = a' * b' / (b * b') = a' * b' / c.

        let [a0, a1, a2, a3] = a;

        let beta = T::from_canonical_usize(BETA);

        let mut b0 = a0 * a0 - beta * (T::two() * a1 * a3 - a2 * a2);
        let mut b2 = T::two() * a0 * a2 - a1 * a1 - beta * a3 * a3;

        let c = b0 * b0 - beta * b2 * b2;
        let inv_c = c.inverse();

        b0 *= inv_c;
        b2 *= inv_c;

        [
            a0 * b0 - a2 * b2 * beta,
            -a1 * b0 + a3 * b2 * beta,
            -a0 * b2 + a2 * b0,
            a1 * b2 - a3 * b0,
        ]
    }
}
