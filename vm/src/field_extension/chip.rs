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
    cpu::trace::Instruction,
    field_extension::air::FieldExtensionArithmeticAir,
    memory::manager::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
};

pub const BETA: usize = 11;
pub const EXTENSION_DEGREE: usize = 4;

/// Records an arithmetic operation that happened at run-time.
#[derive(Clone, Debug)]
pub(crate) struct FieldExtensionArithmeticRecord<F> {
    /// Program counter
    pub(crate) pc: usize,
    /// Timestamp at start of instruction
    pub(crate) timestamp: usize,
    pub(crate) instruction: Instruction<F>,
    pub(crate) x: [F; EXTENSION_DEGREE],
    pub(crate) y: [F; EXTENSION_DEGREE],
    pub(crate) z: [F; EXTENSION_DEGREE],
    /// Memory accesses for reading `x`.
    pub(crate) x_reads: [MemoryReadRecord<1, F>; EXTENSION_DEGREE],
    /// Memory accesses for reading `y`.
    pub(crate) y_reads: [MemoryReadRecord<1, F>; EXTENSION_DEGREE],
    /// Memory accesses for writing `z`.
    pub(crate) z_writes: [MemoryWriteRecord<1, F>; EXTENSION_DEGREE],
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
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        self.process(instruction.clone(), from_state);

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: from_state.timestamp + Self::accesses_per_instruction(instruction.opcode),
        }
    }
}

impl<F: PrimeField32> FieldExtensionArithmeticChip<F> {
    pub fn new(execution_bus: ExecutionBus, memory: MemoryChipRef<F>) -> Self {
        let air =
            FieldExtensionArithmeticAir::new(execution_bus, memory.borrow().make_offline_checker());
        Self {
            air,
            records: vec![],
            memory_chip: memory,
        }
    }

    pub fn accesses_per_instruction(opcode: Opcode) -> usize {
        assert!(FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode));
        3 * EXTENSION_DEGREE
    }

    pub fn process(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> [F; EXTENSION_DEGREE] {
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

        let x_reads = self.read_extension_element(d, op_b);
        let x: [F; EXTENSION_DEGREE] = array::from_fn(|i| x_reads[i].value());

        let y_reads = self.read_extension_element(e, op_c);
        let y: [F; EXTENSION_DEGREE] = array::from_fn(|i| y_reads[i].value());

        let z = FieldExtensionArithmetic::solve(opcode, x, y).unwrap();

        let z_writes = self.write_extension_element(d, op_a, z);

        self.records.push(FieldExtensionArithmeticRecord {
            timestamp: from_state.timestamp,
            pc: from_state.pc,
            instruction,
            x,
            y,
            z,
            x_reads,
            y_reads,
            z_writes,
        });

        z
    }

    fn read_extension_element(
        &mut self,
        address_space: F,
        address: F,
    ) -> [MemoryReadRecord<1, F>; EXTENSION_DEGREE] {
        assert_ne!(address_space, F::zero());

        array::from_fn(|i| {
            self.memory_chip
                .borrow_mut()
                .read_cell(address_space, address + F::from_canonical_usize(i))
        })
    }

    fn write_extension_element(
        &mut self,
        address_space: F,
        address: F,
        result: [F; EXTENSION_DEGREE],
    ) -> [MemoryWriteRecord<1, F>; EXTENSION_DEGREE] {
        assert_ne!(address_space, F::zero());

        array::from_fn(|i| {
            self.memory_chip.borrow_mut().write_cell(
                address_space,
                address + F::from_canonical_usize(i),
                result[i],
            )
        })
    }

    pub fn current_height(&self) -> usize {
        self.records.len()
    }
}

pub struct FieldExtensionArithmetic;

impl FieldExtensionArithmetic {
    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for opcodes not in cpu::FIELD_EXTENSION_INSTRUCTIONS.
    pub(crate) fn solve<T: Field>(
        op: Opcode,
        x: [T; EXTENSION_DEGREE],
        y: [T; EXTENSION_DEGREE],
    ) -> Option<[T; EXTENSION_DEGREE]> {
        match op {
            Opcode::FE4ADD => Some(Self::add(x, y)),
            Opcode::FE4SUB => Some(Self::subtract(x, y)),
            Opcode::BBE4MUL => Some(Self::multiply(x, y)),
            Opcode::BBE4DIV => Some(Self::divide(x, y)),
            _ => None,
        }
    }

    pub(crate) fn add<V, E>(
        x: [V; EXTENSION_DEGREE],
        y: [V; EXTENSION_DEGREE],
    ) -> [E; EXTENSION_DEGREE]
    where
        V: Copy,
        V: Add<V, Output = E>,
    {
        array::from_fn(|i| x[i] + y[i])
    }

    pub(crate) fn subtract<V, E>(
        x: [V; EXTENSION_DEGREE],
        y: [V; EXTENSION_DEGREE],
    ) -> [E; EXTENSION_DEGREE]
    where
        V: Copy,
        V: Sub<V, Output = E>,
    {
        array::from_fn(|i| x[i] - y[i])
    }

    pub(crate) fn multiply<V, E>(
        x: [V; EXTENSION_DEGREE],
        y: [V; EXTENSION_DEGREE],
    ) -> [E; EXTENSION_DEGREE]
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

    pub(crate) fn divide<F: Field>(
        x: [F; EXTENSION_DEGREE],
        y: [F; EXTENSION_DEGREE],
    ) -> [F; EXTENSION_DEGREE] {
        Self::multiply(x, Self::invert(y))
    }

    pub(crate) fn invert<T: Field>(a: [T; EXTENSION_DEGREE]) -> [T; EXTENSION_DEGREE] {
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
