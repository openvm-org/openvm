use std::{
    array,
    cell::RefCell,
    ops::{Add, Mul, Sub},
    rc::Rc,
    sync::Arc,
};

use afs_primitives::range_gate::RangeCheckerGateChip;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    cpu::{trace::Instruction, OpCode, FIELD_EXTENSION_INSTRUCTIONS},
    field_extension::air::FieldExtensionArithmeticAir,
    memory::{
        manager::{MemoryAccess, MemoryManager},
        offline_checker::bridge::{emb, proj, MemoryOfflineChecker},
    },
    vm::config::MemoryConfig,
};

pub const BETA: usize = 11;
pub const EXTENSION_DEGREE: usize = 4;

/// Records an arithmetic operation that happened at run-time.
#[derive(Debug)]
pub struct FieldExtensionArithmeticRecord<const WORD_SIZE: usize, F> {
    pub clk: usize,
    pub opcode: OpCode,
    pub is_valid: bool,
    // TODO[zach]: these entries are redundant with the memory accesses below.
    pub op_a: F,
    pub op_b: F,
    pub op_c: F,
    pub d: F,
    pub e: F,
    pub x: [F; EXTENSION_DEGREE],
    pub y: [F; EXTENSION_DEGREE],
    pub z: [F; EXTENSION_DEGREE],
    /// Memory accesses for reading `x`.
    pub x_reads: [MemoryAccess<WORD_SIZE, F>; EXTENSION_DEGREE],
    /// Memory accesses for reading `y`.
    pub y_reads: [MemoryAccess<WORD_SIZE, F>; EXTENSION_DEGREE],
    /// Memory accesses for writing `z`.
    pub z_writes: [MemoryAccess<WORD_SIZE, F>; EXTENSION_DEGREE],
}

/// A chip for performing arithmetic operations over the field extension
/// defined by the irreducible polynomial x^4 - 11.
pub struct FieldExtensionArithmeticChip<
    const NUM_WORDS: usize,
    const WORD_SIZE: usize,
    F: PrimeField32,
> {
    pub air: FieldExtensionArithmeticAir<WORD_SIZE>,
    pub(crate) memory: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
    pub(crate) range_checker: Arc<RangeCheckerGateChip>,
    pub(crate) records: Vec<FieldExtensionArithmeticRecord<WORD_SIZE, F>>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    FieldExtensionArithmeticChip<NUM_WORDS, WORD_SIZE, F>
{
    #[allow(clippy::new_without_default)]
    pub fn new(
        mem_config: MemoryConfig,
        memory: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        let air = FieldExtensionArithmeticAir {
            mem_oc: MemoryOfflineChecker::new(mem_config.clk_max_bits, mem_config.decomp),
        };
        Self {
            air,
            records: vec![],
            memory,
            range_checker,
        }
    }

    pub fn accesses_per_instruction(opcode: OpCode) -> usize {
        assert!(FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode));
        match opcode {
            OpCode::BBE4INV => 8,
            _ => 12,
        }
    }

    pub fn process(&mut self, clk: usize, instruction: Instruction<F>) -> [F; EXTENSION_DEGREE] {
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
        let x: [F; EXTENSION_DEGREE] = array::from_fn(|i| proj(x_reads[i].op.cell.data));

        let y_reads = if opcode == OpCode::BBE4INV {
            array::from_fn(|_| self.memory.borrow_mut().disabled_read(e))
        } else {
            self.read_extension_element(e, op_c)
        };
        let y: [F; EXTENSION_DEGREE] = array::from_fn(|i| proj(y_reads[i].op.cell.data));

        let z = FieldExtensionArithmetic::solve(opcode, x, y).unwrap();

        let z_writes = self.write_extension_element(d, op_a, z);

        self.records.push(FieldExtensionArithmeticRecord {
            clk,
            opcode,
            is_valid: true,
            op_a,
            op_b,
            op_c,
            d,
            e,
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
    ) -> [MemoryAccess<WORD_SIZE, F>; EXTENSION_DEGREE] {
        assert_ne!(address_space, F::zero());

        array::from_fn(|i| {
            self.memory.borrow_mut().read_word(
                address_space,
                address + F::from_canonical_usize(i * WORD_SIZE),
            )
        })
    }

    fn write_extension_element(
        &mut self,
        address_space: F,
        address: F,
        result: [F; EXTENSION_DEGREE],
    ) -> [MemoryAccess<WORD_SIZE, F>; EXTENSION_DEGREE] {
        assert_ne!(address_space, F::zero());

        array::from_fn(|i| {
            self.memory.borrow_mut().write_word(
                address_space,
                address + F::from_canonical_usize(i),
                emb(result[i]),
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
        op: OpCode,
        x: [T; EXTENSION_DEGREE],
        y: [T; EXTENSION_DEGREE],
    ) -> Option<[T; EXTENSION_DEGREE]> {
        match op {
            OpCode::FE4ADD => Some(Self::add(x, y)),
            OpCode::FE4SUB => Some(Self::subtract(x, y)),
            OpCode::BBE4MUL => Some(Self::multiply(x, y)),
            OpCode::BBE4INV => Some(Self::invert(x)),
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

    fn invert<T: Field>(a: [T; EXTENSION_DEGREE]) -> [T; EXTENSION_DEGREE] {
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
