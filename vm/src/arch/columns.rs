use std::{array::from_fn, borrow::Borrow, fmt::Debug, mem::size_of};

use afs_derive::AlignedBorrow;
use itertools::Itertools;
use p3_field::{AbstractField, Field};

use crate::cpu::trace::Instruction;

#[derive(Clone, Copy, Debug, PartialEq, Default, AlignedBorrow)]
#[repr(C)]
pub struct ExecutionState<T> {
    pub pc: T,
    pub timestamp: T,
}

pub const NUM_OPERANDS: usize = 7;
pub const NUM_INSTRUCTION_COLS: usize = size_of::<InstructionCols<u8>>();

#[derive(Clone, Copy, Debug, PartialEq, Default, AlignedBorrow)]
#[repr(C)]
pub struct InstructionCols<T> {
    pub opcode: T,
    pub operands: [T; NUM_OPERANDS],
}

impl<T> ExecutionState<T> {
    pub fn new(pc: impl Into<T>, timestamp: impl Into<T>) -> Self {
        Self {
            pc: pc.into(),
            timestamp: timestamp.into(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        let mut next = || iter.next().unwrap();
        Self {
            pc: next(),
            timestamp: next(),
        }
    }

    pub fn flatten(self) -> [T; 2] {
        [self.pc, self.timestamp]
    }

    pub fn get_width() -> usize {
        2
    }

    pub fn map<U: Clone, F: Fn(T) -> U>(self, function: F) -> ExecutionState<U> {
        ExecutionState::from_iter(&mut self.flatten().map(function).into_iter())
    }
}

impl<F: AbstractField> InstructionCols<F> {
    pub fn new<const N: usize>(opcode: impl Into<F>, operands: [impl Into<F>; N]) -> Self {
        let mut operands_iter = operands.into_iter();
        debug_assert!(N <= NUM_OPERANDS);
        let operands = from_fn(|_| operands_iter.next().map(Into::into).unwrap_or(F::zero()));
        Self {
            opcode: opcode.into(),
            operands,
        }
    }
}

impl<T> InstructionCols<T> {
    // TODO[jpw]: avoid Vec
    pub fn flatten(&self) -> Vec<T>
    where
        T: Clone,
    {
        let mut result = vec![self.opcode.clone()];
        result.extend(self.operands.clone());
        result
    }
    pub fn get_width() -> usize {
        1 + NUM_OPERANDS
    }
    pub fn map<U, F: Fn(T) -> U>(&self, function: F) -> InstructionCols<U>
    where
        T: Clone,
        U: Clone,
    {
        let vec = self.flatten().into_iter().map(function).collect_vec();
        let cols: &InstructionCols<U> = vec[..].borrow();
        cols.clone()
    }
}

impl<F: Field> InstructionCols<F> {
    pub fn from_instruction(instruction: &Instruction<F>) -> Self {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            op_f,
            op_g,
            ..
        } = instruction;
        Self {
            opcode: F::from_canonical_usize(*opcode as usize),
            operands: [op_a, op_b, op_c, d, e, op_f, op_g].map(|&f| f),
        }
    }
}
