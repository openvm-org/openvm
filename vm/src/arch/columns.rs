use std::fmt::Debug;

use itertools::Itertools;
use p3_field::{AbstractField, Field};

use crate::cpu::trace::Instruction;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ExecutionState<T> {
    pub pc: T,
    pub timestamp: T,
}

pub const NUM_OPERANDS: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct InstructionCols<T> {
    pub opcode: T,
    pub operands: [T; NUM_OPERANDS],
}

impl<T: Clone> ExecutionState<T> {
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
    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            pc: slice[0].clone(),
            timestamp: slice[1].clone(),
        }
    }
    pub fn flatten(&self) -> Vec<T> {
        vec![self.pc.clone(), self.timestamp.clone()]
    }
    pub fn get_width() -> usize {
        2
    }

    pub fn map<U: Clone, F: Fn(T) -> U>(&self, function: F) -> ExecutionState<U> {
        ExecutionState::from_slice(&self.flatten().into_iter().map(function).collect_vec())
    }
}

impl<F: AbstractField> InstructionCols<F> {
    pub fn new<const N: usize>(opcode: impl Into<F>, operands: [impl Into<F>; N]) -> Self {
        let mut operands = operands.into_iter().map(Into::into).collect_vec();
        while operands.len() != NUM_OPERANDS {
            operands.push(F::zero());
        }
        Self {
            opcode: opcode.into(),
            operands: operands.try_into().unwrap(),
        }
    }
}

impl<T: Clone + Debug> InstructionCols<T> {
    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            opcode: slice[0].clone(),
            operands: slice[1..].to_vec().try_into().unwrap(),
        }
    }
    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![self.opcode.clone()];
        result.extend(self.operands.clone());
        result
    }
    pub fn get_width() -> usize {
        1 + NUM_OPERANDS
    }
    pub fn map<U: Clone + Debug, F: Fn(T) -> U>(&self, function: F) -> InstructionCols<U> {
        InstructionCols::from_slice(&self.flatten().into_iter().map(function).collect_vec())
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
