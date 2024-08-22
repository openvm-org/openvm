use itertools::Itertools;
use p3_field::{AbstractField, Field};

use crate::cpu::trace::Instruction;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ExecutionState<T> {
    pub pc: T,
    pub timestamp: T,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct InstructionCols<T> {
    pub opcode: T,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub f: T,
    pub g: T,
}

impl<T: Clone> ExecutionState<T> {
    pub fn new(pc: impl Into<T>, timestamp: impl Into<T>) -> Self {
        Self {
            pc: pc.into(),
            timestamp: timestamp.into(),
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
    pub fn new(
        opcode: impl Into<F>,
        a: impl Into<F>,
        b: impl Into<F>,
        c: impl Into<F>,
        d: impl Into<F>,
        e: impl Into<F>,
    ) -> Self {
        Self {
            opcode: opcode.into(),
            a: a.into(),
            b: b.into(),
            c: c.into(),
            d: d.into(),
            e: e.into(),
            f: F::zero(),
            g: F::zero(),
        }
    }
    pub fn new_large(
        opcode: impl Into<F>,
        a: impl Into<F>,
        b: impl Into<F>,
        c: impl Into<F>,
        d: impl Into<F>,
        e: impl Into<F>,
        f: impl Into<F>,
        g: impl Into<F>,
    ) -> Self {
        Self {
            opcode: opcode.into(),
            a: a.into(),
            b: b.into(),
            c: c.into(),
            d: d.into(),
            e: e.into(),
            f: f.into(),
            g: g.into(),
        }
    }
}

impl<T: Clone> InstructionCols<T> {
    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            opcode: slice[0].clone(),
            a: slice[1].clone(),
            b: slice[2].clone(),
            c: slice[3].clone(),
            d: slice[4].clone(),
            e: slice[5].clone(),
            f: slice[6].clone(),
            g: slice[7].clone(),
        }
    }
    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.opcode.clone(),
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.d.clone(),
            self.e.clone(),
            self.f.clone(),
            self.g.clone(),
        ]
    }
    pub fn get_width() -> usize {
        8
    }

    pub fn map<U: Clone, F: Fn(T) -> U>(&self, function: F) -> InstructionCols<U> {
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
            a: *op_a,
            b: *op_b,
            c: *op_c,
            d: *d,
            e: *e,
            f: *op_f,
            g: *op_g,
        }
    }
}
