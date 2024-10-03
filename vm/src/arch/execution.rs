use std::{array::from_fn, mem::size_of};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::{AbstractField, Field};

use crate::program::{bridge::ProgramBus, ExecutionError, Instruction};

pub type Result<T> = std::result::Result<T, ExecutionError>;

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

#[derive(Clone, Copy, Debug)]
pub struct ExecutionBus(pub usize);

#[derive(Copy, Clone, Debug)]
pub struct ExecutionBridge {
    execution_bus: ExecutionBus,
    program_bus: ProgramBus,
}

pub struct ExecutionBridgeInteractor<AB: InteractionBuilder> {
    execution_bus: ExecutionBus,
    program_bus: ProgramBus,
    opcode: AB::Expr,
    operands: Vec<AB::Expr>,
    from_state: ExecutionState<AB::Expr>,
    to_state: ExecutionState<AB::Expr>,
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

impl ExecutionBus {
    pub fn execute_and_increment_pc<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        multiplicity: impl Into<AB::Expr>,
        prev_state: ExecutionState<AB::Expr>,
        timestamp_change: impl Into<AB::Expr>,
    ) {
        let next_state = ExecutionState {
            pc: prev_state.pc.clone() + AB::F::one(),
            timestamp: prev_state.timestamp.clone() + timestamp_change.into(),
        };
        self.execute(builder, multiplicity, prev_state, next_state);
    }
    pub fn execute<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        multiplicity: impl Into<AB::Expr>,
        prev_state: ExecutionState<impl Into<AB::Expr>>,
        next_state: ExecutionState<impl Into<AB::Expr>>,
    ) {
        let multiplicity = multiplicity.into();
        builder.push_receive(
            self.0,
            [prev_state.pc.into(), prev_state.timestamp.into()],
            multiplicity.clone(),
        );
        builder.push_send(
            self.0,
            [next_state.pc.into(), next_state.timestamp.into()],
            multiplicity,
        );
    }
}

impl ExecutionBridge {
    pub fn new(execution_bus: ExecutionBus, program_bus: ProgramBus) -> Self {
        Self {
            execution_bus,
            program_bus,
        }
    }

    pub fn execute_and_increment_pc<AB: InteractionBuilder>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        timestamp_change: impl Into<AB::Expr>,
    ) -> ExecutionBridgeInteractor<AB> {
        let to_state = ExecutionState {
            pc: from_state.pc.clone().into() + AB::Expr::one(),
            timestamp: from_state.timestamp.clone().into() + timestamp_change.into(),
        };
        self.execute(opcode, operands, from_state, to_state)
    }

    pub fn execute<AB: InteractionBuilder>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        to_state: ExecutionState<impl Into<AB::Expr>>,
    ) -> ExecutionBridgeInteractor<AB> {
        ExecutionBridgeInteractor {
            execution_bus: self.execution_bus,
            program_bus: self.program_bus,
            opcode: opcode.into(),
            operands: operands.into_iter().map(Into::into).collect(),
            from_state: from_state.map(Into::into),
            to_state: to_state.map(Into::into),
        }
    }
}

impl<AB: InteractionBuilder> ExecutionBridgeInteractor<AB> {
    pub fn eval(self, builder: &mut AB, multiplicity: impl Into<AB::Expr>) {
        let multiplicity = multiplicity.into();

        // Interaction with program
        self.program_bus.send_instruction(
            builder,
            self.from_state.pc.clone(),
            self.opcode,
            self.operands,
            multiplicity.clone(),
        );

        self.execution_bus
            .execute(builder, multiplicity, self.from_state, self.to_state);
    }
}
