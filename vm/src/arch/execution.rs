use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use axvm_instructions::program::DEFAULT_PC_STEP;
use p3_field::AbstractField;

use crate::system::program::{ExecutionError, ProgramBus};

pub type Result<T> = std::result::Result<T, ExecutionError>;

#[derive(Clone, Copy, Debug, PartialEq, Default, AlignedBorrow)]
#[repr(C)]
pub struct ExecutionState<T> {
    pub pc: T,
    pub timestamp: T,
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

pub enum PcIncOrSet<T> {
    Inc(T),
    Set(T),
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

    /// If `to_pc` is `Some`, then `pc_inc` is ignored and the `to_state` uses `to_pc`. Otherwise `to_pc = from_pc + pc_inc`.
    pub fn execute_and_increment_or_set_pc<AB: InteractionBuilder>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        timestamp_change: impl Into<AB::Expr>,
        pc_kind: impl Into<PcIncOrSet<AB::Expr>>,
    ) -> ExecutionBridgeInteractor<AB> {
        let to_state = ExecutionState {
            pc: match pc_kind.into() {
                PcIncOrSet::Set(to_pc) => to_pc,
                PcIncOrSet::Inc(pc_inc) => from_state.pc.clone().into() + pc_inc,
            },
            timestamp: from_state.timestamp.clone().into() + timestamp_change.into(),
        };
        self.execute(opcode, operands, from_state, to_state)
    }

    pub fn execute_and_increment_pc<AB: InteractionBuilder>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        timestamp_change: impl Into<AB::Expr>,
    ) -> ExecutionBridgeInteractor<AB> {
        let to_state = ExecutionState {
            pc: from_state.pc.clone().into() + AB::Expr::from_canonical_u32(DEFAULT_PC_STEP),
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

impl<T: AbstractField> From<(u32, Option<T>)> for PcIncOrSet<T> {
    fn from((pc_inc, to_pc): (u32, Option<T>)) -> Self {
        match to_pc {
            None => PcIncOrSet::Inc(T::from_canonical_u32(pc_inc)),
            Some(to_pc) => PcIncOrSet::Set(to_pc),
        }
    }
}
