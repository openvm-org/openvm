use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    prover::types::AirProofInput,
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use axvm_instructions::UsizeOpcode;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    arch::{instructions::CommonOpcode::TERMINATE, ExecutionBus, ExecutionState},
    system::program::ProgramBus,
};

#[cfg(test)]
mod tests;

/// When a program hasn't terminated. There is no constraints on the exit code.
/// But we will use this value when generating the proof.
pub const DEFAULT_SUSPEND_EXIT_CODE: u32 = 42;

#[derive(Debug, Clone)]
pub struct VmConnectorAir {
    pub execution_bus: ExecutionBus,
    pub program_bus: ProgramBus,
}

#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct VmConnectorPvs<F> {
    pub initial_pc: F,
    pub final_pc: F,
    pub exit_code: F,
    pub is_terminate: F,
}

impl<F: Field> BaseAirWithPublicValues<F> for VmConnectorAir {
    fn num_public_values(&self) -> usize {
        VmConnectorPvs::<F>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for VmConnectorAir {}
impl<F: Field> BaseAir<F> for VmConnectorAir {
    fn width(&self) -> usize {
        4
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(RowMajorMatrix::new_col(vec![F::zero(), F::one()]))
    }
}

#[derive(Debug, Copy, Clone, AlignedBorrow)]
#[repr(C)]
pub struct ConnectorCols<T> {
    pub pc: T,
    pub timestamp: T,
    pub is_terminate: T,
    pub exit_code: T,
}

impl<T: Copy> ConnectorCols<T> {
    fn map<F>(self, f: impl Fn(T) -> F) -> ConnectorCols<F> {
        ConnectorCols {
            pc: f(self.pc),
            timestamp: f(self.timestamp),
            is_terminate: f(self.is_terminate),
            exit_code: f(self.exit_code),
        }
    }

    fn flatten(&self) -> [T; 4] {
        [self.pc, self.timestamp, self.is_terminate, self.exit_code]
    }
}

impl<AB: InteractionBuilder + PairBuilder + AirBuilderWithPublicValues> Air<AB> for VmConnectorAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let (begin, end) = (main.row_slice(0), main.row_slice(1));

        let begin: &ConnectorCols<AB::Var> = (*begin).borrow();
        let end: &ConnectorCols<AB::Var> = (*end).borrow();

        let &VmConnectorPvs {
            initial_pc,
            final_pc,
            exit_code,
            is_terminate,
        } = builder.public_values().borrow();

        builder.when_transition().assert_eq(begin.pc, initial_pc);
        builder.when_transition().assert_eq(end.pc, final_pc);
        builder
            .when_transition()
            .when(end.is_terminate)
            .assert_eq(end.exit_code, exit_code);
        builder
            .when_transition()
            .assert_eq(end.is_terminate, is_terminate);

        self.execution_bus.execute(
            builder,
            AB::Expr::one() - prep_local[0], // 1 only if these are [0th, 1st] and not [1st, 0th]
            ExecutionState::new(end.pc, end.timestamp),
            ExecutionState::new(begin.pc, begin.timestamp),
        );
        self.program_bus.send_instruction(
            builder,
            end.pc,
            AB::Expr::from_canonical_usize(TERMINATE.with_default_offset()),
            [AB::Expr::zero(), AB::Expr::zero(), end.exit_code.into()],
            (AB::Expr::one() - prep_local[0]) * end.is_terminate,
        );
    }
}

#[derive(Debug)]
pub struct VmConnectorChip<F: PrimeField32> {
    pub air: VmConnectorAir,
    pub boundary_states: [Option<ConnectorCols<u32>>; 2],
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> VmConnectorChip<F> {
    pub fn new(execution_bus: ExecutionBus, program_bus: ProgramBus) -> Self {
        Self {
            air: VmConnectorAir {
                execution_bus,
                program_bus,
            },
            boundary_states: [None, None],
            _marker: PhantomData,
        }
    }

    pub fn begin(&mut self, state: ExecutionState<u32>) {
        self.boundary_states[0] = Some(ConnectorCols {
            pc: state.pc,
            timestamp: state.timestamp,
            is_terminate: 0,
            exit_code: 0,
        });
    }

    pub fn end(&mut self, state: ExecutionState<u32>, exit_code: Option<u32>) {
        self.boundary_states[1] = Some(ConnectorCols {
            pc: state.pc,
            timestamp: state.timestamp,
            is_terminate: exit_code.is_some() as u32,
            exit_code: exit_code.unwrap_or(DEFAULT_SUSPEND_EXIT_CODE),
        });
    }
}

impl<SC> Chip<SC> for VmConnectorChip<Val<SC>>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let [initial_state, final_state] = self
            .boundary_states
            .map(|state| state.unwrap().map(Val::<SC>::from_canonical_u32));

        let trace = RowMajorMatrix::new(
            [initial_state.flatten(), final_state.flatten()].concat(),
            self.trace_width(),
        );

        let mut public_values = vec![Val::<SC>::zero(); VmConnectorPvs::<Val<SC>>::width()];
        *public_values.as_mut_slice().borrow_mut() = VmConnectorPvs {
            initial_pc: initial_state.pc,
            final_pc: final_state.pc,
            exit_code: final_state.exit_code,
            is_terminate: final_state.is_terminate,
        };
        AirProofInput::simple(Arc::new(self.air), trace, public_values)
    }
}

impl<F: PrimeField32> ChipUsageGetter for VmConnectorChip<F> {
    fn air_name(&self) -> String {
        "VmConnectorAir".to_string()
    }

    fn current_trace_height(&self) -> usize {
        2
    }

    fn trace_width(&self) -> usize {
        4
    }
}
