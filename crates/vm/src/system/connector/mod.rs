use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    sync::Arc,
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::UsizeOpcode;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter, Stateful,
};
use serde::{Deserialize, Serialize};

use crate::{
    arch::{instructions::SystemOpcode::TERMINATE, ExecutionBus, ExecutionState},
    system::program::ProgramBus,
};

#[cfg(test)]
mod tests;

/// When a program hasn't terminated. There is no constraints on the exit code.
/// But we will use this value when generating the proof.
pub const DEFAULT_SUSPEND_EXIT_CODE: u32 = 42;

#[derive(Debug, Clone, Copy)]
pub struct VmConnectorAir {
    pub execution_bus: ExecutionBus,
    pub program_bus: ProgramBus,
}

#[derive(Debug, Clone, Copy, AlignedBorrow)]
#[repr(C)]
pub struct VmConnectorPvs<F> {
    /// The initial PC of this segment.
    pub initial_pc: F,
    /// The final PC of this segment.
    pub final_pc: F,
    /// The exit code of the whole program. 0 means exited normally. This is only meaningful when
    /// `is_terminate` is 1.
    pub exit_code: F,
    /// Whether the whole program is terminated. 0 means not terminated. 1 means terminated.
    /// Only the last segment of an execution can have `is_terminate` = 1.
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
        Some(RowMajorMatrix::new_col(vec![F::ZERO, F::ONE]))
    }
}

#[derive(Debug, Copy, Clone, AlignedBorrow, Serialize, Deserialize)]
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
            AB::Expr::ONE - prep_local[0], // 1 only if these are [0th, 1st] and not [1st, 0th]
            ExecutionState::new(end.pc, end.timestamp),
            ExecutionState::new(begin.pc, begin.timestamp),
        );
        self.program_bus.send_instruction(
            builder,
            end.pc,
            AB::Expr::from_canonical_usize(TERMINATE.global_opcode().as_usize()),
            [AB::Expr::ZERO, AB::Expr::ZERO, end.exit_code.into()],
            (AB::Expr::ONE - prep_local[0]) * end.is_terminate,
        );
    }
}

#[derive(Debug)]
pub struct VmConnectorChip<F> {
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
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let [initial_state, final_state] = self
            .boundary_states
            .map(|state| state.unwrap().map(Val::<SC>::from_canonical_u32));

        let trace = RowMajorMatrix::new(
            [initial_state.flatten(), final_state.flatten()].concat(),
            self.trace_width(),
        );

        let mut public_values = Val::<SC>::zero_vec(VmConnectorPvs::<Val<SC>>::width());
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

    fn constant_trace_height(&self) -> Option<usize> {
        Some(2)
    }

    fn current_trace_height(&self) -> usize {
        2
    }

    fn trace_width(&self) -> usize {
        4
    }
}

impl<F: PrimeField32> Stateful<Vec<u8>> for VmConnectorChip<F> {
    fn load_state(&mut self, state: Vec<u8>) {
        self.boundary_states = bitcode::deserialize(&state).unwrap();
    }

    fn store_state(&self) -> Vec<u8> {
        bitcode::serialize(&self.boundary_states).unwrap()
    }
}
