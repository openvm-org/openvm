use std::{borrow::Borrow, marker::PhantomData, sync::Arc};

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    prover::types::AirProofInput,
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::arch::{ExecutionBus, ExecutionState};

#[derive(Debug, Clone)]
pub struct VmConnectorAir {
    pub execution_bus: ExecutionBus,
}

impl<F: Field> BaseAirWithPublicValues<F> for VmConnectorAir {
    fn num_public_values(&self) -> usize {
        2
    }
}
impl<F: Field> PartitionedBaseAir<F> for VmConnectorAir {}
impl<F: Field> BaseAir<F> for VmConnectorAir {
    fn width(&self) -> usize {
        2
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(RowMajorMatrix::new_col(vec![F::zero(), F::one()]))
    }
}

impl<AB: InteractionBuilder + PairBuilder + AirBuilderWithPublicValues> Air<AB> for VmConnectorAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let (begin, end) = (main.row_slice(0), main.row_slice(1));

        let begin: &ExecutionState<AB::Var> = (*begin).borrow();
        let end: &ExecutionState<AB::Var> = (*end).borrow();

        let initial_pc = builder.public_values()[0];
        let final_pc = builder.public_values()[1];

        builder.when_transition().assert_eq(begin.pc, initial_pc);
        builder.when_transition().assert_eq(end.pc, final_pc);

        self.execution_bus.execute(
            builder,
            AB::Expr::one() - prep_local[0], // 1 only if these are [0th, 1st] and not [1st, 0th]
            ExecutionState::new(end.pc, end.timestamp),
            ExecutionState::new(begin.pc, begin.timestamp),
        );
    }
}

#[derive(Debug)]
pub struct VmConnectorChip<F: PrimeField32> {
    pub air: VmConnectorAir,
    pub boundary_states: [Option<ExecutionState<u32>>; 2],
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> VmConnectorChip<F> {
    pub fn new(execution_bus: ExecutionBus) -> Self {
        Self {
            air: VmConnectorAir { execution_bus },
            boundary_states: [None, None],
            _marker: PhantomData,
        }
    }

    pub fn begin(&mut self, state: ExecutionState<u32>) {
        self.boundary_states[0] = Some(state);
    }

    pub fn end(&mut self, state: ExecutionState<u32>) {
        self.boundary_states[1] = Some(state);
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
        let boundary_states = self
            .boundary_states
            .into_iter()
            .map(|state| state.unwrap().map(Val::<SC>::from_canonical_u32))
            .collect::<Vec<_>>();

        let trace = RowMajorMatrix::new(
            boundary_states
                .iter()
                .flat_map(|state| state.flatten())
                .collect::<Vec<_>>(),
            2,
        );
        let public_values = vec![boundary_states[0].pc, boundary_states[1].pc];
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
        2
    }
}
