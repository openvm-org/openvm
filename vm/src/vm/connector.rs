use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::arch::{ExecutionBus, ExecutionState};

#[derive(Debug)]
pub struct VmConnectorAir {
    pub execution_bus: ExecutionBus,
}

impl<F: Field> BaseAirWithPublicValues<F> for VmConnectorAir {}
impl<F: Field> PartitionedBaseAir<F> for VmConnectorAir {}
impl<F: Field> BaseAir<F> for VmConnectorAir {
    fn width(&self) -> usize {
        2
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(RowMajorMatrix::new_col(vec![F::zero(), F::one()]))
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for VmConnectorAir {
    fn eval(&self, builder: &mut AB) {
        // we only have interactions here, so let's jump straight to it shall we
        let main = builder.main();
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let (begin, end) = (main.row_slice(0), main.row_slice(1));
        self.execution_bus.execute(
            builder,
            AB::Expr::one() - prep_local[0], // 1 only if these are [0th, 1st] and not [1st, 0th]
            ExecutionState::new(end[0], end[1]),
            ExecutionState::new(begin[0], begin[1]),
        );
    }
}

#[derive(Debug)]
pub struct VmConnectorChip<F: PrimeField32> {
    pub air: VmConnectorAir,
    pub boundary_states: [Option<ExecutionState<F>>; 2],
}

impl<F: PrimeField32> VmConnectorChip<F> {
    pub fn new(execution_bus: ExecutionBus) -> Self {
        Self {
            air: VmConnectorAir { execution_bus },
            boundary_states: [None, None],
        }
    }

    pub fn begin(&mut self, state: ExecutionState<F>) {
        self.boundary_states[0] = Some(state);
    }

    pub fn end(&mut self, state: ExecutionState<F>) {
        self.boundary_states[1] = Some(state);
    }

    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        assert!(self.boundary_states.iter().all(|state| state.is_some()));
        RowMajorMatrix::new(
            self.boundary_states
                .iter()
                .map(|state| state.unwrap().flatten().to_vec())
                .collect::<Vec<Vec<F>>>()
                .concat(),
            2,
        )
    }
}
