use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::StarkGenericConfig;

use afs_stark_backend::{interaction::InteractionBuilder, rap::AnyRap};

use crate::{
    arch::{
        chips::{MachineChip, OpCodeExecutor},
        columns::ExecutionState,
    },
    cpu::trace::Instruction,
};

pub trait AirCols<T, A> {
    fn from_slice(slice: &[T], air: &A) -> Self;
    fn flatten(&self, air: &A) -> Vec<T>;
    fn get_width(air: &A) -> usize;
}

pub trait SimpleMachineFeature: Sized + Sync {
    type Dependency<F>;
    type Event<F>;
    type Cols<T>: AirCols<T, Self>;

    fn execute<F: PrimeField32>(
        &self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> (Self::Event<F>, ExecutionState<usize>);

    fn generate_trace_row<F: PrimeField32>(
        &self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
        next_state: ExecutionState<usize>,
        event: Self::Event<F>,
    ) -> Self::Cols<F>;

    fn empty_row<F: PrimeField32>(&self) -> Self::Cols<F>;

    fn eval_constraints<AB: InteractionBuilder + AirBuilderWithPublicValues>(
        &self,
        builder: &mut AB,
        local: Self::Cols<AB::Var>,
    );
}

struct SimpleMachineChipRow<A: SimpleMachineFeature, F> {
    instruction: Instruction<F>,
    prev_state: ExecutionState<usize>,
    next_state: ExecutionState<usize>,
    event: A::Event<F>,
}

struct SimpleMachineAir<A: SimpleMachineFeature> {
    air: A,
}

impl<A: SimpleMachineFeature, F: Field> BaseAir<F> for SimpleMachineAir<A> {
    fn width(&self) -> usize {
        A::Cols::<F>::get_width(&self.air)
    }
}

impl<A: SimpleMachineFeature, AB: InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for SimpleMachineAir<A>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = A::Cols::from_slice(&local, &self.air);
        self.air.eval_constraints(builder, local);
    }
}

pub struct SimpleMachineChip<A: SimpleMachineFeature + 'static, F: PrimeField32> {
    dependency: A::Dependency<F>,
    simple_machine_air: SimpleMachineAir<A>,
    raw_rows: Vec<SimpleMachineChipRow<A, F>>,
}

impl<A: SimpleMachineFeature, F: PrimeField32> SimpleMachineChip<A, F> {
    pub fn new(air: A, dependency: A::Dependency<F>) -> Self {
        Self {
            dependency,
            simple_machine_air: SimpleMachineAir { air },
            raw_rows: vec![],
        }
    }
}

impl<A: SimpleMachineFeature, F: PrimeField32> OpCodeExecutor<F> for SimpleMachineChip<A, F> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let (event, next_state) = self.simple_machine_air.air.execute(instruction, prev_state);
        self.raw_rows.push(SimpleMachineChipRow {
            instruction: instruction.clone(),
            prev_state,
            next_state,
            event,
        });
        next_state
    }
}

impl<A: SimpleMachineFeature, F: PrimeField32> MachineChip<F> for SimpleMachineChip<A, F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        let mut num_rows: usize = 0;
        for SimpleMachineChipRow {
            instruction,
            prev_state,
            next_state,
            event,
        } in self.raw_rows.drain(..)
        {
            rows.extend(
                self.simple_machine_air
                    .air
                    .generate_trace_row(&instruction, prev_state, next_state, event)
                    .flatten(&self.simple_machine_air.air),
            );
            num_rows += 1;
        }
        while !num_rows.is_power_of_two() {
            rows.extend(
                self.simple_machine_air
                    .air
                    .empty_row::<F>()
                    .flatten(&self.simple_machine_air.air),
            );
        }
        RowMajorMatrix::new(rows, A::Cols::<F>::get_width(&self.simple_machine_air.air))
    }

    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC> {
        &self.simple_machine_air
    }
}
