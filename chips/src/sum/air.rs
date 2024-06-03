use std::borrow::Borrow;

use afs_stark_backend::interaction::{Chip, Interaction};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, VirtualPairCol};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{
    columns::{SumGateCols, NUM_SUM_GATE_COLS},
    SumChip,
};

impl<F: Field> Chip<F> for SumChip {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: vec![VirtualPairCol::single_main(0)],
            count: VirtualPairCol::one(),
            argument_index: self.bus_input,
        }]
    }
}

impl<F> BaseAir<F> for SumChip {
    fn width(&self) -> usize {
        NUM_SUM_GATE_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for SumChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let pis = builder.public_values();
        let total_sum = pis[0];

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &SumGateCols<AB::Var> = (*local).borrow();
        let next: &SumGateCols<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();
        when_first_row.assert_eq(local.partial_sum, local.input);

        let mut when_transition = builder.when_transition();
        when_transition.assert_eq(next.partial_sum, local.partial_sum + next.input);

        let mut when_last_row = builder.when_last_row();
        when_last_row.assert_eq(local.partial_sum, total_sum);
    }
}
