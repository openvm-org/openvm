use afs_middleware::interaction::{Chip, Interaction};
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder, VirtualPairCol};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

pub struct SenderAir {}

impl<F: Field> Chip<F> for SenderAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        vec![Interaction::<F> {
            fields: vec![VirtualPairCol::<F>::single_main(1)],
            count: VirtualPairCol::<F>::single_main(0),
            argument_index: 0,
        }]
    }
}

impl<F: Field> BaseAir<F> for SenderAir {
    fn width(&self) -> usize {
        2
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: AirBuilderWithPublicValues + PairBuilder> Air<AB> for SenderAir {
    fn eval(&self, _builder: &mut AB) {}
}
