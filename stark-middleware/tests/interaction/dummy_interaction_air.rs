use afs_middleware::interaction::{Chip, Interaction};
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder, VirtualPairCol};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

pub struct DummyInteractionAir {
    // Send if true. Receive if false.
    pub is_send: bool,
}

impl<F: Field> Chip<F> for DummyInteractionAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        if self.is_send {
            vec![Interaction::<F> {
                fields: vec![VirtualPairCol::<F>::single_main(1)],
                count: VirtualPairCol::<F>::single_main(0),
                argument_index: 0,
            }]
        } else {
            vec![]
        }
    }
    fn receives(&self) -> Vec<Interaction<F>> {
        if !self.is_send {
            vec![Interaction::<F> {
                fields: vec![VirtualPairCol::<F>::single_main(1)],
                count: VirtualPairCol::<F>::single_main(0),
                argument_index: 0,
            }]
        } else {
            vec![]
        }
    }
}

impl<F: Field> BaseAir<F> for DummyInteractionAir {
    fn width(&self) -> usize {
        2
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: AirBuilderWithPublicValues + PairBuilder> Air<AB> for DummyInteractionAir {
    fn eval(&self, _builder: &mut AB) {}
}
