//! Air with columns
//! | count | fields[..] |
//!
//! Chip will either send or receive the fields with multiplicity count.
//! The main Air has no constraints, the only constraints are specified by the Chip trait

use afs_middleware::interaction::{Chip, Interaction};
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder, VirtualPairCol};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

pub struct DummyInteractionCols;
impl DummyInteractionCols {
    pub fn count_col() -> usize {
        0
    }
    pub fn field_col(field_idx: usize) -> usize {
        field_idx + 1
    }
}

pub struct DummyInteractionAir {
    field_width: usize,
    // Send if true. Receive if false.
    pub is_send: bool,
    bus_index: usize,
}

impl DummyInteractionAir {
    pub fn new(field_width: usize, is_send: bool, bus_index: usize) -> Self {
        Self {
            field_width,
            is_send,
            bus_index,
        }
    }

    pub fn field_width(&self) -> usize {
        self.field_width
    }

    fn interactions<F: Field>(&self) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: (0..self.field_width)
                .map(|i| VirtualPairCol::single_main(DummyInteractionCols::field_col(i)))
                .collect(),
            count: VirtualPairCol::single_main(DummyInteractionCols::count_col()),
            argument_index: self.bus_index,
        }]
    }
}

impl<F: Field> Chip<F> for DummyInteractionAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        if self.is_send {
            self.interactions()
        } else {
            vec![]
        }
    }
    fn receives(&self) -> Vec<Interaction<F>> {
        if !self.is_send {
            self.interactions()
        } else {
            vec![]
        }
    }
}

impl<F: Field> BaseAir<F> for DummyInteractionAir {
    fn width(&self) -> usize {
        1 + self.field_width
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: AirBuilderWithPublicValues + PairBuilder> Air<AB> for DummyInteractionAir {
    fn eval(&self, _builder: &mut AB) {}
}
