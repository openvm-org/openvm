//! Air with columns
//! | count | fields[..] |
//!
//! Chip will either send or receive the fields with multiplicity count.
//! The main Air has no constraints, the only constraints are specified by the Chip trait

use afs_stark_backend::interaction::{InteractionBuilder, InteractionType};
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

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
}

impl<F: Field> BaseAir<F> for DummyInteractionAir {
    fn width(&self) -> usize {
        1 + self.field_width
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: InteractionBuilder> Air<AB> for DummyInteractionAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let count = local[DummyInteractionCols::count_col()];
        let fields: Vec<_> = (0..self.field_width)
            .map(|i| local[DummyInteractionCols::field_col(i)])
            .collect();
        let interaction_type = if self.is_send {
            InteractionType::Send
        } else {
            InteractionType::Receive
        };
        builder.push_interaction(self.bus_index, fields, count, interaction_type)
    }
}
