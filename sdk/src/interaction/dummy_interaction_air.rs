//! Air with columns
//! | count | fields[..] |
//!
//! Chip will either send or receive the fields with multiplicity count.
//! The main Air has no constraints, the only constraints are specified by the Chip trait

use afs_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::{InteractionBuilder, InteractionType},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
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
    /// Send if true. Receive if false.
    pub is_send: bool,
    bus_index: usize,
    /// If true, then | count | and | fields[..] | are in separate main trace partitions.
    pub partition: bool,
}

impl DummyInteractionAir {
    pub fn new(field_width: usize, is_send: bool, bus_index: usize) -> Self {
        Self {
            field_width,
            is_send,
            bus_index,
            partition: false,
        }
    }

    pub fn partition(self) -> Self {
        Self {
            partition: true,
            ..self
        }
    }

    pub fn field_width(&self) -> usize {
        self.field_width
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for DummyInteractionAir {}
impl<F: Field> PartitionedBaseAir<F> for DummyInteractionAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        if self.partition {
            vec![1]
        } else {
            vec![]
        }
    }
    fn common_main_width(&self) -> usize {
        if self.partition {
            self.field_width
        } else {
            1 + self.field_width
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

impl<AB: InteractionBuilder + PartitionedAirBuilder> Air<AB> for DummyInteractionAir {
    fn eval(&self, builder: &mut AB) {
        let (fields, count) = if self.partition {
            let local_0 = builder.partitioned_main()[0].row_slice(0);
            let local_1 = builder.partitioned_main()[1].row_slice(0);
            let count = local_0[0];
            let fields = local_1.to_vec();
            (fields, count)
        } else {
            let main = builder.main();
            let local = main.row_slice(0);
            let count = local[DummyInteractionCols::count_col()];
            let fields: Vec<_> = (0..self.field_width)
                .map(|i| local[DummyInteractionCols::field_col(i)])
                .collect();
            (fields, count)
        };
        let interaction_type = if self.is_send {
            InteractionType::Send
        } else {
            InteractionType::Receive
        };
        builder.push_interaction(self.bus_index, fields, count, interaction_type)
    }
}
