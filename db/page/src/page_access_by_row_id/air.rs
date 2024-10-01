use std::{borrow::Borrow, iter};

use afs_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::PageAccessByRowIdAuxCols;

/// Air to access specific rows of the page by row id.
/// The Air automatically builds the row id, starting from 0 on the first row.
#[derive(Copy, Clone, Debug)]
pub struct PageAccessByRowIdAir {
    pub bus_index: usize,
    pub page_width: usize,
}

impl PageAccessByRowIdAir {
    pub fn new(bus_index: usize, page_width: usize) -> Self {
        Self {
            bus_index,
            page_width,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn page_width(&self) -> usize {
        self.page_width
    }

    pub fn air_width(&self) -> usize {
        2 + self.page_width
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for PageAccessByRowIdAir {}
impl<F: Field> PartitionedBaseAir<F> for PageAccessByRowIdAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![self.page_width()]
    }
    fn common_main_width(&self) -> usize {
        self.air_width() - self.page_width()
    }
}
impl<F: Field> BaseAir<F> for PageAccessByRowIdAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for PageAccessByRowIdAir {
    fn eval(&self, builder: &mut AB) {
        // Choosing the second partition of the trace, which looks like (index, mult)
        let main: &<AB as AirBuilder>::M = &builder.partitioned_main()[1];

        let (local_slc, next_slc) = (main.row_slice(0), main.row_slice(1));
        let local: &PageAccessByRowIdAuxCols<AB::Var> = (*local_slc).borrow();
        let next: &PageAccessByRowIdAuxCols<AB::Var> = (*next_slc).borrow();
        let row_id = local.row_id;
        let mult = local.mult;
        let next_row_id = next.row_id;
        drop(local_slc);
        drop(next_slc);

        // Ensuring index starts at 0
        builder.when_first_row().assert_eq(row_id, AB::Expr::zero());

        // Ensuring that index goes up by 1 every row
        builder
            .when_transition()
            .assert_eq(row_id + AB::Expr::one(), next_row_id);

        let page_main = &builder.partitioned_main()[0];
        let page_local = page_main.row_slice(0);
        let page_row_with_row_id = iter::once(row_id)
            .chain(page_local.iter().copied())
            .collect_vec();
        drop(page_local);
        self.eval_interactions(builder, page_row_with_row_id, mult);
    }
}
