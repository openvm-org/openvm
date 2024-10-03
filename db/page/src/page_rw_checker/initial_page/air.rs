use afs_primitives::sub_chip::{AirConfig, SubAir};
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::common::page_cols::PageCols;

/// AIR where the entire trace is assumed to be a cached trace for the page.
/// Every row is sent to the `page_bus` as a single interaction.
#[derive(Copy, Clone, Debug)]
pub struct PageReadAir {
    pub page_bus: usize,
    pub idx_len: usize,
    pub data_len: usize,
}

impl PageReadAir {
    pub fn new(page_bus: usize, idx_len: usize, data_len: usize) -> Self {
        Self {
            page_bus,
            idx_len,
            data_len,
        }
    }

    pub fn air_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for PageReadAir {}
impl<F: Field> PartitionedBaseAir<F> for PageReadAir {}
impl<F: Field> BaseAir<F> for PageReadAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for PageReadAir {
    type Cols<T> = PageCols<T>;
}

impl<AB: InteractionBuilder> Air<AB> for PageReadAir {
    fn eval(&self, builder: &mut AB) {
        let page = PageCols::<AB::Var>::from_slice(
            &builder.main().row_slice(0),
            self.idx_len,
            self.data_len,
        );
        SubAir::eval(self, builder, page, ());
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for PageReadAir {
    type IoView = PageCols<AB::Var>;
    type AuxView = ();

    fn eval(&self, builder: &mut AB, page: Self::IoView, _aux: Self::AuxView) {
        self.eval_interactions(builder, page);
    }
}
