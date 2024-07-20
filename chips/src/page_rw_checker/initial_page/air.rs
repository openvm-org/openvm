use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::PageReadAir;
use crate::common::page_cols::PageCols;
use crate::sub_chip::{AirConfig, SubAir};

impl<F: Field> BaseAir<F> for PageReadAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for PageReadAir {
    type Cols<T> = PageCols<T>;
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for PageReadAir {
    fn eval(&self, builder: &mut AB) {
        let cols = PageCols::<AB::Var>::from_slice(
            &builder.main().row_slice(0),
            self.idx_len,
            self.data_len,
        );
        self.eval_interactions(builder, &cols);
    }
}

impl<AB: AirBuilder> SubAir<AB> for PageReadAir
where
    AB::M: Clone,
{
    type IoView = PageCols<AB::Var>;
    type AuxView = ();

    fn eval(&self, _builder: &mut AB, _io: Self::IoView, _aux: Self::AuxView) {}
}
