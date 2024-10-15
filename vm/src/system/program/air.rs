use afs_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::Field;

use super::{columns::ProgramCols, ProgramAir};
use crate::system::program::columns::ProgramExecutionCols;

impl<F: Field> BaseAirWithPublicValues<F> for ProgramAir {}
impl<F: Field> PartitionedBaseAir<F> for ProgramAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![ProgramExecutionCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        1
    }
}
impl<F: Field> BaseAir<F> for ProgramAir {
    fn width(&self) -> usize {
        ProgramCols::<F>::width()
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for ProgramAir {
    fn eval(&self, builder: &mut AB) {
        self.eval_interactions(builder);
    }
}
