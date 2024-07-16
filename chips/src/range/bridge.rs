use afs_stark_backend::interaction::InteractionBuilder;

use crate::sub_chip::SubAirBridge;

use super::{
    columns::{RangeCols, RangePreprocessedCols},
    RangeCheckerAir,
};

impl<AB: InteractionBuilder> SubAirBridge<AB> for RangeCheckerAir {
    type View = (RangePreprocessedCols<AB::Var>, RangeCols<AB::Var>);

    fn eval_interactions(&self, builder: &mut AB, (preprocessed, main): Self::View) {
        builder.push_receive(self.bus_index, vec![preprocessed.counter], main.mult);
    }
}
