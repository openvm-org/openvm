use afs_stark_backend::interaction::InteractionBuilder;

use crate::sub_chip::SubAirBridge;

use super::{columns::RangeGateCols, RangeCheckerGateAir};

impl<AB: InteractionBuilder> SubAirBridge<AB> for RangeCheckerGateAir {
    type View = RangeGateCols<AB::Var>;

    fn eval_interactions(&self, builder: &mut AB, local: Self::View) {
        builder.push_receive(self.bus_index, vec![local.counter], local.mult);
    }
}
