use afs_stark_backend::interaction::InteractionBuilder;

use crate::sub_chip::SubAirBridge;

use super::{
    columns::{XorLookupCols, XorLookupPreprocessedCols},
    XorLookupAir,
};

impl<AB: InteractionBuilder, const M: usize> SubAirBridge<AB> for XorLookupAir<M> {
    /// Local preprocessed and main trace rows.
    type View = (XorLookupPreprocessedCols<AB::Var>, XorLookupCols<AB::Var>);

    fn eval_interactions(&self, builder: &mut AB, (preprocessed, main): Self::View) {
        builder.push_receive(
            self.bus_index,
            vec![preprocessed.x, preprocessed.y, preprocessed.z],
            main.mult,
        );
    }
}
