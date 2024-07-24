use afs_stark_backend::interaction::InteractionBuilder;

use super::RangeCheckerGateAir;

impl RangeCheckerGateAir {
    /// `counter` is the value to lookup, `mult` is the multiplicity
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        counter: impl Into<AB::Expr>,
        mult: impl Into<AB::Expr>,
    ) {
        builder.push_receive(self.bus_index, vec![counter], mult);
    }
}
