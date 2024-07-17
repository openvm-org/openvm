use afs_stark_backend::interaction::InteractionBuilder;

use super::{
    columns::{XorLookupCols, XorLookupPreprocessedCols},
    XorLookupAir,
};

impl<const M: usize> XorLookupAir<M> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        prep_local: XorLookupPreprocessedCols<AB::Var>,
        local: XorLookupCols<AB::Var>,
    ) {
        builder.push_receive(
            self.bus_index,
            vec![prep_local.x, prep_local.y, prep_local.z],
            local.mult,
        );
    }
}
