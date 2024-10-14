use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use p3_field::AbstractField;

use super::IsLessThanTupleAir;
use crate::is_less_than::columns::IsLessThanAuxCols;

impl IsLessThanTupleAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        less_than_aux: &[IsLessThanAuxCols<AB::Var>],
    ) {
        // We range check the limbs of lower_decomp used for each IsLessThanAir in the tuple.
        for (air, aux_cols) in self.is_less_than_airs.iter().zip_eq(less_than_aux) {
            // This range checks the limbs of lower_decomp
            air.eval_interactions(builder, aux_cols.lower_decomp.clone(), AB::F::one());
        }
    }
}
