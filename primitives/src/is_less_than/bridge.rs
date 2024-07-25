use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::IsLessThanAir;

impl IsLessThanAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        lower_decomp: Vec<impl Into<AB::Expr>>,
    ) {
        // we range check the limbs of the lower_bits so that we know each element
        // of lower_bits has at most limb_bits bits
        for limb in lower_decomp {
            builder.push_send(self.bus_index, vec![limb], AB::F::one());
        }
    }
}
