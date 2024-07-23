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
        for i in 0..lower_decomp.len() {
            if self.max_bits % self.decomp != 0 && i == lower_decomp.len() - 2 {
                // In case limb_bits does not divide decomp, we can skip this range check since,
                // in the next iteration, we range check the same thing shifted to the left
                continue;
            }

            builder.push_send(self.bus_index, vec![lower_decomp[i]], AB::F::one());
        }
    }
}
