use afs_stark_backend::interaction::InteractionBuilder;

use super::AssertLessThanAir;

impl<const AUX_LEN: usize> AssertLessThanAir<AUX_LEN> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        lower_decomp: [impl Into<AB::Expr>; AUX_LEN],
        count: impl Into<AB::Expr>,
    ) {
        let count = count.into();
        // we range check the limbs of the lower_decomp so that we know each element
        // of lower_decomp has at most `decomp` bits
        for limb in lower_decomp {
            self.bus
                .range_check(limb, self.decomp)
                .eval(builder, count.clone());
        }
    }
}
