use afs_stark_backend::interaction::InteractionBuilder;

use super::IsLessThanAir;

impl IsLessThanAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        lower_decomp: Vec<impl Into<AB::Expr>>,
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
