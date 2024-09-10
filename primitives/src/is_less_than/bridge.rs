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
        let mut bits_remaining = self.max_bits;
        // we range check the limbs of the lower_decomp so that we know each element
        // of lower_decomp has at most `range_max_bits` bits or less if `max_bits % range_max_bits != 0`.
        for limb in lower_decomp {
            let range_bits = bits_remaining.min(self.range_max_bits());
            self.bus
                .range_check(limb, range_bits)
                .eval(builder, count.clone());
            bits_remaining = bits_remaining.saturating_sub(self.range_max_bits());
        }
    }
}
