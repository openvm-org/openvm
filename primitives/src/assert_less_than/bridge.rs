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
        let lower_decomp = lower_decomp.map(|limb| limb.into());

        // we range check the limbs of the lower_decomp so that we know each element
        // of lower_decomp has the correct number of bits
        for (i, limb) in lower_decomp.iter().enumerate() {
            if i == lower_decomp.len() - 1 && self.max_bits % self.bus.range_max_bits != 0 {
                // the last limb might have fewer than `bus.range_max_bits` bits
                self.bus
                    .range_check(limb.clone(), self.max_bits % self.bus.range_max_bits)
                    .eval(builder, count.clone());
            } else {
                // the other limbs must have exactly `bus.range_max_bits` bits
                self.bus
                    .range_check(limb.clone(), self.bus.range_max_bits)
                    .eval(builder, count.clone());
            }
        }
    }
}
