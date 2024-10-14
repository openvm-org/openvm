use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::SumAir;

impl SumAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        key: AB::Var,
        value: AB::Var,
        partial_sum: AB::Var,
        is_final: AB::Var,
    ) {
        // Send the final sum
        builder.push_send(self.output_bus, vec![key, partial_sum], is_final);
        builder.push_receive(self.input_bus, vec![key, value], AB::F::one());
    }
}
