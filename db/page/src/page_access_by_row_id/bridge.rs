use afs_stark_backend::interaction::InteractionBuilder;

use super::air::PageAccessByRowIdAir;

impl PageAccessByRowIdAir {
    // receives: ([row_id] | [page]) mult times
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page_row_with_row_id: Vec<AB::Var>,
        mult: AB::Var,
    ) {
        builder.push_receive(self.bus_index, page_row_with_row_id, mult);
    }
}
