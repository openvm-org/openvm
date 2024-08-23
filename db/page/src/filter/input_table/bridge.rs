use afs_stark_backend::interaction::InteractionBuilder;

use super::FilterInputTableAir;

impl FilterInputTableAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        idx: Vec<AB::Var>,
        data: Vec<AB::Var>,
        send_row: AB::Var,
    ) {
        let page_blob = idx.into_iter().chain(data);

        builder.push_send(self.page_bus_index, page_blob, send_row);
    }
}
