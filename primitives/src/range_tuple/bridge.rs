use afs_stark_backend::interaction::InteractionBuilder;

use super::RangeTupleCheckerAir;

impl RangeTupleCheckerAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        counters: Vec<impl Into<AB::Expr>>,
        mult: impl Into<AB::Expr>,
    ) {
        builder.push_receive(
            self.bus_index,
            counters.into_iter().collect::<Vec<_>>(),
            mult,
        );
    }
}
