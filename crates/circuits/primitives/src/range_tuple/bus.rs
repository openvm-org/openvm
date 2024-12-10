use ax_stark_backend::{
    interaction::{InteractionBuilder, InteractionType},
    p3_field::AbstractField,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeTupleCheckerBus<const N: usize> {
    pub index: usize,
    pub sizes: [u32; N],
}

impl<const N: usize> RangeTupleCheckerBus<N> {
    pub fn new(index: usize, sizes: [u32; N]) -> Self {
        Self { index, sizes }
    }

    #[must_use]
    pub fn send<T>(&self, tuple: Vec<impl Into<T>>) -> RangeTupleCheckerBusInteraction<T> {
        self.push(tuple, InteractionType::Send)
    }

    #[must_use]
    pub fn receive<T>(&self, tuple: Vec<impl Into<T>>) -> RangeTupleCheckerBusInteraction<T> {
        self.push(tuple, InteractionType::Receive)
    }

    pub fn push<T>(
        &self,
        tuple: Vec<impl Into<T>>,
        interaction_type: InteractionType,
    ) -> RangeTupleCheckerBusInteraction<T> {
        RangeTupleCheckerBusInteraction {
            tuple: tuple.into_iter().map(|t| t.into()).collect(),
            bus_index: self.index,
            interaction_type,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RangeTupleCheckerBusInteraction<T> {
    pub tuple: Vec<T>,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

impl<T: AbstractField> RangeTupleCheckerBusInteraction<T> {
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        builder.push_interaction(self.bus_index, self.tuple, count, self.interaction_type);
    }
}
