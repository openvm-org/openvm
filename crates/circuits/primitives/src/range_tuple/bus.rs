use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_field::FieldAlgebra,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeTupleCheckerBus<const N: usize> {
    pub inner: LookupBus,
    pub sizes: [u32; N],
}

impl<const N: usize> RangeTupleCheckerBus<N> {
    pub fn new(index: BusIndex, sizes: [u32; N]) -> Self {
        let mut product = 1u32;
        for &size in sizes.iter() {
            product = product
                .checked_mul(size)
                .expect("The number of the range tuple checker rows is too large");
        }
        Self {
            inner: LookupBus::new(index),
            sizes,
        }
    }

    #[must_use]
    pub fn send<T>(&self, tuple: Vec<impl Into<T>>) -> RangeTupleCheckerBusInteraction<T> {
        self.push(tuple, true)
    }

    #[must_use]
    pub fn receive<T>(&self, tuple: Vec<impl Into<T>>) -> RangeTupleCheckerBusInteraction<T> {
        self.push(tuple, false)
    }

    pub fn push<T>(
        &self,
        tuple: Vec<impl Into<T>>,
        is_lookup: bool,
    ) -> RangeTupleCheckerBusInteraction<T> {
        RangeTupleCheckerBusInteraction {
            tuple: tuple.into_iter().map(|t| t.into()).collect(),
            bus: self.inner,
            is_lookup,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RangeTupleCheckerBusInteraction<T> {
    pub tuple: Vec<T>,
    pub bus: LookupBus,
    pub is_lookup: bool,
}

impl<T: FieldAlgebra> RangeTupleCheckerBusInteraction<T> {
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        if self.is_lookup {
            self.bus.lookup_key(builder, self.tuple, count);
        } else {
            self.bus.add_key_with_lookups(builder, self.tuple, count);
        }
    }
}
