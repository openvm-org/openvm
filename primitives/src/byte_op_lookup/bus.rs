use afs_stark_backend::interaction::{InteractionBuilder, InteractionType};
use p3_field::AbstractField;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteOperationLookupBus {
    pub index: usize,
}

impl ByteOperationLookupBus {
    pub const fn new(index: usize) -> Self {
        Self { index }
    }

    #[must_use]
    pub fn send<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        z: impl Into<T>,
        op: impl Into<T>,
    ) -> ByteOperationLookupBusInteraction<T> {
        self.push(x, y, z, op, InteractionType::Send)
    }

    #[must_use]
    pub fn receive<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        z: impl Into<T>,
        op: impl Into<T>,
    ) -> ByteOperationLookupBusInteraction<T> {
        self.push(x, y, z, op, InteractionType::Receive)
    }

    pub fn push<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        z: impl Into<T>,
        op: impl Into<T>,
        interaction_type: InteractionType,
    ) -> ByteOperationLookupBusInteraction<T> {
        ByteOperationLookupBusInteraction {
            x: x.into(),
            y: y.into(),
            z: z.into(),
            op: op.into(),
            bus_index: self.index,
            interaction_type,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ByteOperationLookupBusInteraction<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub op: T,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

impl<T: AbstractField> ByteOperationLookupBusInteraction<T> {
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        builder.push_interaction(
            self.bus_index,
            [self.x, self.y, self.z, self.op],
            count,
            self.interaction_type,
        );
    }
}
