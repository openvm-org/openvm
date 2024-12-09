use ax_stark_backend::interaction::{InteractionBuilder, InteractionType};
use ax_stark_backend::p3_field::AbstractField;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitwiseOperationLookupBus {
    pub index: usize,
}

impl BitwiseOperationLookupBus {
    pub const fn new(index: usize) -> Self {
        Self { index }
    }

    #[must_use]
    pub fn send_range<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
    ) -> BitwiseOperationLookupBusInteraction<T>
    where
        T: AbstractField,
    {
        self.push(x, y, T::ZERO, T::ZERO, InteractionType::Send)
    }

    #[must_use]
    pub fn send_xor<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        z: impl Into<T>,
    ) -> BitwiseOperationLookupBusInteraction<T>
    where
        T: AbstractField,
    {
        self.push(x, y, z, T::ONE, InteractionType::Send)
    }

    #[must_use]
    pub fn receive<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        z: impl Into<T>,
        op: impl Into<T>,
    ) -> BitwiseOperationLookupBusInteraction<T> {
        self.push(x, y, z, op, InteractionType::Receive)
    }

    pub fn push<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        z: impl Into<T>,
        op: impl Into<T>,
        interaction_type: InteractionType,
    ) -> BitwiseOperationLookupBusInteraction<T> {
        BitwiseOperationLookupBusInteraction {
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
pub struct BitwiseOperationLookupBusInteraction<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub op: T,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

impl<T: AbstractField> BitwiseOperationLookupBusInteraction<T> {
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
