use openvm_stark_backend::{
    interaction::{InteractionBuilder, InteractionType},
    p3_field::AbstractField,
};

/// Represents a bus for `(x, y, x ^ y)` identified by a unique bus index (`usize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XorBus(pub usize);

impl XorBus {
    pub fn send<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        x_xor_y: impl Into<T>,
    ) -> XorBusInteraction<T> {
        self.push(x, y, x_xor_y, InteractionType::Send)
    }

    pub fn receive<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        x_xor_y: impl Into<T>,
    ) -> XorBusInteraction<T> {
        self.push(x, y, x_xor_y, InteractionType::Receive)
    }

    pub fn push<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        x_xor_y: impl Into<T>,
        interaction_type: InteractionType,
    ) -> XorBusInteraction<T> {
        XorBusInteraction {
            x: x.into(),
            y: y.into(),
            x_xor_y: x_xor_y.into(),
            bus_index: self.0,
            interaction_type,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct XorBusInteraction<T> {
    pub x: T,
    pub y: T,
    pub x_xor_y: T,

    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

impl<T: AbstractField> XorBusInteraction<T> {
    /// Finalizes and sends/receives over the Xor bus.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        builder.push_interaction(
            self.bus_index,
            [self.x, self.y, self.x_xor_y],
            count,
            self.interaction_type,
        );
    }
}
