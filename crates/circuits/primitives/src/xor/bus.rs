use openvm_stark_backend::{
    interaction::{InteractionBuilder, LookupBus},
    p3_field::FieldAlgebra,
};

/// Represents a bus for `(x, y, x ^ y)` identified by a unique bus index (`usize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XorBus(pub LookupBus);

impl XorBus {
    pub fn send<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        x_xor_y: impl Into<T>,
    ) -> XorBusInteraction<T> {
        self.push(x, y, x_xor_y, true)
    }

    pub fn receive<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        x_xor_y: impl Into<T>,
    ) -> XorBusInteraction<T> {
        self.push(x, y, x_xor_y, false)
    }

    pub fn push<T>(
        &self,
        x: impl Into<T>,
        y: impl Into<T>,
        x_xor_y: impl Into<T>,
        is_lookup: bool,
    ) -> XorBusInteraction<T> {
        XorBusInteraction {
            x: x.into(),
            y: y.into(),
            x_xor_y: x_xor_y.into(),
            bus: self.0,
            is_lookup,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct XorBusInteraction<T> {
    pub x: T,
    pub y: T,
    pub x_xor_y: T,

    pub bus: LookupBus,
    pub is_lookup: bool,
}

impl<T: FieldAlgebra> XorBusInteraction<T> {
    /// Finalizes and sends/receives over the Xor bus.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let key = [self.x, self.y, self.x_xor_y];
        if self.is_lookup {
            self.bus.lookup_key(builder, key, count);
        } else {
            self.bus.add_key_with_lookups(builder, key, count);
        }
    }
}
