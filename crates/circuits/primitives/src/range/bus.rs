use ax_stark_backend::interaction::{InteractionBuilder, InteractionType};
use p3_field::AbstractField;

/// Represents a bus for `x` where `x` must lie in the range `[0, range_max)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeCheckBus {
    pub index: usize,
    pub range_max: u32,
}

impl RangeCheckBus {
    pub const fn new(index: usize, range_max: u32) -> Self {
        Self { index, range_max }
    }

    /// Range check that `x` is in the range `[0, 2^max_bits)`.
    ///
    /// This can be used when `2^max_bits < self.range_max` **if `2 * self.range_max` is less than the field modulus**.
    pub fn range_check<T>(&self, x: impl Into<T>, max_bits: usize) -> BitsCheckBusInteraction<T> {
        debug_assert!((1 << max_bits) <= self.range_max);
        let shift = self.range_max - (1 << max_bits);
        BitsCheckBusInteraction {
            x: x.into(),
            shift,
            bus_index: self.index,
        }
    }

    pub fn send<T>(&self, x: impl Into<T>) -> RangeCheckBusInteraction<T> {
        self.push(x, InteractionType::Send)
    }

    pub fn receive<T>(&self, x: impl Into<T>) -> RangeCheckBusInteraction<T> {
        self.push(x, InteractionType::Receive)
    }

    pub fn push<T>(
        &self,
        x: impl Into<T>,
        interaction_type: InteractionType,
    ) -> RangeCheckBusInteraction<T> {
        RangeCheckBusInteraction {
            x: x.into(),
            bus_index: self.index,
            interaction_type,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BitsCheckBusInteraction<T> {
    pub x: T,
    pub shift: u32,
    pub bus_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct RangeCheckBusInteraction<T> {
    pub x: T,

    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

impl<T: AbstractField> RangeCheckBusInteraction<T> {
    /// Finalizes and sends/receives over the RangeCheck bus.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        builder.push_interaction(self.bus_index, [self.x], count, self.interaction_type);
    }
}

impl<T: AbstractField> BitsCheckBusInteraction<T> {
    /// Send interaction(s) to range check for max bits over the RangeCheck bus.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let count = count.into();
        if self.shift > 0 {
            // if 2^max_bits < range_max, then we also range check that `x + (range_max - 2^max_bits) < range_max`
            // - this will hold if `x < 2^max_bits` (necessary)
            // - if `x < range_max` then we know the integer value `x.as_canonical_u32() + (range_max - 2^max_bits) < 2*range_max`.
            //   **Assuming that `2*range_max < F::MODULUS`, then additionally knowing `x + (range_max - 2^max_bits) < range_max` implies `x < 2^max_bits`.
            builder.push_send(
                self.bus_index,
                [self.x.clone() + AB::Expr::from_canonical_u32(self.shift)],
                count.clone(),
            );
        }
        builder.push_send(self.bus_index, [self.x], count);
    }
}
