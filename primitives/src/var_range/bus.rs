use afs_stark_backend::interaction::{InteractionBuilder, InteractionType};
use p3_field::AbstractField;

// Represents a bus for (x, bits) where either (x, bits) = (0, 0) or
// x is in [0, 2^bits) and bits is in [1, range_max_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariableRangeCheckerBus {
    pub index: usize,
    pub range_max_bits: u32,
}

impl VariableRangeCheckerBus {
    pub const fn new(index: usize, range_max_bits: u32) -> Self {
        Self {
            index,
            range_max_bits,
        }
    }

    pub fn send<T>(
        &self,
        value: impl Into<T>,
        max_bits: impl Into<T>,
    ) -> VariableRangeCheckerBusInteraction<T> {
        self.push(value, max_bits, InteractionType::Send)
    }

    pub fn receive<T>(
        &self,
        value: impl Into<T>,
        max_bits: impl Into<T>,
    ) -> VariableRangeCheckerBusInteraction<T> {
        self.push(value, max_bits, InteractionType::Receive)
    }

    pub fn push<T>(
        &self,
        value: impl Into<T>,
        max_bits: impl Into<T>,
        interaction_type: InteractionType,
    ) -> VariableRangeCheckerBusInteraction<T> {
        VariableRangeCheckerBusInteraction {
            value: value.into(),
            max_bits: max_bits.into(),
            bus_index: self.index,
            interaction_type,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VariableRangeCheckerBusInteraction<T> {
    pub value: T,
    pub max_bits: T,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

impl<T: AbstractField> VariableRangeCheckerBusInteraction<T> {
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        builder.push_interaction(
            self.bus_index,
            [self.value, self.max_bits],
            count,
            self.interaction_type,
        );
    }
}
