use afs_stark_backend::interaction::{InteractionBuilder, InteractionType};
use p3_field::AbstractField;

/// Represents a bus for `x` where `x` must lie in the range `[0, 2^range_max_bits)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariableRangeCheckBus {
    pub index: usize,
    pub range_max_bits: u32,
}

impl VariableRangeCheckBus {
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
    ) -> VariableRangeCheckBusInteraction<T> {
        self.push(value, max_bits, InteractionType::Send)
    }

    pub fn receive<T>(
        &self,
        value: impl Into<T>,
        max_bits: impl Into<T>,
    ) -> VariableRangeCheckBusInteraction<T> {
        self.push(value, max_bits, InteractionType::Receive)
    }

    pub fn push<T>(
        &self,
        value: impl Into<T>,
        max_bits: impl Into<T>,
        interaction_type: InteractionType,
    ) -> VariableRangeCheckBusInteraction<T> {
        VariableRangeCheckBusInteraction {
            value: value.into(),
            max_bits: max_bits.into(),
            bus_index: self.index,
            interaction_type,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VariableRangeCheckBusInteraction<T> {
    pub value: T,
    pub max_bits: T,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

impl<T: AbstractField> VariableRangeCheckBusInteraction<T> {
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
