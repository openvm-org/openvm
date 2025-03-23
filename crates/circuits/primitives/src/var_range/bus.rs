use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_field::FieldAlgebra,
};

// Represents a bus for (x, bits) where either (x, bits) = (0, 0) or
// x is in [0, 2^bits) and bits is in [1, range_max_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariableRangeCheckerBus {
    pub inner: LookupBus,
    pub range_max_bits: usize,
}

impl VariableRangeCheckerBus {
    pub const fn new(index: BusIndex, range_max_bits: usize) -> Self {
        Self {
            inner: LookupBus::new(index),
            range_max_bits,
        }
    }

    #[inline(always)]
    pub fn index(&self) -> BusIndex {
        self.inner.index
    }

    #[must_use]
    pub fn send<T>(
        &self,
        value: impl Into<T>,
        max_bits: impl Into<T>,
    ) -> VariableRangeCheckerBusInteraction<T> {
        self.push(value, max_bits, true)
    }

    #[must_use]
    pub fn receive<T>(
        &self,
        value: impl Into<T>,
        max_bits: impl Into<T>,
    ) -> VariableRangeCheckerBusInteraction<T> {
        self.push(value, max_bits, false)
    }

    // Equivalent to `self.send(value, max_bits)` where max_bits is a usize constant
    #[must_use]
    pub fn range_check<T>(
        &self,
        value: impl Into<T>,
        max_bits: usize,
    ) -> VariableRangeCheckerBusInteraction<T>
    where
        T: FieldAlgebra,
    {
        debug_assert!(max_bits <= self.range_max_bits);
        self.push(value, T::from_canonical_usize(max_bits), true)
    }

    pub fn push<T>(
        &self,
        value: impl Into<T>,
        max_bits: impl Into<T>,
        is_lookup: bool,
    ) -> VariableRangeCheckerBusInteraction<T> {
        VariableRangeCheckerBusInteraction {
            value: value.into(),
            max_bits: max_bits.into(),
            bus: self.inner,
            is_lookup,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VariableRangeCheckerBusInteraction<T> {
    pub value: T,
    pub max_bits: T,
    pub bus: LookupBus,
    pub is_lookup: bool,
}

impl<T: FieldAlgebra> VariableRangeCheckerBusInteraction<T> {
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let key = [self.value, self.max_bits];
        if self.is_lookup {
            self.bus.lookup_key(builder, key, count);
        } else {
            self.bus.add_key_with_lookups(builder, key, count);
        }
    }
}
