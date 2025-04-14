use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_field::{FieldAlgebra, PrimeField32},
};

/// Represents a bus for `x` where `x` must lie in the range `[0, range_max)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeCheckBus {
    pub inner: LookupBus,
    pub range_max: u32,
}

impl RangeCheckBus {
    pub const fn new(index: BusIndex, range_max: u32) -> Self {
        Self {
            inner: LookupBus::new(index),
            range_max,
        }
    }

    /// Range check that `x` is in the range `[0, 2^max_bits)`.
    ///
    /// This can be used when `2^max_bits < self.range_max` **if `2 * self.range_max` is less than
    /// the field modulus**.
    pub fn range_check<T: FieldAlgebra>(
        &self,
        x: impl Into<T>,
        max_bits: usize,
    ) -> BitsCheckBusInteraction<T>
    where
        T::F: PrimeField32,
    {
        debug_assert!((1 << max_bits) <= self.range_max);
        debug_assert!(self.range_max < T::F::ORDER_U32 / 2);
        let shift = self.range_max - (1 << max_bits);
        BitsCheckBusInteraction {
            x: x.into(),
            shift,
            bus: self.inner,
        }
    }

    pub fn send<T>(&self, x: impl Into<T>) -> RangeCheckBusInteraction<T> {
        self.push(x, true)
    }

    pub fn receive<T>(&self, x: impl Into<T>) -> RangeCheckBusInteraction<T> {
        self.push(x, false)
    }

    pub fn push<T>(&self, x: impl Into<T>, is_lookup: bool) -> RangeCheckBusInteraction<T> {
        RangeCheckBusInteraction {
            x: x.into(),
            bus: self.inner,
            is_lookup,
        }
    }

    pub fn index(&self) -> BusIndex {
        self.inner.index
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BitsCheckBusInteraction<T> {
    pub x: T,
    pub shift: u32,
    pub bus: LookupBus,
}

#[derive(Clone, Copy, Debug)]
pub struct RangeCheckBusInteraction<T> {
    pub x: T,

    pub bus: LookupBus,
    pub is_lookup: bool,
}

impl<T: FieldAlgebra> RangeCheckBusInteraction<T> {
    /// Finalizes and sends/receives over the RangeCheck bus.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        if self.is_lookup {
            self.bus.lookup_key(builder, [self.x], count);
        } else {
            self.bus.add_key_with_lookups(builder, [self.x], count);
        }
    }
}

impl<T: FieldAlgebra> BitsCheckBusInteraction<T> {
    /// Send interaction(s) to range check for max bits over the RangeCheck bus.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let count = count.into();
        if self.shift > 0 {
            // if 2^max_bits < range_max, then we also range check that `x + (range_max -
            // 2^max_bits) < range_max`
            // - this will hold if `x < 2^max_bits` (necessary)
            // - if `x < range_max` then we know the integer value `x.as_canonical_u32() +
            //   (range_max - 2^max_bits) < 2*range_max`. **Assuming that `2*range_max <
            //   F::MODULUS`, then additionally knowing `x + (range_max - 2^max_bits) < range_max`
            //   implies `x < 2^max_bits`.
            self.bus.lookup_key(
                builder,
                [self.x.clone() + AB::Expr::from_canonical_u32(self.shift)],
                count.clone(),
            );
        }
        self.bus.lookup_key(builder, [self.x], count);
    }
}
