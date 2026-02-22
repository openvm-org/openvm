use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_field::PrimeCharacteristicRing,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeferralCircuitCountBus(LookupBus);

impl DeferralCircuitCountBus {
    pub const fn new(index: BusIndex) -> Self {
        Self(LookupBus::new(index))
    }

    #[inline(always)]
    pub fn index(&self) -> BusIndex {
        self.0.index
    }

    #[must_use]
    pub fn send<T>(&self, deferral_idx: impl Into<T>) -> DeferralCircuitCountInteraction<T> {
        self.push(deferral_idx, true)
    }

    #[must_use]
    pub fn receive<T>(&self, deferral_idx: impl Into<T>) -> DeferralCircuitCountInteraction<T> {
        self.push(deferral_idx, false)
    }

    pub fn push<T>(
        &self,
        deferral_idx: impl Into<T>,
        is_lookup: bool,
    ) -> DeferralCircuitCountInteraction<T> {
        DeferralCircuitCountInteraction {
            deferral_idx: deferral_idx.into(),
            bus: self.0,
            is_lookup,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeferralCircuitCountInteraction<T> {
    pub deferral_idx: T,
    pub bus: LookupBus,
    pub is_lookup: bool,
}

impl<T: PrimeCharacteristicRing> DeferralCircuitCountInteraction<T> {
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let key = [self.deferral_idx];
        if self.is_lookup {
            self.bus.lookup_key(builder, key, count);
        } else {
            self.bus.add_key_with_lookups(builder, key, count);
        }
    }
}
