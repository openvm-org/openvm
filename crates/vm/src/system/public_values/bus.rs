use std::iter;

use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_field::PrimeCharacteristicRing,
};

use crate::arch::BLOCK_FE_WIDTH;

/// Bus for revealing the `ordinal`th segment-local public output as four `u16` limbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicValuesBus {
    pub inner: PermutationCheckBus,
}

impl PublicValuesBus {
    pub const fn new(index: BusIndex) -> Self {
        Self {
            inner: PermutationCheckBus::new(index),
        }
    }

    #[inline(always)]
    pub fn index(&self) -> BusIndex {
        self.inner.index
    }

    pub fn send<T>(
        &self,
        ordinal: impl Into<T>,
        value: [impl Into<T>; BLOCK_FE_WIDTH],
    ) -> PublicValuesBusInteraction<T> {
        self.interact(true, ordinal, value)
    }

    pub fn receive<T>(
        &self,
        ordinal: impl Into<T>,
        value: [impl Into<T>; BLOCK_FE_WIDTH],
    ) -> PublicValuesBusInteraction<T> {
        self.interact(false, ordinal, value)
    }

    fn interact<T>(
        &self,
        is_send: bool,
        ordinal: impl Into<T>,
        value: [impl Into<T>; BLOCK_FE_WIDTH],
    ) -> PublicValuesBusInteraction<T> {
        PublicValuesBusInteraction {
            bus: self.inner,
            is_send,
            ordinal: ordinal.into(),
            value: value.map(Into::into),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PublicValuesBusInteraction<T> {
    pub bus: PermutationCheckBus,
    pub is_send: bool,
    pub ordinal: T,
    pub value: [T; BLOCK_FE_WIDTH],
}

impl<T: PrimeCharacteristicRing> PublicValuesBusInteraction<T> {
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let fields = iter::once(self.ordinal).chain(self.value);
        let multiplicity = enabled.into();
        if self.is_send {
            self.bus.interact(builder, fields, multiplicity);
        } else {
            self.bus.interact(builder, fields, -multiplicity);
        }
    }
}
