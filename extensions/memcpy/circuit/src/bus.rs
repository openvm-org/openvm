use std::iter;

use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_field::FieldAlgebra,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemcpyBus {
    pub inner: PermutationCheckBus,
}

impl MemcpyBus {
    pub const fn new(index: BusIndex) -> Self {
        Self {
            inner: PermutationCheckBus::new(index),
        }
    }
}

impl MemcpyBus {
    #[inline(always)]
    pub fn index(&self) -> BusIndex {
        self.inner.index
    }

    pub fn send<T: Clone>(
        &self,
        timestamp: impl Into<T>,
        dest: impl Into<T>,
        source: impl Into<T>,
        len: impl Into<T>,
        shift: impl Into<T>,
    ) -> MemcpyBusInteraction<T> {
        self.push(true, timestamp, dest, source, len, shift)
    }

    pub fn receive<T: Clone>(
        &self,
        timestamp: impl Into<T>,
        dest: impl Into<T>,
        source: impl Into<T>,
        len: impl Into<T>,
        shift: impl Into<T>,
    ) -> MemcpyBusInteraction<T> {
        self.push(false, timestamp, dest, source, len, shift)
    }

    fn push<T: Clone>(
        &self,
        is_send: bool,
        timestamp: impl Into<T>,
        dest: impl Into<T>,
        source: impl Into<T>,
        len: impl Into<T>,
        shift: impl Into<T>,
    ) -> MemcpyBusInteraction<T> {
        MemcpyBusInteraction {
            bus: self.inner,
            is_send,
            timestamp: timestamp.into(),
            dest: dest.into(),
            source: source.into(),
            len: len.into(),
            shift: shift.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemcpyBusInteraction<T> {
    pub bus: PermutationCheckBus,
    pub is_send: bool,
    pub timestamp: T,
    pub dest: T,
    pub source: T,
    pub len: T,
    pub shift: T,
}

impl<T: FieldAlgebra> MemcpyBusInteraction<T> {
    pub fn eval<AB>(self, builder: &mut AB, direction: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let fields = iter::empty()
            .chain(iter::once(self.timestamp))
            .chain(iter::once(self.dest))
            .chain(iter::once(self.source))
            .chain(iter::once(self.len))
            .chain(iter::once(self.shift));

        if self.is_send {
            self.bus.interact(builder, fields, direction);
        } else {
            self.bus
                .interact(builder, fields, AB::Expr::NEG_ONE * direction.into());
        }
    }
}
