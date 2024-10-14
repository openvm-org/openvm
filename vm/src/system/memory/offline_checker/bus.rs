use std::iter;

use afs_stark_backend::interaction::{InteractionBuilder, InteractionType};
use p3_field::AbstractField;

use crate::system::memory::MemoryAddress;

/// Represents a memory bus identified by a unique bus index (`usize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBus(pub usize);

impl MemoryBus {
    /// Prepares a write operation through the memory bus.
    pub fn send<T: Clone>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: Vec<impl Into<T>>,
        timestamp: impl Into<T>,
    ) -> MemoryBusInteraction<T> {
        self.send_or_receive(InteractionType::Send, address, data, timestamp)
    }

    /// Prepares a read operation through the memory bus.
    pub fn receive<T: Clone>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: Vec<impl Into<T>>,
        timestamp: impl Into<T>,
    ) -> MemoryBusInteraction<T> {
        self.send_or_receive(InteractionType::Receive, address, data, timestamp)
    }

    /// Prepares a memory operation (read or write) through the memory bus.
    fn send_or_receive<T: Clone>(
        &self,
        interaction_type: InteractionType,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: Vec<impl Into<T>>,
        timestamp: impl Into<T>,
    ) -> MemoryBusInteraction<T> {
        MemoryBusInteraction {
            bus_index: self.0,
            interaction_type,
            address: MemoryAddress::new(address.address_space.into(), address.pointer.into()),
            data: data.into_iter().map(|item| item.into()).collect(),
            timestamp: timestamp.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryBusInteraction<T> {
    pub bus_index: usize,
    pub interaction_type: InteractionType,
    pub address: MemoryAddress<T, T>,
    pub data: Vec<T>,
    pub timestamp: T,
}

impl<T: AbstractField> MemoryBusInteraction<T> {
    /// Finalizes and sends/receives the memory operation with the specified count over the bus.
    ///
    /// A read corresponds to a receive, and a write corresponds to a send.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        // Chain 1 at the end to ensure that different length interactions are always distinct.
        let fields = iter::empty()
            .chain([self.address.address_space, self.address.pointer])
            .chain(self.data)
            .chain([self.timestamp])
            .chain(iter::once(AB::Expr::one()));

        builder.push_interaction(self.bus_index, fields, count, self.interaction_type);
    }
}
