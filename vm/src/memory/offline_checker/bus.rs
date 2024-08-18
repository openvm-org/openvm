use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::memory::{MemoryAddress, OpType};

/// Represents a memory bus identified by a unique bus index (`usize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBus(pub usize);

impl MemoryBus {
    /// Prepares a write operation through the memory bus.
    pub fn write<T, const BLOCK_SIZE: usize>(
        &self,
        timestamp: impl Into<T>,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; BLOCK_SIZE],
    ) -> MemoryBusInteraction<T, BLOCK_SIZE> {
        self.access(OpType::Write, timestamp, address, data)
    }

    /// Prepares a read operation through the memory bus.
    pub fn read<T, const BLOCK_SIZE: usize>(
        &self,
        timestamp: impl Into<T>,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; BLOCK_SIZE],
    ) -> MemoryBusInteraction<T, BLOCK_SIZE> {
        self.access(OpType::Read, timestamp, address, data)
    }

    /// Prepares a memory operation (read or write) through the memory bus.
    pub fn access<T, const BLOCK_SIZE: usize>(
        &self,
        op_type: OpType,
        timestamp: impl Into<T>,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; BLOCK_SIZE],
    ) -> MemoryBusInteraction<T, BLOCK_SIZE> {
        MemoryBusInteraction {
            bus_index: self.0,
            timestamp: timestamp.into(),
            op_type,
            address: MemoryAddress::new(address.address_space.into(), address.pointer.into()),
            data: data.map(Into::into),
        }
    }
}

pub struct MemoryBusInteraction<T, const BLOCK_SIZE: usize> {
    bus_index: usize,
    op_type: OpType,
    timestamp: T,
    address: MemoryAddress<T, T>,
    data: [T; BLOCK_SIZE],
}

impl<const BLOCK_SIZE: usize, T: AbstractField> MemoryBusInteraction<T, BLOCK_SIZE> {
    /// Finalizes and sends/receives the memory operation with the specified count over the bus.
    ///
    /// A read corresponds to a receive, and a write corresponds to a send.
    pub fn push<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let fields = iter::empty()
            .chain([self.address.address_space, self.address.pointer])
            .chain(self.data)
            .chain([self.timestamp]);

        match self.op_type {
            OpType::Read => {
                builder.push_receive(self.bus_index, fields, count);
            }
            OpType::Write => {
                builder.push_send(self.bus_index, fields, count);
            }
        }
    }
}
