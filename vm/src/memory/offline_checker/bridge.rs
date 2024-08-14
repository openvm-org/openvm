use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::memory::OpType;

/// Represents a memory bus identified by a unique bus index (`usize`).
pub struct MemoryBus(pub usize);

impl MemoryBus {
    /// Prepares a write operation through the memory bus.
    pub fn write<'a, const BLOCK_SIZE: usize, T>(
        &self,
        timestamp: impl Into<T>,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryBusInteraction<BLOCK_SIZE, T> {
        self.access(OpType::Write, timestamp, address_space, address, data)
    }

    /// Prepares a read operation through the memory bus.
    pub fn read<'a, const BLOCK_SIZE: usize, T>(
        &self,
        timestamp: impl Into<T>,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryBusInteraction<BLOCK_SIZE, T> {
        self.access(OpType::Read, timestamp, address_space, address, data)
    }

    /// Prepares a memory operation (read or write) through the memory bus.
    pub fn access<const BLOCK_SIZE: usize, T>(
        &self,
        op_type: OpType,
        timestamp: impl Into<T>,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryBusInteraction<BLOCK_SIZE, T> {
        MemoryBusInteraction {
            bus_index: self.0,
            timestamp: timestamp.into(),
            op_type,
            address_space: address_space.into(),
            address: address.into(),
            data,
        }
    }
}

pub struct MemoryBusInteraction<const BLOCK_SIZE: usize, T> {
    bus_index: usize,
    timestamp: T,
    op_type: OpType,
    address_space: T,
    address: T,
    data: [T; BLOCK_SIZE],
}

impl<const BLOCK_SIZE: usize, F: AbstractField> MemoryBusInteraction<BLOCK_SIZE, F> {
    /// Finalizes and sends the memory operation with the specified count over the bus.
    pub fn send<AB>(self, count: impl Into<AB::Expr>, builder: &mut AB)
    where
        AB: InteractionBuilder<Expr = F>,
    {
        let fields = [
            self.timestamp.into(),
            match self.op_type {
                OpType::Read => AB::Expr::zero(),
                OpType::Write => AB::Expr::one(),
            },
            self.address_space.into(),
            self.address.into(),
        ]
        .into_iter()
        .chain(self.data);

        builder.push_send(self.bus_index, fields, count);
    }
}
