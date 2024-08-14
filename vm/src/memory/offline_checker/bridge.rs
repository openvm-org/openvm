use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::memory::OpType;

/// Represents a memory bus identified by a unique bus index (`usize`).
pub struct MemoryBus(pub usize);

impl MemoryBus {
    /// Prepares a write operation through the memory bus.
    pub fn write<T, const BLOCK_SIZE: usize>(
        &self,
        timestamp: impl Into<T>,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryBusInteraction<T, BLOCK_SIZE> {
        self.access(OpType::Write, timestamp, address_space, address, data)
    }

    /// Prepares a read operation through the memory bus.
    pub fn read<T, const BLOCK_SIZE: usize>(
        &self,
        timestamp: impl Into<T>,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryBusInteraction<T, BLOCK_SIZE> {
        self.access(OpType::Read, timestamp, address_space, address, data)
    }

    /// Prepares a memory operation (read or write) through the memory bus.
    pub fn access<T, const BLOCK_SIZE: usize>(
        &self,
        op_type: OpType,
        timestamp: impl Into<T>,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryBusInteraction<T, BLOCK_SIZE> {
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

pub struct MemoryBusInteraction<T, const BLOCK_SIZE: usize> {
    bus_index: usize,
    timestamp: T,
    op_type: OpType,
    address_space: T,
    address: T,
    data: [T; BLOCK_SIZE],
}

impl<const BLOCK_SIZE: usize, T: AbstractField> MemoryBusInteraction<T, BLOCK_SIZE> {
    /// Finalizes and sends the memory operation with the specified count over the bus.
    pub fn send<AB>(self, count: impl Into<AB::Expr>, builder: &mut AB)
    where
        AB: InteractionBuilder<Expr=T>,
    {
        let fields = [
            self.timestamp,
            match self.op_type {
                OpType::Read => AB::Expr::zero(),
                OpType::Write => AB::Expr::one(),
            },
            self.address_space,
            self.address,
        ]
        .into_iter()
        .chain(self.data);

        builder.push_send(self.bus_index, fields, count);
    }
}
