use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::memory::OpType;

/// Represents a memory bus identified by a unique bus index (`usize`).
pub struct MemoryBus(pub usize);

impl MemoryBus {
    /// Send a write operation through the memory bus.
    pub fn write<'a, const BLOCK_SIZE: usize, T>(
        &self,
        timestamp: impl Into<T>,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryInteraction<BLOCK_SIZE, T> {
        self.access(
            timestamp,
            OpType::Write,
            address_space,
            address,
            data,
        )
    }

    /// Send a read operation through the memory bus.
    pub fn read<'a, const BLOCK_SIZE: usize, T>(
        &self,
        timestamp: impl Into<T>,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryInteraction<BLOCK_SIZE, T> {
        self.access(
            timestamp,
            OpType::Read,
            address_space,
            address,
            data,
        )
    }

    /// Sends a memory operation (read or write) through the memory bus.
    pub fn access<const BLOCK_SIZE: usize, T>(
        &self,
        timestamp: impl Into<T>,
        op_type: OpType,
        address_space: impl Into<T>,
        address: impl Into<T>,
        data: [T; BLOCK_SIZE],
    ) -> MemoryInteraction<BLOCK_SIZE, T> {
        MemoryInteraction {
            bus_index: self.0,
            timestamp: timestamp.into(),
            op_type,
            address_space: address_space.into(),
            address: address.into(),
            data,
        }
    }
}


pub struct MemoryInteraction<const BLOCK_SIZE: usize, F> {
    bus_index: usize,
    timestamp: F,
    op_type: OpType,
    address_space: F,
    address: F,
    data: [F; BLOCK_SIZE],
}

impl<const BLOCK_SIZE: usize, F: AbstractField> MemoryInteraction<BLOCK_SIZE, F> {
    pub fn send<AB>(self, count: impl Into<AB::Expr>, builder: &mut AB)
    where
        AB: InteractionBuilder<Expr=F>,
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