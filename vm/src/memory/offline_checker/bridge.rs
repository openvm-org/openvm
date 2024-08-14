use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::memory::OpType;

/// Represents a memory bus identified by a unique bus index (`usize`).
pub struct MemoryBus(pub usize);

impl MemoryBus {
    /// Send a write operation through the memory bus.
    pub fn send_write<const WORD_SIZE: usize, AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        timestamp: impl Into<AB::Expr>,
        address_space: impl Into<AB::Expr>,
        address: impl Into<AB::Expr>,
        data: [AB::Expr; WORD_SIZE],
        count: impl Into<AB::Expr>,
    ) {
        self.send(
            builder,
            timestamp,
            OpType::Write,
            address_space,
            address,
            data.into(),
            count,
        );
    }

    /// Send a read operation through the memory bus.
    pub fn send_read<const WORD_SIZE: usize, AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        timestamp: impl Into<AB::Expr>,
        address_space: impl Into<AB::Expr>,
        address: impl Into<AB::Expr>,
        data: [AB::Expr; WORD_SIZE],
        count: impl Into<AB::Expr>,
    ) {
        self.send(
            builder,
            timestamp,
            OpType::Read,
            address_space,
            address,
            data,
            count,
        );
    }

    /// Sends a memory operation (read or write) through the memory bus.
    pub fn send<const WORD_SIZE: usize, AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        timestamp: impl Into<AB::Expr>,
        op_type: OpType,
        address_space: impl Into<AB::Expr>,
        address: impl Into<AB::Expr>,
        data: [AB::Expr; WORD_SIZE],
        count: impl Into<AB::Expr>,
    ) {
        let fields = [
            timestamp.into(),
            match op_type {
                OpType::Read => AB::Expr::zero(),
                OpType::Write => AB::Expr::one(),
            },
            address_space.into(),
            address.into(),
        ]
        .into_iter()
        .chain(data);

        builder.push_send(self.0, fields, count);
    }
}
