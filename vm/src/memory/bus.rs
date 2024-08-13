use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::memory::{MemoryAccess, OpType};

pub struct MemoryBus(pub usize);

impl MemoryBus {
    pub fn send_interaction<const WORD_SIZE: usize, AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        mem_access: MemoryAccess<WORD_SIZE, AB::Expr>,
        count: impl Into<AB::Expr>,
    ) {
        let fields = [
            mem_access.timestamp,
            match mem_access.op_type {
                OpType::Read => AB::Expr::zero(),
                OpType::Write => AB::Expr::one(),
            },
            mem_access.address_space,
            mem_access.address,
        ]
        .into_iter()
        .chain(mem_access.data);

        builder.push_send(self.0, fields, count);
    }
}
