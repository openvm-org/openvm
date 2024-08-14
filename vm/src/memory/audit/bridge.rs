use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::MemoryAuditAir, columns::AuditCols};
use crate::memory::offline_checker::air::NewMemoryOfflineChecker;

impl<const WORD_SIZE: usize> MemoryAuditAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: AuditCols<WORD_SIZE, AB::Var>,
    ) {
        NewMemoryOfflineChecker::eval_memory_interactions(
            builder,
            local.addr_space,
            local.pointer,
            local.final_cell,
            local.initial_cell,
            AB::Expr::one() - local.is_extra,
        );
    }
}
