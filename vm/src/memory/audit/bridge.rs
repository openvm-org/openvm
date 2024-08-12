use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::MemoryAuditAir, columns::AuditCols};
use crate::memory::manager::eval_memory_interactions;

impl<const WORD_SIZE: usize> MemoryAuditAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: AuditCols<WORD_SIZE, AB::Var>,
    ) {
        eval_memory_interactions(builder, local.op_cols, AB::Expr::one() - local.is_extra);
    }
}
