use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::AuditAir, columns::AuditCols};
use crate::cpu::NEW_MEMORY_BUS;

impl<const WORD_SIZE: usize> AuditAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: AuditCols<WORD_SIZE, AB::Var>,
    ) {
        let op_cols = local.op_cols;

        builder.push_send(
            NEW_MEMORY_BUS,
            vec![op_cols.address_space, op_cols.address]
                .into_iter()
                .chain(op_cols.data_write)
                .chain(iter::once(op_cols.clk_write))
                .map(Into::into)
                .collect::<Vec<AB::Expr>>(),
            AB::Expr::one() - local.is_extra.into(),
        );

        builder.push_receive(
            NEW_MEMORY_BUS,
            vec![op_cols.address_space, op_cols.address]
                .into_iter()
                .chain(op_cols.data_read)
                .chain(iter::once(op_cols.clk_read))
                .map(Into::into)
                .collect::<Vec<AB::Expr>>(),
            AB::Expr::one() - local.is_extra.into(),
        );
    }
}
