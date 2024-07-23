use afs_stark_backend::interaction::InteractionBuilder;

use crate::cpu::MEMORY_BUS;

use super::OfflineChecker;

impl<const WORD_SIZE: usize> OfflineChecker<WORD_SIZE> {
    /// Receives operations (clk, op_type, addr_space, pointer, data)
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        clk: AB::Var,
        op_type: AB::Var,
        mem_row: impl IntoIterator<Item = AB::Expr>,
        is_valid: AB::Var,
    ) {
        let fields = [clk.into(), op_type.into()].into_iter().chain(mem_row);

        builder.push_receive(MEMORY_BUS, fields, is_valid);
    }
}
