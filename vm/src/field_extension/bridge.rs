use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    air::FieldExtensionArithmeticAir,
    chip::EXTENSION_DEGREE,
    columns::{FieldExtensionArithmeticCols, FieldExtensionArithmeticIoCols},
};
use crate::{
    arch::columns::{ExecutionState, InstructionCols},
    field_extension::columns::FieldExtensionArithmeticAuxCols,
    memory::{
        offline_checker::{
            bridge::{MemoryBridge, MemoryOfflineChecker},
            columns::MemoryOfflineCheckerAuxCols,
        },
        MemoryAddress,
    },
};

#[allow(clippy::too_many_arguments)]
fn eval_rw_interactions<AB: InteractionBuilder>(
    builder: &mut AB,
    mem_oc: MemoryOfflineChecker,
    clk_offset: &mut AB::Expr,
    is_enabled: AB::Expr,
    is_write: bool,
    clk: AB::Var,
    addr_space: AB::Var,
    address: AB::Var,
    ext: [AB::Var; EXTENSION_DEGREE],
    mem_oc_aux_cols: [MemoryOfflineCheckerAuxCols<1, AB::Var>; EXTENSION_DEGREE],
) {
    let mut memory_bridge = MemoryBridge::new(mem_oc, mem_oc_aux_cols);

    for (i, element) in ext.into_iter().enumerate() {
        let pointer = address + AB::F::from_canonical_usize(i);

        let clk = clk + clk_offset.clone();
        *clk_offset += is_enabled.clone();

        if is_write {
            memory_bridge
                .write(
                    MemoryAddress::new(addr_space, pointer),
                    [element.into()],
                    clk,
                )
                .eval(builder, is_enabled.clone());
        } else {
            memory_bridge
                .read(
                    MemoryAddress::new(addr_space, pointer),
                    [element.into()],
                    clk,
                )
                .eval(builder, is_enabled.clone());
        }
    }
}

impl FieldExtensionArithmeticAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: FieldExtensionArithmeticCols<AB::Var>,
    ) {
        let mut clk_offset = AB::Expr::zero();

        let FieldExtensionArithmeticCols { io, aux } = local;

        let FieldExtensionArithmeticIoCols {
            pc,
            opcode,
            timestamp,
            op_a,
            op_b,
            op_c,
            d,
            e,
            x,
            y,
            z,
            ..
        } = io;

        let FieldExtensionArithmeticAuxCols {
            read_x_aux_cols,
            read_y_aux_cols,
            write_aux_cols,
            is_valid,
            ..
        } = aux;

        // Reads for x
        eval_rw_interactions(
            builder,
            self.mem_oc,
            &mut clk_offset,
            is_valid.into(),
            false,
            timestamp,
            d,
            op_b,
            x,
            read_x_aux_cols,
        );

        // Reads for y
        eval_rw_interactions(
            builder,
            self.mem_oc,
            &mut clk_offset,
            is_valid.into(),
            false,
            timestamp,
            e,
            op_c,
            y,
            read_y_aux_cols,
        );

        // Writes for z
        eval_rw_interactions(
            builder,
            self.mem_oc,
            &mut clk_offset,
            is_valid.into(),
            true,
            timestamp,
            d,
            op_a,
            z,
            write_aux_cols,
        );

        let timestamp_delta = AB::Expr::from_canonical_usize(3 * EXTENSION_DEGREE);

        self.execution_bus.execute_increment_pc(
            builder,
            aux.is_valid,
            ExecutionState::new(pc, timestamp),
            timestamp_delta,
            InstructionCols::new(opcode, [op_a, op_b, op_c, d, e]),
        );
    }
}
