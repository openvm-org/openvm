use std::array;

use p3_field::AbstractField;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::{
    arch::columns::{ExecutionState, InstructionCols},
    field_extension::columns::FieldExtensionArithmeticAuxCols,
    memory::{MemoryAddress, offline_checker::bridge::MemoryBridge},
};

use super::{columns::FieldExtensionArithmeticCols, EXTENSION_DEGREE, FieldExtensionArithmeticAir};

#[allow(clippy::too_many_arguments)]
fn eval_rw_interactions<AB: InteractionBuilder, const WORD_SIZE: usize>(
    builder: &mut AB,
    memory_bridge: &mut MemoryBridge<AB::Var, WORD_SIZE>,
    clk_offset: &mut AB::Expr,
    is_enabled: AB::Expr,
    is_write: bool,
    clk: AB::Var,
    addr_space: AB::Var,
    address: AB::Var,
    ext: [AB::Var; EXTENSION_DEGREE],
) {
    for (i, element) in ext.into_iter().enumerate() {
        let pointer = address + AB::F::from_canonical_usize(i * WORD_SIZE);

        let clk = clk + clk_offset.clone();
        *clk_offset += is_enabled.clone();

        if is_write {
            memory_bridge
                .write(
                    MemoryAddress::new(addr_space, pointer),
                    emb(element.into()),
                    clk,
                )
                .eval(builder, is_enabled.clone());
        } else {
            memory_bridge
                .read(
                    MemoryAddress::new(addr_space, pointer),
                    emb(element.into()),
                    clk,
                )
                .eval(builder, is_enabled.clone());
        }
    }
}

fn emb<F: AbstractField, const WORD_SIZE: usize>(element: F) -> [F; WORD_SIZE] {
    array::from_fn(|j| if j == 0 { element.clone() } else { F::zero() })
}

impl FieldExtensionArithmeticAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: FieldExtensionArithmeticCols<AB::Var>,
    ) {
        let mut clk_offset = AB::Expr::zero();

        let FieldExtensionArithmeticCols { io, aux } = local;

        let FieldExtensionArithmeticAuxCols {
            op_a,
            op_b,
            op_c,
            d,
            e,
            mem_oc_aux_cols,
            is_valid,
            ..
        } = aux;

        let mut memory_bridge = MemoryBridge::new(self.mem_oc, mem_oc_aux_cols);

        // Reads for x
        eval_rw_interactions(
            builder,
            &mut memory_bridge,
            &mut clk_offset,
            is_valid.into(),
            false,
            io.timestamp,
            d,
            op_b,
            io.x,
        );

        // Reads for y
        eval_rw_interactions(
            builder,
            &mut memory_bridge,
            &mut clk_offset,
            aux.valid_y_read.into(),
            false,
            io.timestamp,
            e,
            op_c,
            io.y,
        );

        // Writes for z
        eval_rw_interactions(
            builder,
            &mut memory_bridge,
            &mut clk_offset,
            is_valid.into(),
            true,
            io.timestamp,
            d,
            op_a,
            io.z,
        );

        self.execution_bus.execute_increment_pc(
            builder,
            aux.is_valid,
            ExecutionState::new(io.pc, io.timestamp),
            AB::F::from_canonical_usize(Self::timestamp_delta()),
            InstructionCols::new(io.opcode, aux.op_a, aux.op_b, aux.op_c, aux.d, aux.e),
        );
    }
}
