use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::columns::FieldExtensionArithmeticCols;
use crate::{
    cpu::FIELD_EXTENSION_BUS,
    field_extension::{
        air::FieldExtensionArithmeticAir, chip::EXTENSION_DEGREE,
        columns::FieldExtensionArithmeticAuxCols,
    },
    memory::{decompose, offline_checker::bridge::MemoryBridge, MemoryAddress},
};

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
        let pointer = address + AB::F::from_canonical_usize(i);

        let clk = clk + clk_offset.clone();
        *clk_offset += is_enabled.clone();

        if is_write {
            memory_bridge
                .write(
                    MemoryAddress::new(addr_space, pointer),
                    decompose(element.into()),
                    clk,
                )
                .eval(builder, is_enabled.clone());
        } else {
            memory_bridge
                .read(
                    MemoryAddress::new(addr_space, pointer),
                    decompose(element.into()),
                    clk,
                )
                .eval(builder, is_enabled.clone());
        }
    }
}

impl<const WORD_SIZE: usize> FieldExtensionArithmeticAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: FieldExtensionArithmeticCols<WORD_SIZE, AB::Var>,
    ) {
        let mut clk_offset = AB::Expr::zero();

        let FieldExtensionArithmeticCols { io, aux } = local;

        let FieldExtensionArithmeticAuxCols {
            op_a,
            op_b,
            op_c,
            d,
            e,
            read_x_aux_cols: read_1_aux_cols,
            read_y_aux_cols: read_2_aux_cols,
            write_aux_cols,
            is_valid,
            ..
        } = aux;

        let mut memory_bridge = MemoryBridge::new(self.mem_oc, read_1_aux_cols);

        // Reads for x
        eval_rw_interactions::<AB, WORD_SIZE>(
            builder,
            &mut memory_bridge,
            &mut clk_offset,
            aux.is_valid.into(),
            false,
            io.clk,
            d,
            op_b,
            io.x,
        );

        let mut memory_bridge = MemoryBridge::new(self.mem_oc, read_2_aux_cols);

        // Reads for y
        eval_rw_interactions::<AB, WORD_SIZE>(
            builder,
            &mut memory_bridge,
            &mut clk_offset,
            aux.valid_y_read.into(),
            false,
            io.clk,
            e,
            op_c,
            io.y,
        );

        let mut memory_bridge = MemoryBridge::new(self.mem_oc, write_aux_cols);

        // Writes for z
        eval_rw_interactions::<AB, WORD_SIZE>(
            builder,
            &mut memory_bridge,
            &mut clk_offset,
            aux.is_valid.into(),
            true,
            io.clk,
            d,
            op_a,
            io.z,
        );

        // Receives all IO columns from another chip on bus 3 (FIELD_EXTENSION_BUS)
        builder.push_receive(
            FIELD_EXTENSION_BUS,
            [io.opcode, io.clk, op_a, op_b, op_c, d, e],
            is_valid,
        );
    }
}
