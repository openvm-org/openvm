use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::columns::{FieldExtensionArithmeticCols, FieldExtensionArithmeticIoCols};
use crate::{
    cpu::FIELD_EXTENSION_BUS,
    field_extension::{
        air::FieldExtensionArithmeticAir, chip::EXTENSION_DEGREE,
        columns::FieldExtensionArithmeticAuxCols,
    },
    memory::{
        offline_checker::{
            bridge::{emb, MemoryBridge, MemoryOfflineChecker},
            columns::MemoryOfflineCheckerAuxCols,
        },
        MemoryAddress,
    },
};

#[allow(clippy::too_many_arguments)]
fn eval_rw_interactions<AB: InteractionBuilder, const WORD_SIZE: usize>(
    builder: &mut AB,
    mem_oc: MemoryOfflineChecker,
    clk_offset: &mut AB::Expr,
    is_enabled: AB::Expr,
    is_write: bool,
    clk: AB::Var,
    addr_space: AB::Var,
    address: AB::Var,
    ext: [AB::Var; EXTENSION_DEGREE],
    mem_oc_aux_cols: [MemoryOfflineCheckerAuxCols<WORD_SIZE, AB::Var>; EXTENSION_DEGREE],
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

impl<const WORD_SIZE: usize> FieldExtensionArithmeticAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: FieldExtensionArithmeticCols<WORD_SIZE, AB::Var>,
    ) {
        let mut clk_offset = AB::Expr::zero();

        let FieldExtensionArithmeticCols { io, aux } = local;

        let FieldExtensionArithmeticIoCols {
            clk,
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
            valid_y_read,
            ..
        } = aux;

        // Reads for x
        eval_rw_interactions::<AB, WORD_SIZE>(
            builder,
            self.mem_oc,
            &mut clk_offset,
            is_valid.into(),
            false,
            clk,
            d,
            op_b,
            x,
            read_x_aux_cols,
        );

        // Reads for y
        eval_rw_interactions::<AB, WORD_SIZE>(
            builder,
            self.mem_oc,
            &mut clk_offset,
            valid_y_read.into(),
            false,
            clk,
            e,
            op_c,
            y,
            read_y_aux_cols,
        );

        // Writes for z
        eval_rw_interactions::<AB, WORD_SIZE>(
            builder,
            self.mem_oc,
            &mut clk_offset,
            is_valid.into(),
            true,
            clk,
            d,
            op_a,
            z,
            write_aux_cols,
        );

        // Receives all IO columns from another chip on bus 3 (FIELD_EXTENSION_BUS)
        builder.push_receive(
            FIELD_EXTENSION_BUS,
            [io.opcode, io.clk, op_a, op_b, op_c, d, e],
            is_valid,
        );
    }
}
