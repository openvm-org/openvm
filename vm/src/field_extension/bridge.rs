use std::array;

use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use p3_field::AbstractField;

use super::{columns::FieldExtensionArithmeticCols, FieldExtensionArithmeticAir, EXTENSION_DEGREE};
use crate::{
    cpu::FIELD_EXTENSION_BUS,
    field_extension::columns::FieldExtensionArithmeticAuxCols,
    memory::{
        manager::{access_cell::AccessCell, operation::MemoryOperation},
        offline_checker::{bridge::NewMemoryOfflineChecker, columns::MemoryOfflineCheckerAuxCols},
    },
};

#[allow(clippy::too_many_arguments)]
fn eval_rw_interactions<AB: InteractionBuilder, const WORD_SIZE: usize>(
    builder: &mut AB,
    mem_oc: &NewMemoryOfflineChecker,
    clk_offset: &mut usize,
    is_write: bool,
    start_timestamp: AB::Var,
    addr_space: AB::Var,
    address: AB::Var,
    ext: [AB::Var; EXTENSION_DEGREE],
    mem_aux_cols: [MemoryOfflineCheckerAuxCols<WORD_SIZE, AB::Var>; EXTENSION_DEGREE],
) {
    for (i, (element, aux_cols)) in ext.into_iter().zip_eq(mem_aux_cols.into_iter()).enumerate() {
        let pointer = address + AB::F::from_canonical_usize(i * WORD_SIZE);
        let data = array::from_fn(|i| {
            if i == 0 {
                element.into()
            } else {
                AB::Expr::zero()
            }
        });

        let clk = start_timestamp + AB::Expr::from_canonical_usize(*clk_offset);
        *clk_offset += 1;

        let op = MemoryOperation::<WORD_SIZE, AB::Expr> {
            addr_space: addr_space.into(),
            pointer,
            op_type: AB::Expr::from_bool(is_write),
            cell: AccessCell { data, clk },
            enabled: AB::Expr::one(),
        };

        mem_oc.subair_eval(builder, op, aux_cols);
    }
}

impl<const WORD_SIZE: usize> FieldExtensionArithmeticAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: FieldExtensionArithmeticCols<WORD_SIZE, AB::Var>,
    ) {
        let mut clk_offset = 0;

        let FieldExtensionArithmeticCols { io, aux } = local;

        let FieldExtensionArithmeticAuxCols {
            op_a,
            op_b,
            op_c,
            d,
            e,
            start_timestamp,
            mem_oc_aux_cols,
            is_valid,
            ..
        } = aux;

        let mut mem_aux_cols_iter = mem_oc_aux_cols.into_iter();

        // Reads for x
        eval_rw_interactions(
            builder,
            &self.mem_oc,
            &mut clk_offset,
            false,
            start_timestamp,
            d,
            op_b,
            io.x,
            array::from_fn(|_| mem_aux_cols_iter.next().unwrap()),
        );

        // Reads for y
        eval_rw_interactions(
            builder,
            &self.mem_oc,
            &mut clk_offset,
            false,
            start_timestamp,
            e,
            op_c,
            io.y,
            array::from_fn(|_| mem_aux_cols_iter.next().unwrap()),
        );

        // Writes for z
        eval_rw_interactions(
            builder,
            &self.mem_oc,
            &mut clk_offset,
            true,
            start_timestamp,
            d,
            op_a,
            io.z,
            array::from_fn(|_| mem_aux_cols_iter.next().unwrap()),
        );

        debug_assert!(mem_aux_cols_iter.next().is_none());

        // Receives all IO columns from another chip on bus 3 (FIELD_EXTENSION_BUS)
        builder.push_receive(
            FIELD_EXTENSION_BUS,
            [io.opcode, op_a, op_b, op_c, d, e],
            is_valid,
        );
    }
}
