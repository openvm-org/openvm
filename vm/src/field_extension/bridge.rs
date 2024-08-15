use std::array;
use itertools::Itertools;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::FieldExtensionArithmeticCols, EXTENSION_DEGREE, FieldExtensionArithmeticAir};
use crate::cpu::FIELD_EXTENSION_BUS;
use crate::field_extension::columns::FieldExtensionArithmeticAuxCols;
use crate::memory::manager::access_cell::AccessCell;
use crate::memory::manager::operation::MemoryOperation;
use crate::memory::offline_checker::air::NewMemoryOfflineChecker;
use crate::memory::offline_checker::columns::MemoryOfflineCheckerAuxCols;

fn eval_rw_interactions<AB: InteractionBuilder, const WORD_SIZE: usize>(
    builder: &mut AB,
    mem_oc: &NewMemoryOfflineChecker<WORD_SIZE>,
    clk_offset: &mut usize,
    is_write: bool,
    start_timestamp: AB::Var,
    addr_space: AB::Var,
    address: AB::Var,
    ext_element: [AB::Var; EXTENSION_DEGREE],
    mem_aux_cols: [MemoryOfflineCheckerAuxCols<WORD_SIZE, AB::Var>; EXTENSION_DEGREE],
) {
    for (i, (element, aux_cols)) in ext_element.into_iter().zip_eq(mem_aux_cols.into_iter()).enumerate() {
        let pointer = address + AB::F::from_canonical_usize(i * WORD_SIZE);
        let data = array::from_fn(|i| if i == 0 { element.into() } else { AB::Expr::zero() });

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

        let FieldExtensionArithmeticCols {
            io,
            aux,
        } = local;

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

        // reads for x
        let mem_aux_cols = array::from_fn(|_| mem_aux_cols_iter.next().unwrap());
        eval_rw_interactions(builder, &self.mem_oc, &mut clk_offset, false, start_timestamp, d, op_b, io.x, mem_aux_cols);
        // reads for y
        let mem_aux_cols = array::from_fn(|_| mem_aux_cols_iter.next().unwrap());
        eval_rw_interactions(builder, &self.mem_oc, &mut clk_offset, false, start_timestamp, e, op_c, io.y, mem_aux_cols);
        // writes for z
        let mem_aux_cols = array::from_fn(|_| mem_aux_cols_iter.next().unwrap());
        eval_rw_interactions(builder, &self.mem_oc, &mut clk_offset, true, start_timestamp, d, op_a, io.z, mem_aux_cols);

        debug_assert!(matches!(mem_aux_cols_iter.next(), None));

        // Receives all IO columns from another chip on bus 3 (FIELD_EXTENSION_BUS)
        let fields = [
            io.opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
        ];
        builder.push_receive(FIELD_EXTENSION_BUS, fields, is_valid);
    }
}