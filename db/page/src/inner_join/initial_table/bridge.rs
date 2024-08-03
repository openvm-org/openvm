use afs_stark_backend::interaction::InteractionBuilder;

use super::{InitialTableAir, TableType};
use crate::common::page_cols::PageCols;

impl InitialTableAir {
    /// For T1:
    /// - Sends idx (primary key) with multiplicity is_alloc on t1_intersector_bus (received by intersector_chip)
    /// - Sends (idx, data) with multiplicity out_mult on t1_output_bus (received by output_chip)
    /// For T2:
    /// - Sends foreign key with multiplicity is_alloc on t2_intersector_bus (received by intersector_chip)
    /// - Sends (idx, data) with multiplicity out_mult on t2_output_bus (received by output_chip)
    ///
    /// For T2:
    /// - Receives foreign key with multiplicity out_mult on intersector_t2_bus (sent by intersector_chip)
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        page: PageCols<AB::Var>,
        out_mult: AB::Var,
    ) {
        match self.table_type {
            TableType::T1 {
                t1_intersector_bus_index,
                t1_output_bus_index,
            } => {
                builder.push_send(t1_intersector_bus_index, page.idx.clone(), page.is_alloc);
                builder.push_send(
                    t1_output_bus_index,
                    page.idx.into_iter().chain(page.data),
                    out_mult,
                );
            }
            TableType::T2 {
                t2_intersector_bus_index,
                t2_output_bus_index,
                intersector_t2_bus_index,
                fkey_start,
                fkey_end,
            } => {
                builder.push_receive(
                    intersector_t2_bus_index,
                    page.data[fkey_start..fkey_end].iter().copied(),
                    out_mult,
                );

                builder.push_send(
                    t2_intersector_bus_index,
                    page.data[fkey_start..fkey_end].iter().copied(),
                    page.is_alloc,
                );

                builder.push_send(
                    t2_output_bus_index,
                    page.idx.into_iter().chain(page.data),
                    out_mult,
                );
            }
        }
    }
}
