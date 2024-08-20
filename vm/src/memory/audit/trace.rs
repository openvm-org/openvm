use std::collections::BTreeMap;

use afs_primitives::sub_chip::LocalTraceInstructions;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::MemoryAuditChip;
use crate::memory::{audit::columns::AuditCols, manager::access_cell::AccessCell};

impl<const WORD_SIZE: usize, F: PrimeField32> MemoryAuditChip<WORD_SIZE, F> {
    pub fn generate_trace(
        &self,
        // TODO[osama]: consider making a struct for address
        final_memory: &BTreeMap<(F, F), AccessCell<WORD_SIZE, F>>,
    ) -> RowMajorMatrix<F> {
        let trace_height = self.initial_memory.len().next_power_of_two();

        let gen_row = |prev_idx: Vec<u32>,
                       cur_idx: Vec<u32>,
                       data_read: [F; WORD_SIZE],
                       clk_read: F,
                       data_write: [F; WORD_SIZE],
                       clk_write: F,
                       is_extra: F| {
            let lt_cols = LocalTraceInstructions::generate_trace_row(
                &self.air.addr_lt_air,
                (prev_idx, cur_idx.clone(), self.range_checker.clone()),
            );

            AuditCols::<WORD_SIZE, F>::new(
                F::from_canonical_u32(cur_idx[0]),
                F::from_canonical_u32(cur_idx[1]),
                AccessCell::<WORD_SIZE, F>::new(data_write, clk_write),
                AccessCell::<WORD_SIZE, F>::new(data_read, clk_read),
                is_extra,
                lt_cols.io.tuple_less_than,
                lt_cols.aux,
            )
        };

        let mut rows_concat = Vec::with_capacity(trace_height * self.air.air_width());
        let mut prev_idx = vec![0, 0];
        for (
            addr,
            AccessCell {
                clk: clk_write,
                data: data_write,
            },
        ) in self.initial_memory.iter()
        {
            let AccessCell {
                clk: clk_read,
                data: data_read,
            } = final_memory.get(addr).unwrap();

            let cur_idx = vec![addr.0.as_canonical_u32(), addr.1.as_canonical_u32()];

            rows_concat.extend(
                gen_row(
                    prev_idx,
                    cur_idx.clone(),
                    *data_read,
                    *clk_read,
                    *data_write,
                    *clk_write,
                    F::zero(),
                )
                .flatten(),
            );

            prev_idx = cur_idx;
        }

        let dummy_idx = vec![0, 0];
        let dummy_data = [F::zero(); WORD_SIZE];
        let dummy_clk = F::zero();

        while rows_concat.len() < trace_height * self.air.air_width() {
            rows_concat.extend(
                gen_row(
                    prev_idx.clone(),
                    dummy_idx.clone(),
                    dummy_data,
                    dummy_clk,
                    dummy_data,
                    dummy_clk,
                    F::one(),
                )
                .flatten(),
            );

            prev_idx.clone_from(&dummy_idx);
        }

        RowMajorMatrix::new(rows_concat, self.air.air_width())
    }
}
