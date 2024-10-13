use std::collections::HashMap;

use afs_primitives::sub_chip::LocalTraceInstructions;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::MemoryAuditChip;
use crate::memory::{audit::columns::AuditCols, manager::TimestampedValue};

impl<F: PrimeField32> MemoryAuditChip<F> {
    pub fn generate_trace(
        &self,
        // TODO[osama]: consider making a struct for address
        final_memory: &HashMap<(F, F), TimestampedValue<F>>,
    ) -> RowMajorMatrix<F> {
        let trace_height = self.initial_memory.len().next_power_of_two();
        self.generate_trace_with_height(final_memory, trace_height)
    }
    pub fn generate_trace_with_height(
        &self,
        // TODO[osama]: consider making a struct for address
        final_memory: &HashMap<(F, F), TimestampedValue<F>>,
        trace_height: usize,
    ) -> RowMajorMatrix<F> {
        let gen_row = |prev_idx: Vec<u32>,
                       cur_idx: Vec<u32>,
                       final_data: F,
                       final_timestamp: F,
                       initial_data: F,
                       is_extra: F| {
            let lt_cols = LocalTraceInstructions::generate_trace_row(
                &self.air.addr_lt_air,
                (prev_idx, cur_idx.clone(), self.range_checker.clone()),
            );

            AuditCols::<F>::new(
                F::from_canonical_u32(cur_idx[0]),
                F::from_canonical_u32(cur_idx[1]),
                initial_data,
                final_data,
                final_timestamp,
                is_extra,
                lt_cols.io.tuple_less_than,
                lt_cols.aux,
            )
        };

        let mut rows_concat = Vec::with_capacity(trace_height * self.air.air_width());
        let mut prev_idx = vec![0, 0];
        for (addr, initial_data) in self.initial_memory.iter() {
            let TimestampedValue {
                timestamp: final_clk,
                value: final_data,
            } = final_memory.get(addr).unwrap();

            let cur_idx = vec![addr.0.as_canonical_u32(), addr.1.as_canonical_u32()];

            rows_concat.extend(
                gen_row(
                    prev_idx,
                    cur_idx.clone(),
                    *final_data,
                    *final_clk,
                    *initial_data,
                    F::zero(),
                )
                .flatten(),
            );

            prev_idx = cur_idx;
        }

        let dummy_idx = vec![0, 0];
        let dummy_data = F::zero();
        let dummy_clk = F::zero();

        while rows_concat.len() < trace_height * self.air.air_width() {
            rows_concat.extend(
                gen_row(
                    prev_idx.clone(),
                    dummy_idx.clone(),
                    dummy_data,
                    dummy_clk,
                    dummy_data,
                    F::one(),
                )
                .flatten(),
            );

            prev_idx.clone_from(&dummy_idx);
        }

        RowMajorMatrix::new(rows_concat, self.air.air_width())
    }
}
