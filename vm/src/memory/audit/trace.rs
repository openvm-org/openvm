use std::{collections::BTreeMap, sync::Arc};

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::air::AuditAir;
use crate::memory::{audit::columns::AuditCols, manager::MemoryReadWriteOpCols};

impl<const WORD_SIZE: usize> AuditAir<WORD_SIZE> {
    pub fn generate_trace<F: PrimeField32>(
        &self,
        // (address_space, address) -> (clk, data)
        // TODO[osama]: update this to use AccessCell struct
        first_access: BTreeMap<(F, F), (F, [F; WORD_SIZE])>,
        mut last_access: BTreeMap<(F, F), (F, [F; WORD_SIZE])>,
        trace_height: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<F> {
        debug_assert_eq!(first_access.len(), last_access.len());

        let gen_row = |prev_idx: Vec<u32>,
                       cur_idx: Vec<u32>,
                       data_read: [F; WORD_SIZE],
                       clk_read: F,
                       data_write: [F; WORD_SIZE],
                       clk_write: F,
                       is_extra: F| {
            let lt_cols = LocalTraceInstructions::generate_trace_row(
                &self.addr_lt_air,
                (prev_idx, cur_idx.clone(), range_checker.clone()),
            );

            AuditCols::<WORD_SIZE, F>::new(
                MemoryReadWriteOpCols::<WORD_SIZE, F>::new(
                    F::from_canonical_u32(cur_idx[0]),
                    F::from_canonical_u32(cur_idx[1]),
                    data_read,
                    clk_read,
                    data_write,
                    clk_write,
                ),
                is_extra,
                lt_cols.io.tuple_less_than,
                lt_cols.aux,
            )
        };

        let mut rows_concat = Vec::with_capacity(trace_height * self.air_width());
        let mut prev_idx = vec![0, 0];
        for (addr, (clk_write, data_write)) in first_access {
            let (clk_read, data_read) = last_access.remove(&addr).unwrap();

            // TODO[osama]: add the ability to generate trace with Field elements as inputs
            let cur_idx = vec![addr.0.as_canonical_u32(), addr.1.as_canonical_u32()];

            rows_concat.extend(
                gen_row(
                    prev_idx,
                    cur_idx.clone(),
                    data_read,
                    clk_read,
                    data_write,
                    clk_write,
                    F::zero(),
                )
                .flatten(),
            );

            prev_idx = cur_idx;
        }

        let dummy_idx = vec![0, 0];
        let dummy_data = [F::zero(); WORD_SIZE];
        let dummy_clk = F::zero();

        while rows_concat.len() < trace_height * self.air_width() {
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

        RowMajorMatrix::new(rows_concat, self.air_width())
    }
}
