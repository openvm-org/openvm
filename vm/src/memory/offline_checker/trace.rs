use std::sync::Arc;

use afs_primitives::offline_checker::OfflineCheckerOperation;
use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_primitives::{offline_checker::OfflineCheckerChip, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "parallel")]
use p3_maybe_rayon::prelude::*;

use crate::memory::{MemoryAccess, OpType};

use super::MemoryChip;

impl<const WORD_SIZE: usize, F: PrimeField32> MemoryChip<WORD_SIZE, F> {
    /// Each row in the trace follow the same order as the Cols struct:
    /// [clk, mem_row, op_type, same_addr_space, same_pointer, same_addr, same_data, lt_bit, is_valid, is_equal_addr_space_aux, is_equal_pointer_aux, is_equal_data_aux, lt_aux]
    ///
    /// The trace consists of a row for every read/write operation plus some extra rows
    /// The trace is sorted by addr (addr_space and pointer) and then by clk, so every addr has a block of consective rows in the trace with the following structure
    /// A row is added to the trace for every read/write operation with the corresponding data
    /// The trace is padded at the end to be of height trace_degree
    pub fn generate_trace(
        &mut self,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<F> {
        #[cfg(feature = "parallel")]
        self.accesses
            .par_sort_by_key(|op| (op.address_space, op.address, op.timestamp));
        #[cfg(not(feature = "parallel"))]
        self.accesses
            .sort_by_key(|op| (op.address_space, op.address, op.timestamp));

        let dummy_op = MemoryAccess {
            timestamp: 0,
            op_type: OpType::Read,
            address_space: F::zero(),
            address: F::zero(),
            data: [F::zero(); WORD_SIZE],
        };

        let offline_checker_chip = OfflineCheckerChip::new(self.air.offline_checker.clone());

        // offline_checker_chip.generate_trace(
        //     range_checker,
        //     self.accesses.clone(),
        //     dummy_op,
        //     self.accesses.len().next_power_of_two(),
        // )

        let mut rows: Vec<Vec<F>> = vec![];
        let mut rows_len = 0;
        let trace_degree = self.accesses.len().next_power_of_two();

        let mut prev_idx: Vec<F> = vec![];
        if !self.accesses.is_empty() {
            rows.push(
                LocalTraceInstructions::generate_trace_row(
                    &offline_checker_chip,
                    (
                        true,
                        true,
                        true,
                        self.accesses[0].clone(),
                        dummy_op.clone(),
                        range_checker.clone(),
                    ),
                )
                .flatten(),
            );
            prev_idx = self.accesses[0].get_idx();
            rows_len += 1;
        }

        for i in 1..self.accesses.len() {
            if self.accesses[i].get_idx() != prev_idx {
                rows[rows_len - 1].push(F::one());
            } else {
                rows[rows_len - 1].push(F::zero());
            }
            rows.push(
                LocalTraceInstructions::generate_trace_row(
                    &offline_checker_chip,
                    (
                        false,
                        true,
                        true,
                        self.accesses[i].clone(),
                        self.accesses[i - 1].clone(),
                        range_checker.clone(),
                    ),
                )
                .flatten(),
            );
            prev_idx = self.accesses[i].get_idx();
            rows_len += 1;
        }

        rows[rows_len - 1].push(F::one());

        if self.accesses.len() < trace_degree {
            rows.push(
                LocalTraceInstructions::generate_trace_row(
                    &offline_checker_chip,
                    (
                        false,
                        false,
                        false,
                        dummy_op.clone(),
                        self.accesses[self.accesses.len() - 1].clone(),
                        range_checker.clone(),
                    ),
                )
                .flatten(),
            );
            rows_len += 1;
            rows[rows_len - 1].push(F::zero());
        }

        for _i in 1..(trace_degree - self.accesses.len()) {
            rows.push(
                LocalTraceInstructions::generate_trace_row(
                    &offline_checker_chip,
                    (
                        false,
                        false,
                        false,
                        dummy_op.clone(),
                        self.accesses[self.accesses.len() - 1].clone(),
                        range_checker.clone(),
                    ),
                )
                .flatten(),
            );
            rows_len += 1;
            rows[rows_len - 1].push(F::zero());
        }

        RowMajorMatrix::new(rows.concat(), self.air.air_width())
    }
}
