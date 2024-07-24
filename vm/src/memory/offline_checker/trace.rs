use std::sync::Arc;

use afs_chips::offline_checker::OfflineCheckerChip;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::{MemoryAccess, OpType};

use super::MemoryChip;
use afs_chips::range_gate::RangeCheckerGateChip;
use p3_maybe_rayon::prelude::*;

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
        self.accesses
            .par_sort_by_key(|op| (op.address_space, op.address, op.timestamp));

        let dummy_op = MemoryAccess {
            timestamp: 0,
            op_type: OpType::Read,
            address_space: F::zero(),
            address: F::zero(),
            data: [F::zero(); WORD_SIZE],
        };

        let mut offline_checker_chip = OfflineCheckerChip::new(self.air.offline_checker.clone());

        offline_checker_chip.generate_trace(
            range_checker,
            self.accesses.clone(),
            dummy_op,
            self.accesses.len().next_power_of_two(),
        )
    }
}
