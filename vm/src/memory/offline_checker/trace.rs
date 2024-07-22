use std::sync::Arc;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::{MemoryAccess, OpType};

use super::{columns::MemoryOfflineCheckerCols, MemoryChip};
use afs_chips::{
    is_equal_vec::IsEqualVecAir,
    offline_checker::{OfflineCheckerChip, OfflineCheckerOperation},
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};
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

        let mut rows: Vec<F> = vec![];

        if !self.accesses.is_empty() {
            let local_input = (
                true,
                1,
                self.accesses[0].clone(),
                dummy_op.clone(),
                range_checker.clone(),
            );

            rows.extend(
                LocalTraceInstructions::<F>::generate_trace_row(self, local_input).flatten(),
            );
        }

        for i in 1..self.accesses.len() {
            let local_input = (
                false,
                1,
                self.accesses[i].clone(),
                self.accesses[i - 1].clone(),
                range_checker.clone(),
            );

            rows.extend(
                LocalTraceInstructions::<F>::generate_trace_row(self, local_input).flatten(),
            );
        }

        // Ensure that trace degree is a power of two
        let trace_degree = self.accesses.len().next_power_of_two();

        if self.accesses.len() < trace_degree {
            let local_input = (
                false,
                0,
                dummy_op.clone(),
                self.accesses[self.accesses.len() - 1].clone(),
                range_checker.clone(),
            );

            rows.extend(
                LocalTraceInstructions::<F>::generate_trace_row(self, local_input).flatten(),
            );
        }

        for _i in 1..(trace_degree - self.accesses.len()) {
            let local_input = (
                false,
                0,
                dummy_op.clone(),
                dummy_op.clone(),
                range_checker.clone(),
            );

            rows.extend(
                LocalTraceInstructions::<F>::generate_trace_row(self, local_input).flatten(),
            );
        }

        RowMajorMatrix::new(rows, self.air.air_width())
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> LocalTraceInstructions<F>
    for MemoryChip<WORD_SIZE, F>
{
    type LocalInput = (
        bool,
        u8,
        MemoryAccess<WORD_SIZE, F>,
        MemoryAccess<WORD_SIZE, F>,
        Arc<RangeCheckerGateChip>,
    );

    fn generate_trace_row(&self, input: Self::LocalInput) -> MemoryOfflineCheckerCols<F> {
        let (is_first_row, _, curr_op, prev_op, _) = input.clone();

        let offline_checker_chip = OfflineCheckerChip::<F, MemoryAccess<WORD_SIZE, F>>::new(
            self.air.offline_checker.clone(),
        );

        let offline_checker_cols =
            LocalTraceInstructions::<F>::generate_trace_row(&offline_checker_chip, input);

        let curr_data = curr_op.get_data();
        let prev_data = prev_op.get_data();
        let mut same_data = if curr_data == prev_data {
            F::one()
        } else {
            F::zero()
        };

        let mut same_idx_and_data = offline_checker_cols.same_idx * same_data;

        let is_equal_data_air = IsEqualVecAir::new(self.air.offline_checker.data_len);

        let is_equal_data_aux = is_equal_data_air
            .generate_trace_row((prev_data.clone(), curr_data.clone()))
            .aux;

        if is_first_row {
            same_data = F::zero();
            same_idx_and_data = F::zero();
        }

        MemoryOfflineCheckerCols {
            offline_checker_cols,
            same_data,
            same_idx_and_data,
            is_equal_data_aux,
        }
    }
}
