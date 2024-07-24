use std::sync::Arc;

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::range_gate::RangeCheckerGateChip;
use crate::sub_chip::LocalTraceInstructions;

use super::{columns::OfflineCheckerCols, OfflineCheckerChip, OfflineCheckerOperation};

impl<F: PrimeField64, Operation: OfflineCheckerOperation<F> + Clone>
    OfflineCheckerChip<F, Operation>
{
    /// Each row in the trace follows the same order as the Cols struct:
    /// [clk, idx, data, op_type, same_idx, lt_bit, is_valid, is_equal_idx_aux, lt_aux]
    ///
    /// The trace consists of a row for every read/write operation plus some extra rows
    /// The trace is sorted by addr (addr_space and pointer) and then by clk, so every addr has a block of consective rows in the trace with the following structure
    /// A row is added to the trace for every read/write operation with the corresponding data
    /// The trace is padded at the end to be of height trace_degree
    pub fn generate_trace(
        &mut self,
        range_checker: Arc<RangeCheckerGateChip>,
        // should be already sorted by address_space, address, timestamp
        accesses: Vec<Operation>,
        dummy_op: Operation,
        trace_degree: usize,
    ) -> RowMajorMatrix<F> {
        let mut rows: Vec<Vec<F>> = vec![];

        if !accesses.is_empty() {
            rows.push(
                self.generate_trace_row((
                    true,
                    true,
                    true,
                    accesses[0].clone(),
                    dummy_op.clone(),
                    range_checker.clone(),
                ))
                .flatten(),
            );
        }

        for i in 1..accesses.len() {
            rows.push(
                self.generate_trace_row((
                    false,
                    true,
                    true,
                    accesses[i].clone(),
                    accesses[i - 1].clone(),
                    range_checker.clone(),
                ))
                .flatten(),
            );
        }

        if accesses.len() < trace_degree {
            rows.push(
                self.generate_trace_row((
                    false,
                    false,
                    false,
                    dummy_op.clone(),
                    accesses[accesses.len() - 1].clone(),
                    range_checker.clone(),
                ))
                .flatten(),
            );
        }

        for _i in 1..(trace_degree - accesses.len()) {
            rows.push(
                self.generate_trace_row((
                    false,
                    false,
                    false,
                    dummy_op.clone(),
                    dummy_op.clone(),
                    range_checker.clone(),
                ))
                .flatten(),
            );
        }

        RowMajorMatrix::new(rows.concat(), self.air.air_width())
    }
}

impl<F: PrimeField64, Operation: OfflineCheckerOperation<F>> LocalTraceInstructions<F>
    for OfflineCheckerChip<F, Operation>
{
    // is_first_row, is_valid, is_receive, curr_op, prev_op, range_checker
    type LocalInput = (
        bool,
        bool,
        bool,
        Operation,
        Operation,
        Arc<RangeCheckerGateChip>,
    );

    fn generate_trace_row(&self, input: Self::LocalInput) -> OfflineCheckerCols<F> {
        let (is_first_row, is_valid, is_receive, curr_op, prev_op, range_checker) = input;
        let op_type = curr_op.get_op_type();

        let curr_timestamp = curr_op.get_timestamp();
        let prev_timestamp = prev_op.get_timestamp();

        let curr_idx = curr_op.get_idx();
        let prev_idx = prev_op.get_idx();
        let mut same_idx = if curr_idx == prev_idx { 1 } else { 0 };

        let curr_data = curr_op.get_data();

        let mut lt_bit = 1;
        for i in 0..curr_idx.len() {
            match curr_idx[i].cmp(&prev_idx[i]) {
                std::cmp::Ordering::Greater => break,
                std::cmp::Ordering::Less => {
                    lt_bit = 0;
                    break;
                }
                std::cmp::Ordering::Equal => {
                    if i == curr_idx.len() - 1 && curr_op.get_timestamp() <= prev_op.get_timestamp()
                    {
                        lt_bit = 0;
                    }
                }
            }
        }

        let is_equal_idx_aux = self
            .air
            .is_equal_idx_air
            .generate_trace_row((prev_idx.clone(), curr_idx.clone()))
            .aux;

        let mut prev_idx_timestamp = prev_idx
            .clone()
            .into_iter()
            .map(|x| x.as_canonical_u64() as u32)
            .collect::<Vec<_>>();
        prev_idx_timestamp.push(prev_timestamp as u32);

        let mut curr_idx_timestamp = curr_idx
            .clone()
            .into_iter()
            .map(|x| x.as_canonical_u64() as u32)
            .collect::<Vec<_>>();
        curr_idx_timestamp.push(curr_timestamp as u32);

        let lt_aux = self
            .air
            .lt_tuple_air
            .generate_trace_row((prev_idx_timestamp, curr_idx_timestamp, range_checker))
            .aux;

        if is_first_row {
            same_idx = 0;
            lt_bit = 1;
        }

        OfflineCheckerCols {
            clk: F::from_canonical_usize(curr_timestamp),
            idx: curr_idx,
            data: curr_data,
            op_type: F::from_canonical_u8(op_type),
            same_idx: F::from_canonical_u8(same_idx),
            is_valid: F::from_bool(is_valid),
            is_receive: F::from_bool(is_receive),
            lt_bit: F::from_canonical_u8(lt_bit),
            is_equal_idx_aux,
            lt_aux,
        }
    }
}
