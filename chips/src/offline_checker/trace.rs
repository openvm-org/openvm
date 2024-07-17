use std::sync::Arc;

use crate::is_less_than_tuple::columns::IsLessThanTupleIOCols;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::is_equal_vec::IsEqualVecAir;
use crate::is_less_than_tuple::IsLessThanTupleAir;
use crate::range_gate::RangeCheckerGateChip;
use crate::sub_chip::LocalTraceInstructions;

use super::{OfflineChecker, OfflineCheckerOperation};

impl<const WORD_SIZE: usize> OfflineChecker<WORD_SIZE> {
    /// Each row in the trace follow the same order as the Cols struct:
    /// [clk, mem_row, op_type, same_addr_space, same_pointer, same_addr, same_data, lt_bit, is_valid, is_equal_addr_space_aux, is_equal_pointer_aux, is_equal_data_aux, lt_aux]
    ///
    /// The trace consists of a row for every read/write operation plus some extra rows
    /// The trace is sorted by addr (addr_space and pointer) and then by clk, so every addr has a block of consective rows in the trace with the following structure
    /// A row is added to the trace for every read/write operation with the corresponding data
    /// The trace is padded at the end to be of height trace_degree
    pub fn generate_trace<F: PrimeField32, Operation: OfflineCheckerOperation<F>>(
        &mut self,
        range_checker: Arc<RangeCheckerGateChip>,
        // should be already sorted by address_space, address, timestamp
        accesses: Vec<Operation>,
        dummy_op: Operation,
    ) -> RowMajorMatrix<F> {
        let mut rows: Vec<F> = vec![];

        if !accesses.is_empty() {
            rows.extend(self.generate_trace_row((
                true,
                1,
                &accesses[0],
                &dummy_op,
                range_checker.clone(),
            )));
        }

        for i in 1..accesses.len() {
            rows.extend(self.generate_trace_row((
                false,
                1,
                &accesses[i],
                &accesses[i - 1],
                range_checker.clone(),
            )));
        }

        // Ensure that trace degree is a power of two
        let trace_degree = accesses.len().next_power_of_two();

        if accesses.len() < trace_degree {
            rows.extend(self.generate_trace_row((
                false,
                0,
                &dummy_op,
                &accesses[accesses.len() - 1],
                range_checker.clone(),
            )));
        }

        for _i in 1..(trace_degree - accesses.len()) {
            rows.extend(self.generate_trace_row((
                false,
                0,
                &dummy_op,
                &dummy_op,
                range_checker.clone(),
            )));
        }

        RowMajorMatrix::new(rows, self.air_width())
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32, Operation: OfflineCheckerOperation<F>>
    LocalTraceInstructions<F> for OfflineChecker<WORD_SIZE>
{
    type LocalInput = (bool, u8, Operation, Operation, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Vec<F> {
        let (is_first_row, is_valid, curr_op, prev_op, range_checker) = input;
        let mut row: Vec<F> = vec![];
        let op_type = curr_op.get_op_type();

        let curr_timestamp = curr_op.get_timestamp();
        let prev_timestamp = prev_op.get_timestamp();

        row.push(F::from_canonical_usize(curr_timestamp));
        row.extend(curr_op.get_idx());
        row.extend(curr_op.get_data());
        row.push(F::from_canonical_u8(op_type));

        let curr_idx = curr_op.get_idx();
        let prev_idx = prev_op.get_idx();
        let same_idx = if curr_idx == prev_idx { 1 } else { 0 };

        let curr_data = curr_op.get_data();
        let prev_data = prev_op.get_data();
        let same_data = if curr_data == prev_data { 1 } else { 0 };

        row.push(F::from_canonical_u8(same_idx));
        row.push(F::from_canonical_u8(same_data));
        row.push(F::from_canonical_u8(same_idx * same_data));

        let mut lt_bit = 1;
        for i in 0..curr_idx.len() {
            #[allow(clippy::comparison_chain)]
            if curr_idx[i] > prev_idx[i] {
                break;
            } else if curr_idx[i] < prev_idx[i] {
                lt_bit = 0;
                break;
            }

            if i == curr_idx.len() - 1 && curr_op.get_timestamp() <= prev_op.get_timestamp() {
                lt_bit = 0;
            }
        }

        row.push(F::from_canonical_u8(lt_bit));
        row.push(F::from_canonical_u8(is_valid));

        let is_equal_idx_air = IsEqualVecAir::new(WORD_SIZE);
        let is_equal_data_air = IsEqualVecAir::new(WORD_SIZE);
        let lt_air = IsLessThanTupleAir::new(
            range_checker.bus_index(),
            self.idx_clk_limb_bits.clone(),
            self.decomp,
        );

        let is_equal_idx_aux = is_equal_idx_air
            .generate_trace_row((prev_idx.clone(), curr_idx.clone()))
            .aux
            .flatten();

        let is_equal_data_aux = is_equal_data_air
            .generate_trace_row((prev_data, curr_data))
            .aux
            .flatten();

        let mut prev_idx_timestamp = prev_idx
            .clone()
            .into_iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        prev_idx_timestamp.push(prev_timestamp as u32);

        let mut curr_idx_timestamp = curr_idx
            .clone()
            .into_iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        curr_idx_timestamp.push(curr_timestamp as u32);

        let lt_aux: Vec<F> = lt_air
            .generate_trace_row((prev_idx_timestamp, curr_idx_timestamp, range_checker))
            .flatten()[IsLessThanTupleIOCols::<F>::get_width(self.idx_len + 1)..]
            .to_vec();

        row.extend(is_equal_idx_aux);
        row.extend(is_equal_data_aux);
        row.extend(lt_aux);

        let idx_data_len = self.idx_len + self.data_len;

        if is_first_row {
            // same_idx should be 0
            row[2 + idx_data_len] = F::zero();
            // same_data should be 0
            row[3 + idx_data_len] = F::zero();
            // same_idx_and_data should be 0
            row[4 + idx_data_len] = F::zero();
            // lt_bit should be 1
            row[5 + idx_data_len] = F::one();
        }

        row
    }
}
