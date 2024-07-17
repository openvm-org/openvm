use std::sync::Arc;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::is_equal_vec::IsEqualVecAir;
use crate::is_less_than_tuple::IsLessThanTupleAir;
use crate::range_gate::RangeCheckerGateChip;
use crate::sub_chip::LocalTraceInstructions;

use super::{columns::OfflineCheckerCols, OfflineCheckerChip, OfflineCheckerOperation};

impl<const WORD_SIZE: usize, F: PrimeField32, Operation: OfflineCheckerOperation<F> + Clone>
    OfflineCheckerChip<WORD_SIZE, F, Operation>
{
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
        // should be already sorted by address_space, address, timestamp
        accesses: Vec<Operation>,
        dummy_op: Operation,
    ) -> RowMajorMatrix<F> {
        let mut rows: Vec<F> = vec![];

        if !accesses.is_empty() {
            rows.extend(
                self.generate_trace_row((
                    true,
                    1,
                    accesses[0].clone(),
                    dummy_op.clone(),
                    range_checker.clone(),
                ))
                .flatten(),
            );
        }

        for i in 1..accesses.len() {
            rows.extend(
                self.generate_trace_row((
                    false,
                    1,
                    accesses[i].clone(),
                    accesses[i - 1].clone(),
                    range_checker.clone(),
                ))
                .flatten(),
            );
        }

        // Ensure that trace degree is a power of two
        let trace_degree = accesses.len().next_power_of_two();

        if accesses.len() < trace_degree {
            rows.extend(
                self.generate_trace_row((
                    false,
                    0,
                    dummy_op.clone(),
                    accesses[accesses.len() - 1].clone(),
                    range_checker.clone(),
                ))
                .flatten(),
            );
        }

        for _i in 1..(trace_degree - accesses.len()) {
            rows.extend(
                self.generate_trace_row((
                    false,
                    0,
                    dummy_op.clone(),
                    dummy_op.clone(),
                    range_checker.clone(),
                ))
                .flatten(),
            );
        }

        RowMajorMatrix::new(rows, self.air.air_width())
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32, Operation: OfflineCheckerOperation<F>>
    LocalTraceInstructions<F> for OfflineCheckerChip<WORD_SIZE, F, Operation>
{
    type LocalInput = (bool, u8, Operation, Operation, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> OfflineCheckerCols<F> {
        let (is_first_row, is_valid, curr_op, prev_op, range_checker) = input;
        let op_type = curr_op.get_op_type();

        let curr_timestamp = curr_op.get_timestamp();
        let prev_timestamp = prev_op.get_timestamp();

        let curr_idx = curr_op.get_idx();
        let prev_idx = prev_op.get_idx();
        let mut same_idx = if curr_idx == prev_idx { 1 } else { 0 };

        let curr_data = curr_op.get_data();
        let prev_data = prev_op.get_data();
        let mut same_data = if curr_data == prev_data { 1 } else { 0 };

        let mut same_idx_and_data = same_idx * same_data;

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

        let is_equal_idx_air = IsEqualVecAir::new(WORD_SIZE);
        let is_equal_data_air = IsEqualVecAir::new(WORD_SIZE);
        let lt_air = IsLessThanTupleAir::new(
            range_checker.bus_index(),
            self.air.idx_clk_limb_bits.clone(),
            self.air.decomp,
        );

        let is_equal_idx_aux = is_equal_idx_air
            .generate_trace_row((prev_idx.clone(), curr_idx.clone()))
            .aux;

        let is_equal_data_aux = is_equal_data_air
            .generate_trace_row((prev_data.clone(), curr_data.clone()))
            .aux;

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

        let lt_aux = lt_air
            .generate_trace_row((prev_idx_timestamp, curr_idx_timestamp, range_checker))
            .aux;

        if is_first_row {
            same_idx = 0;
            same_data = 0;
            same_idx_and_data = 0;
            lt_bit = 1;
        }

        OfflineCheckerCols {
            clk: F::from_canonical_usize(curr_timestamp),
            idx: curr_idx,
            data: curr_data,
            op_type: F::from_canonical_u8(op_type),
            same_idx: F::from_canonical_u8(same_idx),
            same_data: F::from_canonical_u8(same_data),
            same_idx_and_data: F::from_canonical_u8(same_idx_and_data),
            is_valid: F::from_canonical_u8(is_valid),
            lt_bit: F::from_canonical_u8(lt_bit),
            is_equal_idx_aux,
            is_equal_data_aux,
            lt_aux,
        }
    }
}
