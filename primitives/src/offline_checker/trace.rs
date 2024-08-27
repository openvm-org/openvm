use std::sync::Arc;

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{OfflineCheckerCols, OfflineCheckerColsMut},
    OfflineCheckerChip, OfflineCheckerOperation,
};
use crate::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};

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
                LocalTraceInstructions::generate_trace_row(
                    self,
                    (
                        true,
                        true,
                        true,
                        accesses[0].clone(),
                        dummy_op.clone(),
                        range_checker.clone(),
                    ),
                )
                .flatten(),
            );
        }

        for i in 1..accesses.len() {
            rows.push(
                LocalTraceInstructions::generate_trace_row(
                    self,
                    (
                        false,
                        true,
                        true,
                        accesses[i].clone(),
                        accesses[i - 1].clone(),
                        range_checker.clone(),
                    ),
                )
                .flatten(),
            );
        }

        if accesses.len() < trace_degree {
            rows.push(
                LocalTraceInstructions::generate_trace_row(
                    self,
                    (
                        false,
                        false,
                        false,
                        dummy_op.clone(),
                        accesses[accesses.len() - 1].clone(),
                        range_checker.clone(),
                    ),
                )
                .flatten(),
            );
        }

        for _i in 1..(trace_degree - accesses.len()) {
            rows.push(
                LocalTraceInstructions::generate_trace_row(
                    self,
                    (
                        false,
                        false,
                        false,
                        dummy_op.clone(),
                        dummy_op.clone(),
                        range_checker.clone(),
                    ),
                )
                .flatten(),
            );
        }

        RowMajorMatrix::new(rows.concat(), self.air.air_width())
    }
}

impl<F: PrimeField64, Operation: OfflineCheckerOperation<F>> OfflineCheckerChip<F, Operation> {
    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace_row(
        &self,
        is_first_row: bool,
        is_valid: bool,
        is_receive: bool,
        curr_op: &Operation,
        prev_op: &Operation,
        range_checker: Arc<RangeCheckerGateChip>,
        oc_cols: &mut OfflineCheckerColsMut<F>,
    ) {
        let op_type = curr_op.get_op_type();

        let curr_timestamp = curr_op.get_timestamp();
        let prev_timestamp = prev_op.get_timestamp();

        let curr_idx = curr_op.get_idx();
        let prev_idx = prev_op.get_idx();
        let mut same_idx = if curr_idx == prev_idx { 1 } else { 0 };

        let curr_data = curr_op.get_data();

        let mut lt_bit = (&prev_idx, prev_timestamp) < (&curr_idx, curr_timestamp);

        self.air.is_equal_idx_air.generate_trace_row_aux(
            &prev_idx,
            &curr_idx,
            &mut oc_cols.is_equal_idx_aux,
        );

        // TODO: I don't like all the conversions
        let mut prev_idx_clk = Vec::with_capacity(prev_idx.len() + 1);
        prev_idx_clk.extend(
            prev_idx
                .clone()
                .into_iter()
                .map(|x| x.as_canonical_u64() as u32),
        );
        prev_idx_clk.push(prev_timestamp as u32);

        let mut curr_idx_clk = Vec::with_capacity(curr_idx.len() + 1);
        curr_idx_clk.extend(
            curr_idx
                .clone()
                .into_iter()
                .map(|x| x.as_canonical_u64() as u32),
        );
        curr_idx_clk.push(curr_timestamp as u32);

        self.air.lt_tuple_air.generate_trace_row_aux(
            &prev_idx_clk,
            &curr_idx_clk,
            &range_checker,
            &mut oc_cols.lt_aux,
        );

        if is_first_row {
            same_idx = 0;
            lt_bit = true;
        }

        *oc_cols.clk = F::from_canonical_usize(curr_timestamp);
        oc_cols.idx.clone_from_slice(&curr_idx);
        oc_cols.data.clone_from_slice(&curr_data);
        *oc_cols.op_type = F::from_canonical_u8(op_type);
        *oc_cols.same_idx = F::from_canonical_u8(same_idx);
        *oc_cols.is_valid = F::from_bool(is_valid);
        *oc_cols.is_receive = F::from_bool(is_receive);
        *oc_cols.lt_bit = F::from_bool(lt_bit);
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

        let width: usize = OfflineCheckerCols::<F>::width(&self.air);

        let mut row = vec![F::zero(); width];
        let mut oc_cols = OfflineCheckerColsMut::<F>::from_slice(&mut row, &self.air);

        self.generate_trace_row(
            is_first_row,
            is_valid,
            is_receive,
            &curr_op,
            &prev_op,
            range_checker,
            &mut oc_cols,
        );

        OfflineCheckerCols::<F>::from_slice(&row, &self.air)
    }
}
