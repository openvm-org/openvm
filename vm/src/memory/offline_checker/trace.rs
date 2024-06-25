use std::sync::Arc;

use afs_chips::is_equal::IsEqualAir;
use afs_chips::is_less_than_tuple::columns::IsLessThanTupleIOCols;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::{MemoryAccess, OpType};

use super::OfflineChecker;
use afs_chips::is_equal_vec::IsEqualVecAir;
use afs_chips::is_less_than_tuple::IsLessThanTupleAir;
use afs_chips::range_gate::RangeCheckerGateChip;
use afs_chips::sub_chip::LocalTraceInstructions;

impl OfflineChecker {
    /// Each row in the trace follow the same order as the Cols struct:
    /// [clk, mem_row, op_type, same_addr_space, same_pointer, same_addr, same_data, lt_bit, is_valid, is_equal_addr_space_aux, is_equal_pointer_aux, is_equal_data_aux, lt_aux]
    ///
    /// The trace consists of a row for every read/write operation plus some extra rows
    /// The trace is sorted by addr (addr_space and pointer) and then by clk, so every addr has a block of consective rows in the trace with the following structure
    /// A row is added to the trace for every read/write operation with the corresponding data
    /// The trace is padded at the end to be of height trace_degree
    pub fn generate_trace<F: PrimeField32>(
        &self,
        mut ops: Vec<MemoryAccess<F>>,
        range_checker: Arc<RangeCheckerGateChip>,
        trace_degree: usize,
    ) -> RowMajorMatrix<F> {
        ops.sort_by_key(|op| (op.address_space, op.address, op.timestamp));

        let mut rows: Vec<F> = vec![];

        let dummy_op = MemoryAccess {
            timestamp: 0,
            op_type: OpType::Read,
            address_space: F::zero(),
            address: F::zero(),
            data: vec![F::zero(); self.data_len],
        };

        if !ops.is_empty() {
            rows.extend(self.generate_trace_row::<F>(
                true,
                1,
                &ops[0],
                &dummy_op,
                range_checker.clone(),
            ));
        }

        for i in 1..ops.len() {
            rows.extend(self.generate_trace_row::<F>(
                false,
                1,
                &ops[i],
                &ops[i - 1],
                range_checker.clone(),
            ));
        }

        // Ensure that trace degree is a power of two
        assert!(trace_degree > 0 && trace_degree & (trace_degree - 1) == 0);

        if ops.len() < trace_degree {
            rows.extend(self.generate_trace_row::<F>(
                false,
                0,
                &dummy_op,
                &ops[ops.len() - 1],
                range_checker.clone(),
            ));
        }

        for _i in 1..(trace_degree - ops.len()) {
            rows.extend(self.generate_trace_row::<F>(
                false,
                0,
                &dummy_op,
                &dummy_op,
                range_checker.clone(),
            ));
        }

        RowMajorMatrix::new(rows, self.air_width())
    }

    pub fn generate_trace_row<F: PrimeField32>(
        &self,
        is_first_row: bool,
        is_valid: u8,
        curr_op: &MemoryAccess<F>,
        prev_op: &MemoryAccess<F>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Vec<F> {
        let mut row: Vec<F> = vec![];
        let op_type = (curr_op.op_type == OpType::Write) as u8;

        row.push(F::from_canonical_usize(curr_op.timestamp));
        row.push(curr_op.address_space);
        row.push(curr_op.address);
        row.extend(curr_op.data.clone());
        row.push(F::from_canonical_u8(op_type));

        let same_addr_space = if curr_op.address_space == prev_op.address_space {
            1
        } else {
            0
        };
        let same_pointer = if curr_op.address == prev_op.address {
            1
        } else {
            0
        };
        let same_addr = same_addr_space * same_pointer;
        let same_data = if curr_op.data == prev_op.data { 1 } else { 0 };

        row.push(F::from_canonical_u8(same_addr_space));
        row.push(F::from_canonical_u8(same_pointer));
        row.push(F::from_canonical_u8(same_addr));
        row.push(F::from_canonical_u8(same_data));

        let lt_bit = if curr_op.address_space > prev_op.address_space
            || (curr_op.address_space == prev_op.address_space && curr_op.address > prev_op.address)
            || (curr_op.address_space == prev_op.address_space
                && curr_op.address == prev_op.address
                && curr_op.timestamp > prev_op.timestamp)
        {
            1
        } else {
            0
        };

        row.push(F::from_canonical_u8(lt_bit));
        row.push(F::from_canonical_u8(is_valid));

        let is_equal_addr_space_air = IsEqualAir {};
        let is_equal_pointer_air = IsEqualAir {};
        let is_equal_data_air = IsEqualVecAir::new(self.data_len);
        let lt_air = IsLessThanTupleAir::new(
            range_checker.bus_index(),
            self.addr_clk_limb_bits.clone(),
            self.decomp,
        );

        let is_equal_addr_space_aux = is_equal_addr_space_air
            .generate_trace_row((prev_op.address_space, curr_op.address_space))
            .flatten()[3];
        let is_equal_pointer_aux = is_equal_pointer_air
            .generate_trace_row((prev_op.address, curr_op.address))
            .flatten()[3];
        let is_equal_data_aux = is_equal_data_air
            .generate_trace_row((prev_op.data.clone(), curr_op.data.clone()))
            .flatten()[2 * self.data_len..]
            .to_vec();
        let lt_aux: Vec<F> = lt_air
            .generate_trace_row((
                vec![
                    prev_op.address_space.as_canonical_u32(),
                    prev_op.address.as_canonical_u32(),
                    prev_op.timestamp as u32,
                ],
                vec![
                    curr_op.address_space.as_canonical_u32(),
                    curr_op.address.as_canonical_u32(),
                    curr_op.timestamp as u32,
                ],
                range_checker,
            ))
            .flatten()[IsLessThanTupleIOCols::<F>::get_width(3)..]
            .to_vec();

        row.push(is_equal_addr_space_aux);
        row.push(is_equal_pointer_aux);
        row.extend(is_equal_data_aux);
        row.extend(lt_aux);

        if is_first_row {
            // same_addr_space should be 0
            row[2 + self.mem_width()] = F::zero();
            // same_pointer should be 0
            row[3 + self.mem_width()] = F::zero();
            // same_addr should be 0
            row[4 + self.mem_width()] = F::zero();
            // same_data should be 0
            row[5 + self.mem_width()] = F::zero();
            // lt_bit should be 1
            row[6 + self.mem_width()] = F::one();
        }

        row
    }
}
