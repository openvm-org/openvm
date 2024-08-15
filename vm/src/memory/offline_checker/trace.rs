use std::sync::Arc;

use afs_primitives::{
    offline_checker::OfflineCheckerChip, range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "parallel")]
use p3_maybe_rayon::prelude::*;

use super::{
    air::NewMemoryOfflineChecker,
    columns::{MemoryOfflineCheckerAuxCols, MemoryOfflineCheckerCols, NewMemoryAccess},
    MemoryChip,
};
use crate::memory::{
    manager::{access_cell::AccessCell, operation::MemoryOperation},
    MemoryAccess, OpType,
};

// TODO[osama]: to be deleted
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

        let mut offline_checker_chip = OfflineCheckerChip::new(self.air.offline_checker.clone());

        offline_checker_chip.generate_trace(
            range_checker,
            self.accesses.clone(),
            dummy_op,
            self.accesses.len().next_power_of_two(),
        )
    }
}

impl<const WORD_SIZE: usize> NewMemoryOfflineChecker<WORD_SIZE> {
    pub fn memory_access_to_checker_aux_cols<F: PrimeField32>(
        &self,
        memory_access: &NewMemoryAccess<WORD_SIZE, F>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        let clk_lt_cols = LocalTraceInstructions::generate_trace_row(
            &self.clk_lt_air,
            (
                memory_access.old_cell.clk.as_canonical_u32(),
                memory_access.op.cell.clk.as_canonical_u32(),
                range_checker.clone(),
            ),
        );

        let addr_space_is_zero_cols = LocalTraceInstructions::generate_trace_row(
            &self.is_zero_air,
            memory_access.op.addr_space,
        );

        MemoryOfflineCheckerAuxCols::new(
            memory_access.old_cell,
            addr_space_is_zero_cols.io.is_zero,
            addr_space_is_zero_cols.inv,
            clk_lt_cols.io.less_than,
            clk_lt_cols.aux,
        )
    }

    pub fn memory_access_to_checker_cols<F: PrimeField32>(
        &self,
        memory_access: &NewMemoryAccess<WORD_SIZE, F>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerCols<WORD_SIZE, F> {
        MemoryOfflineCheckerCols::<WORD_SIZE, F>::new(
            memory_access.op.clone(),
            self.memory_access_to_checker_aux_cols(memory_access, range_checker.clone()),
        )
    }

    pub fn disabled_memory_checker_aux_cols<F: PrimeField32>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        self.memory_access_to_checker_aux_cols(
            &NewMemoryAccess::<WORD_SIZE, F>::new(
                MemoryOperation::new(
                    F::zero(),
                    F::zero(),
                    F::from_canonical_u8(OpType::Read as u8),
                    AccessCell::new([F::zero(); WORD_SIZE], F::zero()),
                    F::zero(),
                ),
                AccessCell::new([F::zero(); WORD_SIZE], F::zero()),
            ),
            range_checker,
        )
    }

    pub fn disabled_memory_checker_cols<F: PrimeField32>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerCols<WORD_SIZE, F> {
        self.memory_access_to_checker_cols(
            &NewMemoryAccess::<WORD_SIZE, F>::new(
                MemoryOperation::new(
                    F::zero(),
                    F::zero(),
                    F::from_canonical_u8(OpType::Read as u8),
                    AccessCell::new([F::zero(); WORD_SIZE], F::zero()),
                    F::zero(),
                ),
                AccessCell::new([F::zero(); WORD_SIZE], F::zero()),
            ),
            range_checker,
        )
    }
}
