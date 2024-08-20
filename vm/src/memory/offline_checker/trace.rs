use std::sync::Arc;

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;
#[cfg(feature = "parallel")]
use p3_maybe_rayon::prelude::*;

use super::{
    bridge::MemoryOfflineChecker,
    columns::{MemoryAccess, MemoryOfflineCheckerAuxCols, MemoryOfflineCheckerCols},
};
use crate::memory::{
    manager::{access_cell::AccessCell, operation::MemoryOperation},
    OpType,
};

impl MemoryOfflineChecker {
    pub fn memory_access_to_checker_aux_cols<F: PrimeField32, const WORD_SIZE: usize>(
        &self,
        memory_access: &MemoryAccess<WORD_SIZE, F>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        let clk_lt_cols = LocalTraceInstructions::generate_trace_row(
            &self.timestamp_lt_air,
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

    pub fn memory_access_to_checker_cols<F: PrimeField32, const WORD_SIZE: usize>(
        &self,
        memory_access: &MemoryAccess<WORD_SIZE, F>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerCols<WORD_SIZE, F> {
        MemoryOfflineCheckerCols::<WORD_SIZE, F>::new(
            memory_access.op.clone(),
            self.memory_access_to_checker_aux_cols(memory_access, range_checker.clone()),
        )
    }

    pub fn disabled_memory_checker_aux_cols_from_op<F: PrimeField32, const WORD_SIZE: usize>(
        &self,
        addr_space: F,
        clk: F,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        self.memory_access_to_checker_aux_cols(
            &MemoryAccess::<WORD_SIZE, F>::new(
                MemoryOperation::new(
                    addr_space,
                    F::zero(),
                    F::zero(),
                    AccessCell::new([F::zero(); WORD_SIZE], clk),
                    F::zero(),
                ),
                AccessCell::new([F::zero(); WORD_SIZE], F::zero()),
            ),
            range_checker,
        )
    }

    /// Assumes that addr_space in memory operation is zero
    pub fn disabled_memory_checker_aux_cols<F: PrimeField32, const WORD_SIZE: usize>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        self.memory_access_to_checker_aux_cols(
            &MemoryAccess::<WORD_SIZE, F>::new(
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

    /// Assumes that IO memory operation is all zeros
    pub fn disabled_memory_checker_cols<F: PrimeField32, const WORD_SIZE: usize>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryOfflineCheckerCols<WORD_SIZE, F> {
        self.memory_access_to_checker_cols(
            &MemoryAccess::<WORD_SIZE, F>::new(
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
