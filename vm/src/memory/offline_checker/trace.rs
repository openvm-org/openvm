use std::sync::Arc;

use afs_primitives::{
    assert_less_than::columns::AssertLessThanAuxCols, is_zero::IsZeroAir,
    sub_chip::LocalTraceInstructions, var_range::VariableRangeCheckerChip,
};
use p3_field::PrimeField32;

use super::{
    bridge::{MemoryOfflineChecker, AUX_LEN},
    columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use crate::memory::{
    manager::{MemoryReadRecord, MemoryWriteRecord},
    offline_checker::MemoryReadOrImmediateAuxCols,
};

// NOTE[jpw]: The `make_*_aux_cols` functions should be thread-safe so they can be used in parallelized trace generation.
impl MemoryOfflineChecker {
    pub fn make_read_aux_cols<const N: usize, F: PrimeField32>(
        &self,
        range_checker: Arc<VariableRangeCheckerChip>,
        read: MemoryReadRecord<N, F>,
    ) -> MemoryReadAuxCols<N, F> {
        assert!(
            !read.address_space.is_zero(),
            "cannot make `MemoryReadAuxCols` for address space 0"
        );
        MemoryReadAuxCols::new(
            read.prev_timestamps,
            self.generate_timestamp_lt_cols(range_checker, &read.prev_timestamps, read.timestamp),
        )
    }

    pub fn make_read_or_immediate_aux_cols<F: PrimeField32>(
        &self,
        range_checker: Arc<VariableRangeCheckerChip>,
        read: MemoryReadRecord<1, F>,
    ) -> MemoryReadOrImmediateAuxCols<F> {
        let [prev_timestamp] = read.prev_timestamps;

        let addr_space_is_zero_cols = IsZeroAir.generate_trace_row(read.address_space);
        let [timestamp_lt_cols] =
            self.generate_timestamp_lt_cols(range_checker, &[prev_timestamp], read.timestamp);

        MemoryReadOrImmediateAuxCols::new(
            prev_timestamp,
            addr_space_is_zero_cols.io.is_zero,
            addr_space_is_zero_cols.inv,
            timestamp_lt_cols,
        )
    }

    pub fn make_write_aux_cols<const N: usize, F: PrimeField32>(
        &self,
        range_checker: Arc<VariableRangeCheckerChip>,
        write: MemoryWriteRecord<N, F>,
    ) -> MemoryWriteAuxCols<N, F> {
        MemoryWriteAuxCols::new(
            write.prev_data,
            write.prev_timestamps,
            self.generate_timestamp_lt_cols(range_checker, &write.prev_timestamps, write.timestamp),
        )
    }

    fn generate_timestamp_lt_cols<const N: usize, F: PrimeField32>(
        &self,
        range_checker: Arc<VariableRangeCheckerChip>,
        prev_timestamps: &[F; N],
        timestamp: F,
    ) -> [AssertLessThanAuxCols<F, AUX_LEN>; N] {
        prev_timestamps.map(|prev_timestamp| {
            debug_assert!(prev_timestamp.as_canonical_u32() < timestamp.as_canonical_u32());
            let mut aux: AssertLessThanAuxCols<F, AUX_LEN> =
                AssertLessThanAuxCols::<F, AUX_LEN>::new([F::zero(); AUX_LEN]);
            self.timestamp_lt_air.generate_trace_row_aux(
                prev_timestamp.as_canonical_u32(),
                timestamp.as_canonical_u32(),
                &range_checker,
                &mut aux,
            );
            aux
        })
    }
}
