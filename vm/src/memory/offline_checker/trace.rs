use std::{array::from_fn, sync::Arc};

use afs_primitives::{
    is_zero::IsZeroAir, range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions,
};
use p3_field::PrimeField32;

use super::{
    bridge::MemoryOfflineChecker,
    columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use crate::memory::manager::{MemoryReadRecord, MemoryWriteRecord};

impl MemoryOfflineChecker {
    // NOTE[jpw]: this function should be thread-safe so it can be used in parallelized
    // trace generation
    pub fn make_read_aux_cols<const N: usize, F: PrimeField32>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
        read: MemoryReadRecord<N, F>,
    ) -> MemoryReadAuxCols<N, F> {
        let timestamp = read.timestamp.as_canonical_u32();
        for prev_timestamp in &read.prev_timestamps {
            debug_assert!(prev_timestamp.as_canonical_u32() < timestamp);
        }

        let clk_lt_cols = from_fn(|i| {
            LocalTraceInstructions::generate_trace_row(
                &self.timestamp_lt_air,
                (
                    read.prev_timestamps[i].as_canonical_u32(),
                    timestamp,
                    range_checker.clone(),
                ),
            )
        });

        let addr_space_is_zero_cols = IsZeroAir.generate_trace_row(read.address_space);

        MemoryReadAuxCols::new(
            read.prev_timestamps,
            addr_space_is_zero_cols.io.is_zero,
            addr_space_is_zero_cols.inv,
            clk_lt_cols.clone().map(|x| x.io.less_than),
            clk_lt_cols.map(|x| x.aux),
        )
    }

    // NOTE[jpw]: this function should be thread-safe so it can be used in parallelized
    // trace generation
    pub fn make_write_aux_cols<const N: usize, F: PrimeField32>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
        write: MemoryWriteRecord<N, F>,
    ) -> MemoryWriteAuxCols<N, F> {
        let timestamp = write.timestamp.as_canonical_u32();
        for prev_timestamp in &write.prev_timestamps {
            debug_assert!(prev_timestamp.as_canonical_u32() < timestamp);
        }

        let clk_lt_cols = from_fn(|i| {
            LocalTraceInstructions::generate_trace_row(
                &self.timestamp_lt_air,
                (
                    write.prev_timestamps[i].as_canonical_u32(),
                    timestamp,
                    range_checker.clone(),
                ),
            )
        });

        MemoryWriteAuxCols::new(
            write.prev_data,
            write.prev_timestamps,
            clk_lt_cols.clone().map(|x| x.io.less_than),
            clk_lt_cols.map(|x| x.aux),
        )
    }
}
