use std::{array::from_fn, sync::Arc};

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;

use super::{
    bridge::MemoryOfflineChecker,
    columns::{MemoryOfflineCheckerAuxCols, MemoryReadAuxCols, MemoryWriteAuxCols},
};
use crate::memory::manager::{MemoryReadRecord, MemoryWriteRecord};

impl MemoryOfflineChecker {
    pub fn make_read_aux_cols<const N: usize, F: PrimeField32>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
        read: MemoryReadRecord<N, F>,
    ) -> MemoryReadAuxCols<N, F> {
        self.make_aux_cols(
            range_checker,
            read.timestamp,
            read.address_space,
            read.data,
            read.prev_timestamps,
        )
    }

    pub fn make_write_aux_cols<const N: usize, F: PrimeField32>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
        write: MemoryWriteRecord<N, F>,
    ) -> MemoryWriteAuxCols<N, F> {
        self.make_aux_cols(
            range_checker,
            write.timestamp,
            write.address_space,
            write.prev_data,
            write.prev_timestamps,
        )
    }

    // NOTE[jpw]: this function should be thread-safe so it can be used in parallelized
    // trace generation
    pub fn make_aux_cols<const N: usize, F: PrimeField32>(
        &self,
        range_checker: Arc<RangeCheckerGateChip>,
        timestamp: F,
        address_space: F,
        prev_data: [F; N],
        prev_timestamps: [F; N],
    ) -> MemoryOfflineCheckerAuxCols<N, F> {
        let timestamp = timestamp.as_canonical_u32();
        for prev_timestamp in &prev_timestamps {
            debug_assert!(prev_timestamp.as_canonical_u32() < timestamp);
        }

        let clk_lt_cols = from_fn(|i| {
            LocalTraceInstructions::generate_trace_row(
                &self.timestamp_lt_air,
                (
                    prev_timestamps[i].as_canonical_u32(),
                    timestamp,
                    range_checker.clone(),
                ),
            )
        });

        let addr_space_is_zero_cols = self.is_zero_air.generate_trace_row(address_space);

        MemoryOfflineCheckerAuxCols::new(
            prev_data,
            prev_timestamps,
            addr_space_is_zero_cols.io.is_zero,
            addr_space_is_zero_cols.inv,
            clk_lt_cols.clone().map(|x| x.io.less_than),
            clk_lt_cols.map(|x| x.aux),
        )
    }
}
