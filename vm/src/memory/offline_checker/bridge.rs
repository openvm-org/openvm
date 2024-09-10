use std::{iter::zip, marker::PhantomData};

use afs_primitives::{
    is_less_than::{columns::IsLessThanIoCols, IsLessThanAir},
    is_zero::{
        columns::{IsZeroCols, IsZeroIoCols},
        IsZeroAir,
    },
    utils::not,
    var_range::bus::VariableRangeCheckerBus,
};
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::AirBuilder;
use p3_field::AbstractField;

use super::bus::MemoryBus;
use crate::{
    cpu::RANGE_CHECKER_BUS,
    memory::{
        offline_checker::columns::{
            MemoryBaseAuxCols, MemoryReadAuxCols, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols,
        },
        MemoryAddress,
    },
};

/// The [MemoryBridge] is used within AIR evaluation functions to constrain logical memory operations (read/write).
/// It adds all necessary constraints and interactions.
#[derive(Clone, Debug)]
pub struct MemoryBridge<V> {
    offline_checker: MemoryOfflineChecker,
    _marker: PhantomData<V>,
}

impl<V> MemoryBridge<V> {
    /// Create a new [MemoryBridge] with the provided offline_checker.
    pub fn new(offline_checker: MemoryOfflineChecker) -> Self {
        Self {
            offline_checker,
            _marker: PhantomData,
        }
    }

    /// Prepare a logical memory read operation.
    #[must_use]
    pub fn read<T, const N: usize>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; N],
        timestamp: impl Into<T>,
        aux: MemoryReadAuxCols<N, V>,
    ) -> MemoryReadOperation<T, V, N> {
        MemoryReadOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            timestamp: timestamp.into(),
            aux,
        }
    }

    /// Prepare a logical memory read or immediate operation.
    #[must_use]
    pub fn read_or_immediate<T>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: impl Into<T>,
        timestamp: impl Into<T>,
        aux: MemoryReadOrImmediateAuxCols<V>,
    ) -> MemoryReadOrImmediateOperation<T, V> {
        MemoryReadOrImmediateOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.into(),
            timestamp: timestamp.into(),
            aux,
        }
    }

    /// Prepare a logical memory write operation.
    #[must_use]
    pub fn write<T, const N: usize>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; N],
        timestamp: impl Into<T>,
        aux: MemoryWriteAuxCols<N, V>,
    ) -> MemoryWriteOperation<T, V, N> {
        MemoryWriteOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            timestamp: timestamp.into(),
            aux,
        }
    }
}

/// Constraints and interactions for a logical memory read of `(address, data)` at time `timestamp`.
/// This reads `(address, data, timestamp_prev)` from the memory bus and writes
/// `(address, data, timestamp)` to the memory bus.
/// Includes constraints for `timestamp_prev < timestamp`.
///
/// The generic `T` type is intended to be `AB::Expr` where `AB` is the [AirBuilder].
/// The auxiliary columns are not expected to be expressions, so the generic `V` type is intended
/// to be `AB::Var`.
pub struct MemoryReadOperation<T, V, const N: usize> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; N],
    timestamp: T,
    aux: MemoryReadAuxCols<N, V>,
}

impl<F: AbstractField, V: Copy + Into<F>, const N: usize> MemoryReadOperation<F, V, N> {
    /// Evaluate constraints and send/receive interactions.
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Var = V, Expr = F>,
    {
        let enabled = enabled.into();

        // NOTE: We do not need to constrain `address_space != 0` since this is done implicitly by
        // the memory interactions argument together with initial/final memory chips.

        self.offline_checker.eval_timestamps(
            builder,
            self.timestamp.clone(),
            &self.aux.base,
            enabled.clone(),
        );

        self.offline_checker.eval_bulk_access(
            builder,
            &self.address,
            &self.data.clone(),
            &self.data,
            &self.timestamp,
            &self.aux.base.prev_timestamps,
            enabled,
        );
    }
}

/// Constraints and interactions for a logical memory read of `(address, data)` at time `timestamp`,
/// supporting `address.address_space = 0` for immediates.
///
/// If `address.address_space` is non-zero, it behaves like `MemoryReadOperation`. Otherwise,
/// it constrains the immediate value appropriately.
///
/// The generic `T` type is intended to be `AB::Expr` where `AB` is the [AirBuilder].
/// The auxiliary columns are not expected to be expressions, so the generic `V` type is intended
/// to be `AB::Var`.
pub struct MemoryReadOrImmediateOperation<T, V> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: T,
    timestamp: T,
    aux: MemoryReadOrImmediateAuxCols<V>,
}

impl<F: AbstractField, V: Copy + Into<F>> MemoryReadOrImmediateOperation<F, V> {
    /// Evaluate constraints and send/receive interactions.
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Var = V, Expr = F>,
    {
        let enabled = enabled.into();

        // `is_immediate` should be an indicator for `address_space == 0` (when `enabled`).
        {
            let addr_space_is_zero_cols = IsZeroCols::new(
                IsZeroIoCols::new(
                    self.address.address_space.clone(),
                    self.aux.is_immediate.into(),
                ),
                self.aux.is_zero_aux.into(),
            );
            IsZeroAir.subair_eval(
                &mut builder.when(enabled.clone()), // when not enabled, allow aux to be all 0s no matter what
                addr_space_is_zero_cols.io,
                addr_space_is_zero_cols.inv,
            );
        }
        // When `is_immediate`, the data should be the pointer value.
        builder
            .when(self.aux.is_immediate)
            .assert_eq(self.data.clone(), self.address.pointer.clone());

        // Timestamps should be increasing (when enabled).
        self.offline_checker.eval_timestamps(
            builder,
            self.timestamp.clone(),
            &self.aux.base,
            enabled.clone(),
        );

        self.offline_checker.eval_bulk_access(
            builder,
            &self.address,
            &[self.data.clone()],
            &[self.data],
            &self.timestamp,
            &self.aux.base.prev_timestamps,
            enabled * not(self.aux.is_immediate),
        );
    }
}

/// Constraints and interactions for a logical memory write of `(address, data)` at time `timestamp`.
/// This reads `(address, data_prev, timestamp_prev)` from the memory bus and writes
/// `(address, data, timestamp)` to the memory bus.
/// Includes constraints for `timestamp_prev < timestamp`.
///
/// **Note:** This can be used as a logical read operation by setting `data_prev = data`.
pub struct MemoryWriteOperation<T, V, const N: usize> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; N],
    /// The timestamp of the current read
    timestamp: T,
    aux: MemoryWriteAuxCols<N, V>,
}

impl<T: AbstractField, V: Copy + Into<T>, const N: usize> MemoryWriteOperation<T, V, N> {
    /// Evaluate constraints and send/receive interactions. `enabled` must be boolean.
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Var = V, Expr = T>,
    {
        let enabled = enabled.into();
        self.offline_checker.eval_timestamps(
            builder,
            self.timestamp.clone(),
            &self.aux.base,
            enabled.clone(),
        );

        self.offline_checker.eval_bulk_access(
            builder,
            &self.address,
            &self.data,
            &self.aux.prev_data.map(Into::into),
            &self.timestamp,
            &self.aux.base.prev_timestamps,
            enabled,
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryOfflineChecker {
    pub memory_bus: MemoryBus,
    pub timestamp_lt_air: IsLessThanAir,
}

impl MemoryOfflineChecker {
    // TODO[jpw]: pass in range bus
    pub fn new(memory_bus: MemoryBus, clk_max_bits: usize, decomp: usize) -> Self {
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, decomp);
        Self {
            memory_bus,
            timestamp_lt_air: IsLessThanAir::new(range_bus, clk_max_bits),
        }
    }

    fn eval_timestamps<AB: InteractionBuilder, const N: usize>(
        &self,
        builder: &mut AB,
        timestamp: AB::Expr,
        base: &MemoryBaseAuxCols<AB::Var, N>,
        enabled: AB::Expr,
    ) {
        for (prev_timestamp, clk_lt_aux) in zip(base.prev_timestamps, base.clk_lt_aux.clone()) {
            let clk_lt_io_cols =
                IsLessThanIoCols::<AB::Expr>::new(prev_timestamp, timestamp.clone(), AB::F::one());
            self.timestamp_lt_air.conditional_eval(
                builder,
                clk_lt_io_cols,
                clk_lt_aux,
                enabled.clone(),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_bulk_access<AB, const N: usize>(
        &self,
        builder: &mut AB,
        address: &MemoryAddress<AB::Expr, AB::Expr>,
        data: &[AB::Expr; N],
        prev_data: &[AB::Expr; N],
        timestamp: &AB::Expr,
        prev_timestamps: &[AB::Var; N],
        enabled: AB::Expr,
    ) where
        AB: InteractionBuilder,
    {
        for (i, (&prev_timestamp, prev_datum, datum)) in
            izip!(prev_timestamps, prev_data, data).enumerate()
        {
            let address = MemoryAddress::new(
                address.address_space.clone(),
                address.pointer.clone() + AB::Expr::from_canonical_usize(i),
            );

            self.memory_bus
                .read(address.clone(), [prev_datum.clone()], prev_timestamp)
                .eval(builder, enabled.clone());

            self.memory_bus
                .write(address, [datum.clone()], timestamp.clone())
                .eval(builder, enabled.clone());
        }
    }
}
