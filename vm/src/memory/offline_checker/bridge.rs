use std::iter::zip;
use std::marker::PhantomData;

use afs_primitives::{
    is_less_than::{columns::IsLessThanIoCols, IsLessThanAir},
    range::bus::RangeCheckBus,
    utils::not,
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::AirBuilder;
use p3_field::AbstractField;

use super::bus::MemoryBus;
use crate::{
    cpu::RANGE_CHECKER_BUS,
    memory::{
        offline_checker::columns::{MemoryBaseAuxCols, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};

/// The [MemoryBridge] can be created within any AIR evaluation function to be used as the
/// interface for constraining logical memory read or write operations. The bridge will add
/// all necessary constraints and interactions.
///
/// ## Usage
/// [MemoryBridge] must be initialized with the correct number of auxiliary columns to match the
/// exact number of memory operations to be constrained.
#[derive(Clone, Debug)]
pub struct MemoryBridge<V> {
    offline_checker: MemoryOfflineChecker,
    _marker: PhantomData<V>,
}

impl<V> MemoryBridge<V> {
    /// Create a new [MemoryBridge] with the given number of auxiliary columns.
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

// **TODO[jpw]**: Read does not need duplicate trace cells for old_data and data since they are the same.
// **Move old_cell out of AuxCols**
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
    /// The timestamp of the last write to this address
    // timestamp_prev: T,
    /// The timestamp of the current read
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

        // FIXME[jpw]: this should not be here because op.enabled could be an
        // expression of degree > 1 and assert_bool is quadratic
        // builder.assert_bool(op.enabled.clone());

        // TODO[jpw] immediate checks should not be in memory bridge
        // Currently: expected is that enabled = 0, is_immediate = 0, all aux = 0 works

        // Ensuring is_immediate is correct
        // let addr_space_is_zero_cols = IsZeroCols::<AB::Expr>::new(
        //     IsZeroIoCols::<AB::Expr>::new(op.addr_space.clone(), aux.is_immediate.into()),
        //     aux.is_zero_aux.into(),
        // );

        // self.is_zero_air.subair_eval(
        //     &mut builder.when(op.enabled.clone()), // when not enabled, allow aux to be all 0s no matter what
        //     addr_space_is_zero_cols.io,
        //     addr_space_is_zero_cols.inv,
        // );

        self.offline_checker.assert_increasing_timestamps(
            builder,
            self.timestamp.clone(),
            &self.aux.base,
            enabled.clone(),
        );

        builder
            .when(self.aux.is_immediate)
            .assert_eq(self.data[0].clone(), self.address.pointer.clone());

        // TODO[osama]: resolve is_immediate stuff
        // builder.assert_one(implies(aux.is_immediate.into(), op.enabled.clone()));
        // TODO[jpw]: make this degree 1 after removing is_immediate
        let count = enabled * not(self.aux.is_immediate);

        for i in 0..N {
            let address = MemoryAddress::new(
                self.address.address_space.clone(),
                self.address.pointer.clone() + AB::Expr::from_canonical_usize(i),
            );
            self.offline_checker
                .memory_bus
                .read(
                    address.clone(),
                    [self.data[i].clone()],
                    self.aux.base.prev_timestamps[i],
                )
                .eval(builder, count.clone());
            self.offline_checker
                .memory_bus
                .write(
                    address,
                    [self.data[i].clone()],
                    self.timestamp.clone() + AB::Expr::from_canonical_usize(i),
                )
                .eval(builder, count.clone());
        }
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
        self.offline_checker.assert_increasing_timestamps(
            builder,
            self.timestamp.clone(),
            &self.aux.base,
            enabled.clone(),
        );

        let count = enabled.clone();

        for i in 0..N {
            let address = MemoryAddress::new(
                self.address.address_space.clone(),
                self.address.pointer.clone() + AB::Expr::from_canonical_usize(i),
            );
            self.offline_checker
                .memory_bus
                .read(
                    address.clone(),
                    [self.aux.prev_data[i]],
                    self.aux.base.prev_timestamps[i],
                )
                .eval(builder, count.clone());

            self.offline_checker
                .memory_bus
                .write(
                    address,
                    [self.data[i].clone()],
                    self.timestamp.clone() + AB::Expr::from_canonical_usize(i),
                )
                .eval(builder, count.clone());
        }
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
        let range_bus = RangeCheckBus::new(RANGE_CHECKER_BUS, 1 << decomp);
        Self {
            memory_bus,
            timestamp_lt_air: IsLessThanAir::new(range_bus, clk_max_bits, decomp),
        }
    }

    fn assert_increasing_timestamps<AB: InteractionBuilder, const N: usize>(
        &self,
        builder: &mut AB,
        timestamp: AB::Expr,
        base: &MemoryBaseAuxCols<AB::Var, N>,
        enabled: AB::Expr,
    ) {
        for (prev_timestamp, clk_lt_aux) in
            zip(base.prev_timestamps, base.clk_lt_aux.clone())
        {
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
}
