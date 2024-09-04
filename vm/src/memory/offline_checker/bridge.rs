use std::collections::VecDeque;

use afs_primitives::{
    is_less_than::{columns::IsLessThanIoCols, IsLessThanAir},
    is_zero::IsZeroAir,
    range::bus::RangeCheckBus,
    utils::{and, implies, not},
};
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::AirBuilder;
use p3_field::AbstractField;

use super::{bus::MemoryBus, columns::MemoryOfflineCheckerAuxCols};
use crate::{
    cpu::RANGE_CHECKER_BUS,
    memory::{offline_checker::operation::MemoryOperation, MemoryAddress},
};

/// The [MemoryBridge] can be created within any AIR evaluation function to be used as the
/// interface for constraining logical memory read or write operations. The bridge will add
/// all necessary constraints and interactions.
///
/// ## Usage
/// [MemoryBridge] must be initialized with the correct number of auxiliary columns to match the
/// exact number of memory operations to be constrained.
#[derive(Clone, Debug)]
// TODO: WORD_SIZE should not be here, refactor
pub struct MemoryBridge<V, const WORD_SIZE: usize> {
    offline_checker: MemoryOfflineChecker,
    // TODO[jpw]:
    // Need separate VecDeque for writes to keep track of data_prev (since reads don't need)
    // TODO[jpw]: MemoryOfflineCheckerAuxCols needs to be refactored to deal with variable word size
    pub aux: VecDeque<MemoryOfflineCheckerAuxCols<WORD_SIZE, V>>,
    // @dev: do not let MemoryBridge own &mut builder. The mutable borrow will not allow builder to be
    // used again elsewhere while MemoryBridge is in scope.
}

impl<V, const WORD_SIZE: usize> MemoryBridge<V, WORD_SIZE> {
    /// Create a new [MemoryBridge] with the given number of auxiliary columns.
    pub fn new(
        offline_checker: MemoryOfflineChecker,
        aux: impl IntoIterator<Item = MemoryOfflineCheckerAuxCols<WORD_SIZE, V>>,
    ) -> Self {
        Self {
            offline_checker,
            aux: VecDeque::from_iter(aux),
        }
    }

    /// Prepare a logical memory read operation.
    #[must_use]
    pub fn read<T>(
        // , const WORD_SIZE: usize>(
        &mut self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; WORD_SIZE],
        timestamp: impl Into<T>,
    ) -> MemoryReadOperation<T, V, WORD_SIZE> {
        let aux = self.aux.pop_front().expect("Overflowed memory accesses");

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
    pub fn write<T>(
        &mut self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; WORD_SIZE],
        timestamp: impl Into<T>,
    ) -> MemoryWriteOperation<T, V, WORD_SIZE> {
        let aux = self.aux.pop_front().expect("Overflowed memory accesses");

        MemoryWriteOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            timestamp: timestamp.into(),
            aux,
        }
    }
}

impl<V, const WORD_SIZE: usize> Drop for MemoryBridge<V, WORD_SIZE> {
    fn drop(&mut self) {
        // panic messes up rust backtrace
        if !self.aux.is_empty() {
            println!(
                "[WARN] Underflowed memory accesses: {} remaining",
                self.aux.len()
            );
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
pub struct MemoryReadOperation<T, V, const WORD_SIZE: usize> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; WORD_SIZE],
    /// The timestamp of the last write to this address
    // timestamp_prev: T,
    /// The timestamp of the current read
    timestamp: T,
    aux: MemoryOfflineCheckerAuxCols<WORD_SIZE, V>,
}

impl<T: AbstractField, V, const WORD_SIZE: usize> MemoryReadOperation<T, V, WORD_SIZE> {
    /// Evaluate constraints and send/receive interactions.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Var = V, Expr = T>,
    {
        let op = MemoryOperation {
            addr_space: self.address.address_space,
            pointer: self.address.pointer,
            timestamp: self.timestamp,
            data: self.data,
            enabled: count.into(),
        };
        self.offline_checker
            .subair_eval(builder, op, self.aux, false);
    }
}

/// Constraints and interactions for a logical memory write of `(address, data)` at time `timestamp`.
/// This reads `(address, data_prev, timestamp_prev)` from the memory bus and writes
/// `(address, data, timestamp)` to the memory bus.
/// Includes constraints for `timestamp_prev < timestamp`.
///
/// **Note:** This can be used as a logical read operation by setting `data_prev = data`.
pub struct MemoryWriteOperation<T, V, const WORD_SIZE: usize> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; WORD_SIZE],
    /// The timestamp of the current read
    timestamp: T,
    aux: MemoryOfflineCheckerAuxCols<WORD_SIZE, V>,
}

impl<T: AbstractField, V, const WORD_SIZE: usize> MemoryWriteOperation<T, V, WORD_SIZE> {
    /// Evaluate constraints and send/receive interactions.
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Var = V, Expr = T>,
    {
        let op = MemoryOperation {
            addr_space: self.address.address_space,
            pointer: self.address.pointer,
            timestamp: self.timestamp,
            data: self.data,
            enabled: count.into(),
        };
        self.offline_checker
            .subair_eval(builder, op, self.aux, true);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryOfflineChecker {
    pub memory_bus: MemoryBus,
    pub timestamp_lt_air: IsLessThanAir,
    pub is_zero_air: IsZeroAir,
}

impl MemoryOfflineChecker {
    // TODO[jpw]: pass in range bus
    pub fn new(memory_bus: MemoryBus, clk_max_bits: usize, decomp: usize) -> Self {
        let range_bus = RangeCheckBus::new(RANGE_CHECKER_BUS, 1 << decomp);
        Self {
            memory_bus,
            timestamp_lt_air: IsLessThanAir::new(range_bus, clk_max_bits, decomp),
            is_zero_air: IsZeroAir,
        }
    }
}

impl MemoryOfflineChecker {
    pub fn subair_eval<AB: InteractionBuilder, const N: usize>(
        &self,
        builder: &mut AB,
        op: MemoryOperation<N, AB::Expr>,
        aux: MemoryOfflineCheckerAuxCols<N, AB::Var>,
        is_write: bool,
    ) {
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

        // is_immediate => read
        // if is_write {
        //     builder
        //         .when(op.enabled.clone())
        //         .assert_zero(aux.is_immediate);
        // }

        for (prev_timestamp, clk_lt, clk_lt_aux) in
            izip!(aux.prev_timestamps, aux.clk_lt, aux.clk_lt_aux)
        {
            let clk_lt_io_cols = IsLessThanIoCols::<AB::Expr>::new(
                prev_timestamp.into(),
                op.timestamp.clone(),
                clk_lt.into(),
            );

            self.timestamp_lt_air.conditional_eval(
                builder,
                clk_lt_io_cols,
                clk_lt_aux,
                op.enabled.clone(),
            );

            builder.assert_one(implies(
                and::<AB::Expr>(op.enabled.clone(), not(aux.is_immediate)),
                clk_lt,
            ));
        }

        // Ensuring that if op_type is Read, data_read is the same as data_write
        if !is_write {
            for i in 0..N {
                builder
                    .when(op.enabled.clone())
                    .assert_eq(op.data[i].clone(), aux.prev_data[i]);
            }
        }
        builder
            .when(aux.is_immediate)
            .assert_eq(op.data[0].clone(), op.pointer.clone());

        // TODO[osama]: resolve is_immediate stuff
        // builder.assert_one(implies(aux.is_immediate.into(), op.enabled.clone()));
        // TODO[jpw]: make this degree 1 after removing is_immediate
        let count = op.enabled * not(aux.is_immediate);

        for i in 0..N {
            let address = MemoryAddress::new(
                op.addr_space.clone(),
                op.pointer.clone() + AB::Expr::from_canonical_usize(i),
            );
            self.memory_bus
                .read(address.clone(), [aux.prev_data[i]], aux.prev_timestamps[i])
                .eval(builder, count.clone());
            self.memory_bus
                .write(
                    address,
                    [op.data[i].clone()],
                    op.timestamp.clone() + AB::Expr::from_canonical_usize(i),
                )
                .eval(builder, count.clone());
        }
    }
}
