use getset::CopyGetters;
use openvm_circuit_primitives::{
    assert_less_than::{AssertLessThanIo, AssertLtSubAir},
    var_range::VariableRangeCheckerBus,
    SubAir,
};
use openvm_stark_backend::{interaction::InteractionBuilder, p3_field::PrimeCharacteristicRing};

use super::bus::MemoryBus;
use crate::{
    arch::{BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    system::memory::{
        offline_checker::columns::{MemoryBaseAuxCols, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};

/// AUX_LEN is the number of auxiliary columns (aka the number of limbs that the input numbers will
/// be decomposed into) for the `AssertLtSubAir` in the `MemoryOfflineChecker`.
/// Warning: This requires that (timestamp_max_bits + decomp - 1) / decomp = AUX_LEN
///         in MemoryOfflineChecker (or whenever AssertLtSubAir is used)
pub const AUX_LEN: usize = 2;

const _: () = assert!(
    MEMORY_BLOCK_BYTES == 2 * BLOCK_FE_WIDTH,
    "byte block packing assumes 2 bytes per bus cell"
);

/// The [MemoryBridge] is used within AIR evaluation functions to constrain logical memory
/// operations (read/write). It adds all necessary constraints and interactions.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBridge {
    offline_checker: MemoryOfflineChecker,
}

impl MemoryBridge {
    /// Create a new [MemoryBridge] with the provided offline_checker.
    pub fn new(
        memory_bus: MemoryBus,
        timestamp_max_bits: usize,
        range_bus: VariableRangeCheckerBus,
    ) -> Self {
        Self {
            offline_checker: MemoryOfflineChecker::new(memory_bus, timestamp_max_bits, range_bus),
        }
    }

    pub fn memory_bus(&self) -> MemoryBus {
        self.offline_checker.memory_bus
    }

    pub fn range_bus(&self) -> VariableRangeCheckerBus {
        self.offline_checker.timestamp_lt_air.bus
    }

    /// Prepare a logical memory read.
    #[must_use]
    pub fn read<'a, T, V>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; BLOCK_FE_WIDTH],
        timestamp: impl Into<T>,
        aux: &'a MemoryReadAuxCols<V>,
    ) -> MemoryReadOperation<'a, T, V> {
        MemoryReadOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            timestamp: timestamp.into(),
            aux,
        }
    }

    /// Prepare a logical memory write.
    #[must_use]
    pub fn write<'a, T, V, A>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; BLOCK_FE_WIDTH],
        timestamp: impl Into<T>,
        aux: A,
    ) -> MemoryWriteOperation<'a, T, V>
    where
        A: Into<MemoryWriteAuxInput<'a, T, V>>,
    {
        let aux = aux.into();
        MemoryWriteOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            prev_data: aux.prev_data,
            timestamp: timestamp.into(),
            aux_base: aux.base,
        }
    }
}

/// Auxiliary write input for [`MemoryBridge::write`].
///
/// Most chips pass `&MemoryWriteAuxCols`, where `prev_data` is read directly
/// from aux columns. Chips that already compute the previous bus payload as
/// expressions can use [`Self::from_prev_data_exprs`] with just the base
/// timestamp metadata.
pub struct MemoryWriteAuxInput<'a, T, V> {
    base: &'a MemoryBaseAuxCols<V>,
    prev_data: [T; BLOCK_FE_WIDTH],
}

impl<'a, T, V> MemoryWriteAuxInput<'a, T, V> {
    /// Use when the chip provides `prev_data` expressions instead of storing
    /// them in `MemoryWriteAuxCols`.
    pub fn from_prev_data_exprs<P>(
        base: &'a MemoryBaseAuxCols<V>,
        prev_data: [P; BLOCK_FE_WIDTH],
    ) -> Self
    where
        P: Into<T>,
    {
        Self {
            base,
            prev_data: prev_data.map(Into::into),
        }
    }
}

impl<'a, T, V> From<&'a MemoryWriteAuxCols<V, BLOCK_FE_WIDTH>> for MemoryWriteAuxInput<'a, T, V>
where
    V: Copy + Into<T>,
{
    fn from(aux: &'a MemoryWriteAuxCols<V, BLOCK_FE_WIDTH>) -> Self {
        Self {
            base: &aux.base,
            prev_data: aux.prev_data.map(Into::into),
        }
    }
}

/// Constraints and interactions for a logical memory read of `(address, data)` at time `timestamp`.
/// This reads `(address, data, timestamp_prev)` from the memory bus and writes
/// `(address, data, timestamp)` to the memory bus.
/// Includes constraints for `timestamp_prev < timestamp`.
///
/// The generic `T` type is intended to be `AB::Expr` where `AB` is the `AirBuilder`.
/// The auxiliary columns are not expected to be expressions, so the generic `V` type is intended
/// to be `AB::Var`.
pub struct MemoryReadOperation<'a, T, V> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; BLOCK_FE_WIDTH],
    timestamp: T,
    aux: &'a MemoryReadAuxCols<V>,
}

/// The max degree of constraints is:
/// eval_timestamps: deg(enabled) + max(1, deg(self.timestamp))
/// eval_bulk_access: refer to private function MemoryOfflineChecker::eval_bulk_access
impl<F: PrimeCharacteristicRing, V: Copy + Into<F>> MemoryReadOperation<'_, F, V> {
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
            self.address,
            &self.data,
            &self.data,
            self.timestamp.clone(),
            self.aux.base.prev_timestamp,
            enabled,
        );
    }
}

/// Constraints and interactions for a logical memory write of `(address, data)` at time
/// `timestamp`. This reads `(address, data_prev, timestamp_prev)` from the memory bus and writes
/// `(address, data, timestamp)` to the memory bus.
/// Includes constraints for `timestamp_prev < timestamp`.
///
/// **Note:** This can be used as a logical read operation by setting `data_prev = data`.
pub struct MemoryWriteOperation<'a, T, V> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; BLOCK_FE_WIDTH],
    prev_data: [T; BLOCK_FE_WIDTH],
    timestamp: T,
    aux_base: &'a MemoryBaseAuxCols<V>,
}

/// The max degree of constraints is:
/// eval_timestamps: deg(enabled) + max(1, deg(self.timestamp))
/// eval_bulk_access: refer to private function MemoryOfflineChecker::eval_bulk_access
impl<T: PrimeCharacteristicRing, V: Copy + Into<T>> MemoryWriteOperation<'_, T, V> {
    /// Evaluate constraints and send/receive interactions. `enabled` must be boolean.
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Var = V, Expr = T>,
    {
        let enabled = enabled.into();
        self.offline_checker.eval_timestamps(
            builder,
            self.timestamp.clone(),
            self.aux_base,
            enabled.clone(),
        );

        self.offline_checker.eval_bulk_access(
            builder,
            self.address,
            &self.data,
            &self.prev_data,
            self.timestamp,
            self.aux_base.prev_timestamp,
            enabled,
        );
    }
}

#[derive(Clone, Copy, Debug, CopyGetters)]
struct MemoryOfflineChecker {
    #[get_copy = "pub"]
    memory_bus: MemoryBus,
    #[get_copy = "pub"]
    timestamp_lt_air: AssertLtSubAir,
}

impl MemoryOfflineChecker {
    fn new(
        memory_bus: MemoryBus,
        timestamp_max_bits: usize,
        range_bus: VariableRangeCheckerBus,
    ) -> Self {
        Self {
            memory_bus,
            timestamp_lt_air: AssertLtSubAir::new(range_bus, timestamp_max_bits),
        }
    }

    /// The max degree of constraints is:
    /// deg(enabled) + max(1, deg(timestamp))
    /// Note: deg(prev_timestamp) = 1 since prev_timestamp is Var
    fn eval_timestamps<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        timestamp: AB::Expr,
        base: &MemoryBaseAuxCols<AB::Var>,
        enabled: AB::Expr,
    ) {
        let lt_io = AssertLessThanIo::new(base.prev_timestamp, timestamp.clone(), enabled);
        self.timestamp_lt_air
            .eval(builder, (lt_io, &base.timestamp_lt_aux.lower_decomp));
    }

    /// At the core, eval_bulk_access is a bunch of push_sends and push_receives.
    /// The max constraint degree of expressions in sends/receives is:
    /// max(max_deg(data), max_deg(prev_data), max_deg(timestamp), max_deg(prev_timestamps))
    /// Also, each one of them has count with degree: deg(enabled)
    #[allow(clippy::too_many_arguments)]
    fn eval_bulk_access<AB>(
        &self,
        builder: &mut AB,
        address: MemoryAddress<AB::Expr, AB::Expr>,
        data: &[AB::Expr; BLOCK_FE_WIDTH],
        prev_data: &[AB::Expr; BLOCK_FE_WIDTH],
        timestamp: AB::Expr,
        prev_timestamp: AB::Var,
        enabled: AB::Expr,
    ) where
        AB: InteractionBuilder,
    {
        self.memory_bus
            .receive(address.clone(), prev_data.to_vec(), prev_timestamp)
            .eval(builder, enabled.clone());

        self.memory_bus
            .send(address, data.to_vec(), timestamp)
            .eval(builder, enabled);
    }
}

/// Pack `MEMORY_BLOCK_BYTES` byte expressions into `BLOCK_FE_WIDTH` bus cells.
pub fn pack_u8_block<AB: InteractionBuilder>(
    data: &[AB::Expr; MEMORY_BLOCK_BYTES],
) -> [AB::Expr; BLOCK_FE_WIDTH] {
    let mut out: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|_| AB::Expr::ZERO);
    for i in 0..BLOCK_FE_WIDTH {
        out[i] = data[i * 2].clone() + AB::Expr::from_u64(256) * data[i * 2 + 1].clone();
    }
    out
}

/// Concrete-value form of [`pack_u8_block`].
pub fn pack_u8_block_value<F: PrimeCharacteristicRing + Copy>(
    data: &[F; MEMORY_BLOCK_BYTES],
) -> [F; BLOCK_FE_WIDTH] {
    let mut out = [F::ZERO; BLOCK_FE_WIDTH];
    for i in 0..BLOCK_FE_WIDTH {
        out[i] = data[i * 2] + F::from_u64(256) * data[i * 2 + 1];
    }
    out
}

/// Concrete-value form of [`pack_u8_block`] for raw bytes.
pub fn pack_u8_block_bytes<F: PrimeCharacteristicRing>(
    data: &[u8; MEMORY_BLOCK_BYTES],
) -> [F; BLOCK_FE_WIDTH] {
    std::array::from_fn(|i| {
        let lo = data[i * 2] as u64;
        let hi = data[i * 2 + 1] as u64;
        F::from_u64(lo + 256 * hi)
    })
}
