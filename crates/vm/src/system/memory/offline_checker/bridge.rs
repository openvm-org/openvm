use getset::CopyGetters;
use openvm_circuit_primitives::{
    assert_less_than::{AssertLessThanIo, AssertLtSubAir},
    var_range::VariableRangeCheckerBus,
    SubAir,
};
use openvm_stark_backend::{interaction::InteractionBuilder, p3_field::PrimeCharacteristicRing};

use super::bus::MemoryBus;
use crate::system::memory::{
    offline_checker::columns::{MemoryBaseAuxCols, MemoryReadAuxCols, MemoryWriteAuxCols},
    MemoryAddress,
};

/// AUX_LEN is the number of auxiliary columns (aka the number of limbs that the input numbers will
/// be decomposed into) for the `AssertLtSubAir` in the `MemoryOfflineChecker`.
/// Warning: This requires that (timestamp_max_bits + decomp - 1) / decomp = AUX_LEN
///         in MemoryOfflineChecker (or whenever AssertLtSubAir is used)
pub const AUX_LEN: usize = 2;

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

    /// Prepare a logical memory read whose chip-side AIR produces the
    /// `BLOCK_FE_WIDTH`-shaped bus message (= `[T; 4]`). Chips are
    /// responsible for composing the 4 expressions from their native
    /// columns — `[col[0]+256·col[1], …]` for Pattern A (u8) chips or 4 u16
    /// columns directly for Pattern B.
    #[must_use]
    pub fn read_4<'a, T, V>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; crate::arch::BLOCK_FE_WIDTH],
        timestamp: impl Into<T>,
        aux: &'a MemoryReadAuxCols<V>,
    ) -> MemoryReadOperation<'a, T, V, { crate::arch::BLOCK_FE_WIDTH }> {
        MemoryReadOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            timestamp: timestamp.into(),
            aux,
        }
    }

    /// Prepare a logical memory write whose chip-side AIR produces the
    /// `BLOCK_FE_WIDTH`-shaped bus message. See [`MemoryBridge::read_4`].
    #[must_use]
    pub fn write_4<'a, T, V>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; crate::arch::BLOCK_FE_WIDTH],
        timestamp: impl Into<T>,
        aux: &'a MemoryWriteAuxCols<V, { crate::arch::BLOCK_FE_WIDTH }>,
    ) -> MemoryWriteOperation<'a, T, V, { crate::arch::BLOCK_FE_WIDTH }> {
        MemoryWriteOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            timestamp: timestamp.into(),
            aux,
        }
    }

    /// Variant of [`MemoryBridge::write_4`] where `prev_data` is supplied as
    /// `BLOCK_FE_WIDTH` field-element expressions (e.g. composed from the
    /// chip's own u8 columns) instead of being read off an
    /// `MemoryWriteAuxCols` column slot. The aux only contains the timestamp
    /// metadata (`MemoryBaseAuxCols`). Used by chips like Keccak256 / SHA2 /
    /// LoadStore that already materialize the prev_data bytes in their own
    /// columns and don't want to duplicate them in an adapter aux slot.
    #[must_use]
    pub fn write_4_with_prev<'a, T, V>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; crate::arch::BLOCK_FE_WIDTH],
        prev_data: [impl Into<T>; crate::arch::BLOCK_FE_WIDTH],
        timestamp: impl Into<T>,
        aux_base: &'a MemoryBaseAuxCols<V>,
    ) -> MemoryWriteOperationWithPrev<'a, T, V> {
        MemoryWriteOperationWithPrev {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            prev_data: prev_data.map(Into::into),
            timestamp: timestamp.into(),
            aux_base,
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
pub struct MemoryReadOperation<'a, T, V, const N: usize> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; N],
    timestamp: T,
    aux: &'a MemoryReadAuxCols<V>,
}

/// The max degree of constraints is:
/// eval_timestamps: deg(enabled) + max(1, deg(self.timestamp))
/// eval_bulk_access: refer to private function MemoryOfflineChecker::eval_bulk_access
impl<F: PrimeCharacteristicRing, V: Copy + Into<F>, const N: usize>
    MemoryReadOperation<'_, F, V, N>
{
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
pub struct MemoryWriteOperation<'a, T, V, const N: usize> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; N],
    /// The timestamp of the current read
    timestamp: T,
    aux: &'a MemoryWriteAuxCols<V, N>,
}

/// The max degree of constraints is:
/// eval_timestamps: deg(enabled) + max(1, deg(self.timestamp))
/// eval_bulk_access: refer to private function MemoryOfflineChecker::eval_bulk_access
impl<T: PrimeCharacteristicRing, V: Copy + Into<T>, const N: usize>
    MemoryWriteOperation<'_, T, V, N>
{
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
            self.address,
            &self.data,
            &self.aux.prev_data.map(Into::into),
            self.timestamp,
            self.aux.base.prev_timestamp,
            enabled,
        );
    }
}

/// Constraints and interactions for a write whose `prev_data` is supplied as
/// `BLOCK_FE_WIDTH` field-element expressions rather than being loaded from a
/// `MemoryWriteAuxCols` column slot. See
/// [`MemoryBridge::write_4_with_prev`].
pub struct MemoryWriteOperationWithPrev<'a, T, V> {
    offline_checker: MemoryOfflineChecker,
    address: MemoryAddress<T, T>,
    data: [T; crate::arch::BLOCK_FE_WIDTH],
    prev_data: [T; crate::arch::BLOCK_FE_WIDTH],
    timestamp: T,
    aux_base: &'a MemoryBaseAuxCols<V>,
}

impl<T: PrimeCharacteristicRing, V: Copy + Into<T>> MemoryWriteOperationWithPrev<'_, T, V> {
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
    ///
    /// **Bus pack**: the bus message is `BLOCK_FE_WIDTH` field elements wide. When
    /// `N > BLOCK_FE_WIDTH` (the legacy u8 path with `N = 8`), we pack groups of
    /// `N / BLOCK_FE_WIDTH` consecutive input elements into a single field
    /// element using base-256: `out[i] = sum_k input[i*ratio + k] * 256^k`. With
    /// `BUS_PTR_SCALE = 2`, `BLOCK_FE_WIDTH = 4` and the pack ratio is 2 —
    /// `out[i] = input[2i] + 256 * input[2i+1]`.
    #[allow(clippy::too_many_arguments)]
    fn eval_bulk_access<AB, const N: usize>(
        &self,
        builder: &mut AB,
        address: MemoryAddress<AB::Expr, AB::Expr>,
        data: &[AB::Expr; N],
        prev_data: &[AB::Expr; N],
        timestamp: AB::Expr,
        prev_timestamp: AB::Var,
        enabled: AB::Expr,
    ) where
        AB: InteractionBuilder,
    {
        let packed_data = pack_for_bus::<AB, N>(data);
        let packed_prev = pack_for_bus::<AB, N>(prev_data);

        self.memory_bus
            .receive(address.clone(), packed_prev, prev_timestamp)
            .eval(builder, enabled.clone());

        self.memory_bus
            .send(address, packed_data, timestamp)
            .eval(builder, enabled);
    }
}

/// Pack `MEMORY_BLOCK_BYTES` u8-typed field expressions into
/// `BLOCK_FE_WIDTH` bus expressions via base-256:
/// `out[i] = data[2i] + 256·data[2i+1]`.
///
/// Pattern A chips that keep their `[T; 8]` u8 columns but want to call the
/// new [`MemoryBridge::read_4`] / [`MemoryBridge::write_4`] API can pass
/// the result of this helper directly to the bridge.
pub fn pack_u8_for_bus<AB: InteractionBuilder>(
    data: &[AB::Expr; crate::arch::MEMORY_BLOCK_BYTES],
) -> [AB::Expr; crate::arch::BLOCK_FE_WIDTH] {
    // `data[i * 2]` / `data[i * 2 + 1]` hardcodes a 2:1 byte-to-cell ratio.
    // Surface that assumption locally so a future change to
    // `MEMORY_BLOCK_BYTES` or `BLOCK_FE_WIDTH` trips here instead of
    // silently producing wrong packings.
    const {
        assert!(
            crate::arch::MEMORY_BLOCK_BYTES / crate::arch::BLOCK_FE_WIDTH == 2,
            "pack_u8_for_bus assumes 2 bytes per bus cell"
        )
    };
    let mut out: [AB::Expr; crate::arch::BLOCK_FE_WIDTH] = std::array::from_fn(|_| AB::Expr::ZERO);
    for i in 0..crate::arch::BLOCK_FE_WIDTH {
        out[i] = data[i * 2].clone() + AB::Expr::from_u64(256) * data[i * 2 + 1].clone();
    }
    out
}

/// Pack `N` input field expressions into `BLOCK_FE_WIDTH` output field
/// expressions for the memory bus message. `N` must be a multiple of
/// `BLOCK_FE_WIDTH`. With `BUS_PTR_SCALE = 2`, `BLOCK_FE_WIDTH = 4` and
/// `N = 8` callers get packed pairwise: `out[i] = input[2i] + 256·input[2i+1]`.
fn pack_for_bus<AB, const N: usize>(data: &[AB::Expr; N]) -> Vec<AB::Expr>
where
    AB: InteractionBuilder,
{
    let pack_ratio = N / crate::arch::BLOCK_FE_WIDTH;
    assert_eq!(
        pack_ratio * crate::arch::BLOCK_FE_WIDTH,
        N,
        "bridge bus pack: N={N} must be a multiple of BLOCK_FE_WIDTH={}",
        crate::arch::BLOCK_FE_WIDTH
    );
    let mut packed = Vec::with_capacity(crate::arch::BLOCK_FE_WIDTH);
    for i in 0..crate::arch::BLOCK_FE_WIDTH {
        let mut acc = AB::Expr::ZERO;
        let mut mult: u64 = 1;
        for k in 0..pack_ratio {
            acc += AB::Expr::from_u64(mult) * data[i * pack_ratio + k].clone();
            mult *= 256;
        }
        packed.push(acc);
    }
    packed
}
