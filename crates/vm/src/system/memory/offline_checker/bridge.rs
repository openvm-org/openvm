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

    /// Prepare a logical memory read operation.
    ///
    /// **Legacy API (transitional, will be removed in Stage 2).** Used by
    /// every chip that today takes `[T; 8]` data arrays. In commit 6 this
    /// path packs `data` and `prev_data` from `[T; 8]` to `[T; 4]` for the
    /// bus; pre-commit-6 it emits N elements as-is.
    ///
    /// For new chip code that produces 4 bus-shaped field expressions
    /// directly, use [`MemoryBridge::read_4`] instead — it skips the legacy
    /// pack.
    #[must_use]
    pub fn read<'a, T, V, const N: usize>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; N],
        timestamp: impl Into<T>,
        aux: &'a MemoryReadAuxCols<V>,
    ) -> MemoryReadOperation<'a, T, V, N> {
        MemoryReadOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            timestamp: timestamp.into(),
            aux,
        }
    }

    /// Prepare a logical memory write operation.
    ///
    /// **Legacy API (transitional, will be removed in Stage 2).** See
    /// [`MemoryBridge::read`] for the rationale; the matching forward-compat
    /// method is [`MemoryBridge::write_4`].
    #[must_use]
    pub fn write<'a, T, V, const N: usize>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; N],
        timestamp: impl Into<T>,
        aux: &'a MemoryWriteAuxCols<V, N>,
    ) -> MemoryWriteOperation<'a, T, V, N> {
        MemoryWriteOperation {
            offline_checker: self.offline_checker,
            address: MemoryAddress::from(address),
            data: data.map(Into::into),
            timestamp: timestamp.into(),
            aux,
        }
    }

    /// Forward-compat: prepare a logical memory read whose chip-side AIR
    /// already produces the `BLOCK_FE_WIDTH`-shaped bus message (= `[T; 4]`
    /// after commit 6's flip). No packing on the bridge side; chips are
    /// responsible for composing the 4 expressions from their native
    /// columns — `[col[0]+256·col[1], …]` for Pattern A (u8) chips or 4
    /// u16 columns directly for Pattern B.
    ///
    /// **Soundness note**: until commit 6 flips `BLOCK_FE_WIDTH` from 8 to
    /// 4, the bus is logically 8-wide and emitting 4-wide messages would
    /// imbalance the permutation. **No callers may invoke this method
    /// before commit 6.** It is defined so commit 6 doesn't have to
    /// simultaneously add new APIs and migrate callers.
    #[must_use]
    pub fn read_4<'a, T, V>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; crate::arch::BLOCK_FE_WIDTH],
        timestamp: impl Into<T>,
        aux: &'a MemoryReadAuxCols<V>,
    ) -> MemoryReadOperation<'a, T, V, { crate::arch::BLOCK_FE_WIDTH }> {
        self.read(address, data, timestamp, aux)
    }

    /// Forward-compat counterpart of [`MemoryBridge::write`]. See
    /// [`MemoryBridge::read_4`].
    #[must_use]
    pub fn write_4<'a, T, V>(
        &self,
        address: MemoryAddress<impl Into<T>, impl Into<T>>,
        data: [impl Into<T>; crate::arch::BLOCK_FE_WIDTH],
        timestamp: impl Into<T>,
        aux: &'a MemoryWriteAuxCols<V, { crate::arch::BLOCK_FE_WIDTH }>,
    ) -> MemoryWriteOperation<'a, T, V, { crate::arch::BLOCK_FE_WIDTH }> {
        self.write(address, data, timestamp, aux)
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
    /// `N > BLOCK_FE_WIDTH` (today only the legacy u8 path with `N = 8` post-flip),
    /// we pack groups of `N / BLOCK_FE_WIDTH` consecutive input elements into a
    /// single field element using base-256: `out[i] = sum_k input[i*ratio + k] *
    /// 256^k`. With `BUS_PTR_SCALE = 1` today, `BLOCK_FE_WIDTH = MEMORY_BLOCK_BYTES
    /// = 8` and the pack ratio is 1 — `out[i] = input[i]`, a no-op pass-through.
    /// In commit 6 with `BUS_PTR_SCALE = 2`, `BLOCK_FE_WIDTH = 4` and the pack
    /// ratio is 2 — `out[i] = input[2i] + 256 * input[2i+1]`.
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

/// Pack `N` input field expressions into `BLOCK_FE_WIDTH` output field
/// expressions for the memory bus message. `N` must be a multiple of
/// `BLOCK_FE_WIDTH`; today `BLOCK_FE_WIDTH = MEMORY_BLOCK_BYTES = 8` and the
/// pack is a pass-through. In commit 6 when `BUS_PTR_SCALE` flips to 2,
/// `BLOCK_FE_WIDTH` becomes 4 and `N = 8` callers get packed pairwise.
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
