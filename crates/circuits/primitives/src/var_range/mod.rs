//! A chip which provides a lookup table for range checking a variable `x` has `b` bits
//! where `b` can be any integer in `[0, range_max_bits]` without using preprocessed trace.
//! In other words, the same chip can be used to range check for different bit sizes.
//! We define `0` to have `0` bits.

use core::mem::size_of;
use std::{
    borrow::{Borrow, BorrowMut},
    sync::{atomic::AtomicU32, Arc},
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::{get_air_name, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use tracing::instrument;

use crate::utils::select;

mod bus;
pub use bus::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
pub mod tests;

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct VariableRangeCols<T> {
    /// The value being range checked
    pub value: T,
    /// The maximum number of bits for this value
    pub max_bits: T,
    /// Helper column used to handle wrap-around transitions, stores 2^max_bits
    pub two_to_max_bits: T,
    /// The inverse of the selector (value + 1 - two_to_max_bits), unconstrained if selector is 0.
    /// Used to create a boolean selector for detecting wrap transitions
    pub selector_inverse: T,
    /// Boolean selector: 1 if NOT a wrap transition, 0 if wrapping
    /// Used to reduce degree of transition constraints
    pub is_not_wrap: T,
    /// Number of range checks requested for each (value, max_bits) pair
    pub mult: T,
}

pub const NUM_VARIABLE_RANGE_COLS: usize = size_of::<VariableRangeCols<u8>>();

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct VariableRangeCheckerAir {
    pub bus: VariableRangeCheckerBus,
}

impl VariableRangeCheckerAir {
    pub fn range_max_bits(&self) -> usize {
        self.bus.range_max_bits
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for VariableRangeCheckerAir {}
impl<F: Field> PartitionedBaseAir<F> for VariableRangeCheckerAir {}
impl<F: Field> BaseAir<F> for VariableRangeCheckerAir {
    fn width(&self) -> usize {
        VariableRangeCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for VariableRangeCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &VariableRangeCols<AB::Var> = (*local).borrow();
        let next: &VariableRangeCols<AB::Var> = (*next).borrow();

        // First-row constraints: ensure we start at [0, 0] with two_to_max_bits = 1.
        // Note: selector_inverse is unconstrained on the first row because selector = 0.
        builder.when_first_row().assert_zero(local.value);
        builder.when_first_row().assert_zero(local.max_bits);
        builder.when_first_row().assert_one(local.two_to_max_bits);

        // The selector is (value + 1 - two_to_max_bits), which is 0 when it is a wrap transition.
        let selector = local.value + AB::Expr::ONE - local.two_to_max_bits;
        // Constraints to ensure that is_not_wrap is 1 when selector is non-zero, and 0 when
        // selector is 0.
        builder.when(selector.clone()).assert_one(local.is_not_wrap);
        builder.assert_eq(local.is_not_wrap, selector.clone() * local.selector_inverse);

        // If not a wrap transition, value should increment by 1, otherwise, value should reset to 0
        builder.when_transition().assert_eq(
            next.value,
            select::<AB::Expr>(
                local.is_not_wrap,
                local.value + AB::Expr::ONE,
                AB::Expr::ZERO,
            ),
        );
        // If not a wrap transition, max_bits should stay the same, otherwise, max_bits should
        // increment by 1
        builder.when_transition().assert_eq(
            next.max_bits,
            select::<AB::Expr>(
                local.is_not_wrap,
                local.max_bits,
                local.max_bits + AB::Expr::ONE,
            ),
        );
        // If not wrap transition, two_to_max_bits should stay the same, otherwise, two_to_max_bits
        // should be multiplied by 2
        builder.when_transition().assert_eq(
            next.two_to_max_bits,
            select::<AB::Expr>(
                local.is_not_wrap,
                local.two_to_max_bits,
                local.two_to_max_bits * AB::Expr::TWO,
            ),
        );

        // Ensure the last row is our dummy row: value=0, max_bits=range_max_bits+1, mult=0
        // This dummy row makes the trace height a power of 2.
        builder.when_last_row().assert_zero(local.value);
        builder.when_last_row().assert_eq(
            local.max_bits,
            AB::F::from_canonical_usize(self.bus.range_max_bits + 1),
        );
        builder.when_last_row().assert_zero(local.mult);

        self.bus
            .receive(local.value, local.max_bits)
            .eval(builder, local.mult);
    }
}

pub struct VariableRangeCheckerChip {
    pub air: VariableRangeCheckerAir,
    pub count: Vec<AtomicU32>,
}

pub type SharedVariableRangeCheckerChip = Arc<VariableRangeCheckerChip>;

impl VariableRangeCheckerChip {
    pub fn new(bus: VariableRangeCheckerBus) -> Self {
        let num_rows = (1 << (bus.range_max_bits + 1)) as usize;
        let count = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        Self {
            air: VariableRangeCheckerAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> VariableRangeCheckerBus {
        self.air.bus
    }

    pub fn range_max_bits(&self) -> usize {
        self.air.range_max_bits()
    }

    pub fn air_width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }

    #[instrument(
        name = "VariableRangeCheckerChip::add_count",
        skip(self),
        level = "trace"
    )]
    pub fn add_count(&self, value: u32, max_bits: usize) {
        // index is 2^max_bits + value - 1
        // if each [value, max_bits] is valid, the sends multiset will be exactly the receives
        // multiset
        let idx = (1 << max_bits) + (value as usize) - 1;
        assert!(
            idx < self.count.len(),
            "range exceeded: {} >= {}",
            idx,
            self.count.len()
        );
        let val_atomic = &self.count[idx];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for i in 0..self.count.len() {
            self.count[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Generates trace and resets the internal counters all to 0.
    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let mut rows = F::zero_vec(self.count.len() * NUM_VARIABLE_RANGE_COLS);
        for (i, row) in rows.chunks_exact_mut(NUM_VARIABLE_RANGE_COLS).enumerate() {
            let cols: &mut VariableRangeCols<F> = (*row).borrow_mut();
            let max_bits = (i + 1).ilog2();
            let two_to_max_bits = 1 << max_bits;
            let value = i + 1 - two_to_max_bits;

            // Convert to field elements before computing selector
            cols.value = F::from_canonical_usize(value);
            cols.max_bits = F::from_canonical_u32(max_bits);
            cols.two_to_max_bits = F::from_canonical_usize(two_to_max_bits);
            let selector = cols.value + F::ONE - cols.two_to_max_bits;
            cols.selector_inverse = selector.try_inverse().unwrap_or(F::ZERO);
            cols.is_not_wrap = F::from_bool(!selector.is_zero());
            cols.mult =
                F::from_canonical_u32(self.count[i].swap(0, std::sync::atomic::Ordering::Relaxed));
        }
        RowMajorMatrix::new(rows, NUM_VARIABLE_RANGE_COLS)
    }

    /// Range checks that `value` is `bits` bits by decomposing into `limbs` where all but
    /// last limb is `range_max_bits` bits. Assumes there are enough limbs.
    pub fn decompose<F: Field>(&self, mut value: u32, bits: usize, limbs: &mut [F]) {
        debug_assert!(
            limbs.len() >= bits.div_ceil(self.range_max_bits()),
            "Not enough limbs: len {}",
            limbs.len()
        );
        let mask = (1 << self.range_max_bits()) - 1;
        let mut bits_remaining = bits;
        for limb in limbs.iter_mut() {
            let limb_u32 = value & mask;
            *limb = F::from_canonical_u32(limb_u32);
            self.add_count(limb_u32, bits_remaining.min(self.range_max_bits()));

            value >>= self.range_max_bits();
            bits_remaining = bits_remaining.saturating_sub(self.range_max_bits());
        }
        debug_assert_eq!(value, 0);
        debug_assert_eq!(bits_remaining, 0);
    }
}

// We allow any `R` type so this can work with arbitrary record arenas.
impl<R, SC: StarkGenericConfig> Chip<R, CpuBackend<SC>> for VariableRangeCheckerChip
where
    Val<SC>: PrimeField32,
{
    /// Generates trace and resets the internal counters all to 0.
    fn generate_proving_ctx(&self, _: R) -> AirProvingContext<CpuBackend<SC>> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProvingContext::simple_no_pis(Arc::new(trace))
    }
}

impl ChipUsageGetter for VariableRangeCheckerChip {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn constant_trace_height(&self) -> Option<usize> {
        Some(self.count.len())
    }
    fn current_trace_height(&self) -> usize {
        self.count.len()
    }
    fn trace_width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }
}
