//! A chip which uses preprocessed trace to provide a lookup table for range checking
//! a variable `x` has `b` bits where `b` can be any integer in `[0, range_max_bits]`.
//! In other words, the same chip can be used to range check for different bit sizes.
//! We define `0` to have `0` bits.

use core::mem::size_of;
use std::{
    borrow::{Borrow, BorrowMut},
    sync::{atomic::AtomicU32, Arc},
};

use afs_derive::AlignedBorrow;
use ax_stark_backend::{
    config::StarkGenericConfig,
    interaction::InteractionBuilder,
    p3_uni_stark::Val,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use tracing::instrument;

mod bus;
#[cfg(test)]
pub mod tests;

pub use bus::*;

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct VariableRangeCols<T> {
    pub mult: T,
}

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct VariableRangePreprocessedCols<T> {
    pub value: T,
    pub max_bits: T,
}

pub const NUM_VARIABLE_RANGE_COLS: usize = size_of::<VariableRangeCols<u8>>();
pub const NUM_VARIABLE_RANGE_PREPROCESSED_COLS: usize =
    size_of::<VariableRangePreprocessedCols<u8>>();

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
        NUM_VARIABLE_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows: Vec<F> = [F::zero(); NUM_VARIABLE_RANGE_PREPROCESSED_COLS]
            .into_iter()
            .chain((0..=self.range_max_bits()).flat_map(|bits| {
                (0..(1 << bits)).flat_map(move |value| {
                    [F::from_canonical_u32(value), F::from_canonical_usize(bits)].into_iter()
                })
            }))
            .collect();
        Some(RowMajorMatrix::new(
            rows,
            NUM_VARIABLE_RANGE_PREPROCESSED_COLS,
        ))
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for VariableRangeCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local: &VariableRangePreprocessedCols<AB::Var> = (*prep_local).borrow();
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &VariableRangeCols<AB::Var> = (*local).borrow();
        // Omit creating separate bridge.rs file for brevity
        self.bus
            .receive(prep_local.value, prep_local.max_bits)
            .eval(builder, local.mult);
    }
}

#[derive(Debug)]
pub struct VariableRangeCheckerChip {
    pub air: VariableRangeCheckerAir,
    count: Vec<AtomicU32>,
}

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
        // index is 2^max_bits + value - 1 + 1 for the extra [0, 0] row
        // if each [value, max_bits] is valid, the sends multiset will be exactly the receives multiset
        let idx = (1 << max_bits) + (value as usize);
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

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![F::zero(); self.count.len() * NUM_VARIABLE_RANGE_COLS];
        for (n, row) in rows.chunks_mut(NUM_VARIABLE_RANGE_COLS).enumerate() {
            let cols: &mut VariableRangeCols<F> = row.borrow_mut();
            cols.mult =
                F::from_canonical_u32(self.count[n].load(std::sync::atomic::Ordering::SeqCst));
        }
        RowMajorMatrix::new(rows, NUM_VARIABLE_RANGE_COLS)
    }

    /// Range checks that `value` is `bits` bits by decomposing into `limbs` where all but
    /// last limb is `range_max_bits` bits. Assumes there are enough limbs.
    pub(crate) fn decompose<F: Field>(&self, mut value: u32, bits: usize, limbs: &mut [F]) {
        debug_assert!(
            limbs.len() <= bits.div_ceil(self.range_max_bits()),
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

impl<SC: StarkGenericConfig> Chip<SC> for VariableRangeCheckerChip
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProofInput::simple_no_pis(Arc::new(self.air), trace)
    }
}

impl ChipUsageGetter for VariableRangeCheckerChip {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.count.len()
    }

    fn trace_width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }
}
