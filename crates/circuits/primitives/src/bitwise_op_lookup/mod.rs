use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
    sync::{atomic::AtomicU32, Arc},
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir, PairBuilder},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::{get_air_name, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};

mod bus;
pub use bus::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct BitwiseOperationLookupCols<T> {
    /// Number of range check operations requested for each (x, y) pair
    pub mult_range: T,
    /// Number of XOR operations requested for each (x, y) pair
    pub mult_xor: T,
}

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct BitwiseOperationLookupPreprocessedCols<T> {
    pub x: T,
    pub y: T,
    /// XOR result of x and y (x ⊕ y)
    pub z_xor: T,
}

pub const NUM_BITWISE_OP_LOOKUP_COLS: usize = size_of::<BitwiseOperationLookupCols<u8>>();
pub const NUM_BITWISE_OP_LOOKUP_PREPROCESSED_COLS: usize =
    size_of::<BitwiseOperationLookupPreprocessedCols<u8>>();

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct BitwiseOperationLookupAir<const NUM_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
}

impl<F: Field, const NUM_BITS: usize> BaseAirWithPublicValues<F>
    for BitwiseOperationLookupAir<NUM_BITS>
{
}
impl<F: Field, const NUM_BITS: usize> PartitionedBaseAir<F>
    for BitwiseOperationLookupAir<NUM_BITS>
{
}
impl<F: Field, const NUM_BITS: usize> BaseAir<F> for BitwiseOperationLookupAir<NUM_BITS> {
    fn width(&self) -> usize {
        NUM_BITWISE_OP_LOOKUP_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows: Vec<F> = (0..(1 << NUM_BITS))
            .flat_map(|x: u32| {
                (0..(1 << NUM_BITS)).flat_map(move |y: u32| {
                    [
                        F::from_canonical_u32(x),
                        F::from_canonical_u32(y),
                        F::from_canonical_u32(x ^ y),
                    ]
                })
            })
            .collect();
        Some(RowMajorMatrix::new(
            rows,
            NUM_BITWISE_OP_LOOKUP_PREPROCESSED_COLS,
        ))
    }
}

impl<AB: InteractionBuilder + PairBuilder, const NUM_BITS: usize> Air<AB>
    for BitwiseOperationLookupAir<NUM_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local: &BitwiseOperationLookupPreprocessedCols<AB::Var> = (*prep_local).borrow();

        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BitwiseOperationLookupCols<AB::Var> = (*local).borrow();

        self.bus
            .receive(prep_local.x, prep_local.y, AB::F::ZERO, AB::F::ZERO)
            .eval(builder, local.mult_range);
        self.bus
            .receive(prep_local.x, prep_local.y, prep_local.z_xor, AB::F::ONE)
            .eval(builder, local.mult_xor);
    }
}

// Lookup chip for operations on size NUM_BITS integers. Currently has pre-processed columns
// for x ^ y and range check. Interactions are of form [x, y, z] where z is either x ^ y for
// XOR or 0 for range check.

pub struct BitwiseOperationLookupChip<const NUM_BITS: usize> {
    pub air: BitwiseOperationLookupAir<NUM_BITS>,
    pub count_range: Vec<AtomicU32>,
    pub count_xor: Vec<AtomicU32>,
}

pub type SharedBitwiseOperationLookupChip<const NUM_BITS: usize> =
    Arc<BitwiseOperationLookupChip<NUM_BITS>>;

impl<const NUM_BITS: usize> BitwiseOperationLookupChip<NUM_BITS> {
    pub fn new(bus: BitwiseOperationLookupBus) -> Self {
        let num_rows = (1 << NUM_BITS) * (1 << NUM_BITS);
        let count_range = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        let count_xor = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        Self {
            air: BitwiseOperationLookupAir::new(bus),
            count_range,
            count_xor,
        }
    }

    pub fn bus(&self) -> BitwiseOperationLookupBus {
        self.air.bus
    }

    pub fn air_width(&self) -> usize {
        NUM_BITWISE_OP_LOOKUP_COLS
    }

    pub fn request_range(&self, x: u32, y: u32) {
        let upper_bound = 1 << NUM_BITS;
        debug_assert!(x < upper_bound, "x out of range: {} >= {}", x, upper_bound);
        debug_assert!(y < upper_bound, "y out of range: {} >= {}", y, upper_bound);
        self.count_range[Self::idx(x, y)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn request_xor(&self, x: u32, y: u32) -> u32 {
        let upper_bound = 1 << NUM_BITS;
        debug_assert!(x < upper_bound, "x out of range: {} >= {}", x, upper_bound);
        debug_assert!(y < upper_bound, "y out of range: {} >= {}", y, upper_bound);
        self.count_xor[Self::idx(x, y)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        x ^ y
    }

    pub fn clear(&self) {
        for i in 0..self.count_range.len() {
            self.count_range[i].store(0, std::sync::atomic::Ordering::Relaxed);
            self.count_xor[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Generates trace and resets all internal counters to 0.
    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let mut rows = F::zero_vec(self.count_range.len() * NUM_BITWISE_OP_LOOKUP_COLS);
        for (n, row) in rows.chunks_mut(NUM_BITWISE_OP_LOOKUP_COLS).enumerate() {
            let cols: &mut BitwiseOperationLookupCols<F> = row.borrow_mut();
            cols.mult_range = F::from_canonical_u32(
                self.count_range[n].swap(0, std::sync::atomic::Ordering::SeqCst),
            );
            cols.mult_xor = F::from_canonical_u32(
                self.count_xor[n].swap(0, std::sync::atomic::Ordering::SeqCst),
            );
        }
        RowMajorMatrix::new(rows, NUM_BITWISE_OP_LOOKUP_COLS)
    }

    fn idx(x: u32, y: u32) -> usize {
        (x * (1 << NUM_BITS) + y) as usize
    }
}

impl<R, SC: StarkGenericConfig, const NUM_BITS: usize> Chip<R, CpuBackend<SC>>
    for BitwiseOperationLookupChip<NUM_BITS>
{
    /// Generates trace and resets all internal counters to 0.
    fn generate_proving_ctx(&self, _: R) -> AirProvingContext<CpuBackend<SC>> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProvingContext::simple_no_pis(Arc::new(trace))
    }
}

impl<const NUM_BITS: usize> ChipUsageGetter for BitwiseOperationLookupChip<NUM_BITS> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn constant_trace_height(&self) -> Option<usize> {
        Some(1 << (2 * NUM_BITS))
    }
    fn current_trace_height(&self) -> usize {
        1 << (2 * NUM_BITS)
    }
    fn trace_width(&self) -> usize {
        NUM_BITWISE_OP_LOOKUP_COLS
    }
}
