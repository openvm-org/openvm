use std::{
    borrow::{Borrow, BorrowMut},
    sync::{atomic::AtomicU32, Arc},
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
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

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct BitwiseOperationLookupCols<T, const NUM_BITS: usize> {
    /// Binary decomposition of x (x_bits[0] is LSB, x_bits[NUM_BITS-1] is MSB)
    pub x_bits: [T; NUM_BITS],
    /// Binary decomposition of y (y_bits[0] is LSB, y_bits[NUM_BITS-1] is MSB)
    pub y_bits: [T; NUM_BITS],
    /// Number of range check operations requested for each (x, y) pair
    pub mult_range: T,
    /// Number of XOR operations requested for each (x, y) pair
    pub mult_xor: T,
}

/// Number of multiplicity columns (mult_range and mult_xor)
pub const NUM_BITWISE_OP_LOOKUP_MULT_COLS: usize = 2;

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
        BitwiseOperationLookupCols::<F, NUM_BITS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_BITS: usize> Air<AB>
    for BitwiseOperationLookupAir<NUM_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &BitwiseOperationLookupCols<AB::Var, NUM_BITS> = (*local).borrow();
        let next: &BitwiseOperationLookupCols<AB::Var, NUM_BITS> = (*next).borrow();

        // 1. Binary constraints: ensure each bit is boolean
        for i in 0..NUM_BITS {
            builder.assert_bool(local.x_bits[i]);
            builder.assert_bool(local.y_bits[i]);
        }

        // 2. Reconstruct x and y from their binary decompositions
        // x = Σ(x_bits[i] * 2^i), y = Σ(y_bits[i] * 2^i)
        let reconstruct = |bits: &[AB::Var; NUM_BITS]| {
            bits.iter().enumerate().fold(AB::Expr::ZERO, |acc, (i, &bit)| {
                acc + bit * AB::Expr::from_canonical_usize(1 << i)
            })
        };
        let x_reconstructed = reconstruct(&local.x_bits);
        let y_reconstructed = reconstruct(&local.y_bits);

        // 3. Compute z_xor algebraically from bits
        // z_xor_bits[i] = x_bits[i] ^ y_bits[i] = x_bits[i] + y_bits[i] - 2 * x_bits[i] * y_bits[i]
        // z_xor = Σ(z_xor_bits[i] * 2^i)
        let z_xor_reconstructed = local
            .x_bits
            .iter()
            .zip(local.y_bits.iter())
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, (&x_bit, &y_bit))| {
                let xor_bit = x_bit + y_bit - AB::Expr::TWO * x_bit * y_bit;
                acc + xor_bit * AB::Expr::from_canonical_usize(1 << i)
            });

        // 4. Combined index: idx = x * (2^NUM_BITS) + y
        let combined_idx = x_reconstructed.clone()
            * AB::Expr::from_canonical_usize(1 << NUM_BITS)
            + y_reconstructed.clone();
        let next_combined_idx = reconstruct(&next.x_bits) * AB::Expr::from_canonical_usize(1 << NUM_BITS)
            + reconstruct(&next.y_bits);

        // 5. Constrain that combined index increments by 1 each row
        builder
            .when_transition()
            .assert_one(next_combined_idx.clone() - combined_idx.clone());

        // 6. Boundary constraints: first row has idx = 0, last row has idx = 2^(2*NUM_BITS) - 1
        builder.when_first_row().assert_zero(combined_idx.clone());
        builder.when_last_row().assert_eq(
            combined_idx,
            AB::Expr::from_canonical_usize((1 << (2 * NUM_BITS)) - 1),
        );

        // 7. Use reconstructed values for lookup bus interactions
        self.bus
            .receive(
                x_reconstructed.clone(),
                y_reconstructed.clone(),
                AB::F::ZERO,
                AB::F::ZERO,
            )
            .eval(builder, local.mult_range);
        self.bus
            .receive(
                x_reconstructed,
                y_reconstructed,
                z_xor_reconstructed,
                AB::F::ONE,
            )
            .eval(builder, local.mult_xor);
    }
}

// Lookup chip for operations on size NUM_BITS integers. Uses gate-based constraints
// with binary decomposition instead of preprocessed trace. Interactions are of form [x, y, z]
// where z is either x ^ y for XOR or 0 for range check.

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
        BitwiseOperationLookupCols::<u8, NUM_BITS>::width()
    }

    pub fn request_range(&self, x: u32, y: u32) {
        let upper_bound = 1 << NUM_BITS;
        debug_assert!(x < upper_bound, "x out of range: {x} >= {upper_bound}");
        debug_assert!(y < upper_bound, "y out of range: {y} >= {upper_bound}");
        self.count_range[Self::idx(x, y)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn request_xor(&self, x: u32, y: u32) -> u32 {
        let upper_bound = 1 << NUM_BITS;
        debug_assert!(x < upper_bound, "x out of range: {x} >= {upper_bound}");
        debug_assert!(y < upper_bound, "y out of range: {y} >= {upper_bound}");
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
        let num_cols = BitwiseOperationLookupCols::<F, NUM_BITS>::width();
        let num_rows = (1 << NUM_BITS) * (1 << NUM_BITS);
        let mut rows = F::zero_vec(num_rows * num_cols);
        
        for (n, row) in rows.chunks_mut(num_cols).enumerate() {
            let cols: &mut BitwiseOperationLookupCols<F, NUM_BITS> = row.borrow_mut();
            
            // Compute x and y from row index: row n corresponds to (x, y) where
            // x = n / (2^NUM_BITS), y = n % (2^NUM_BITS)
            let x = (n / (1 << NUM_BITS)) as u32;
            let y = (n % (1 << NUM_BITS)) as u32;
            
            // Set x_bits and y_bits: decompose x and y into binary
            for i in 0..NUM_BITS {
                cols.x_bits[i] = F::from_canonical_u32((x >> i) & 1);
                cols.y_bits[i] = F::from_canonical_u32((y >> i) & 1);
            }
            
            // Set multiplicities
            cols.mult_range = F::from_canonical_u32(
                self.count_range[n].swap(0, std::sync::atomic::Ordering::SeqCst),
            );
            cols.mult_xor = F::from_canonical_u32(
                self.count_xor[n].swap(0, std::sync::atomic::Ordering::SeqCst),
            );
        }
        RowMajorMatrix::new(rows, num_cols)
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
        BitwiseOperationLookupCols::<u8, NUM_BITS>::width()
    }
}
