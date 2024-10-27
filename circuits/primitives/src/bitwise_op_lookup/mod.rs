use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
    sync::atomic::AtomicU32,
};

use afs_derive::AlignedBorrow;
use ax_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

mod bus;
#[cfg(test)]
mod tests;

pub use bus::*;

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct BitwiseOperationLookupCols<T> {
    pub mult_add: T,
    pub mult_xor: T,
}

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct BitwiseOperationLookupPreprocessedCols<T> {
    pub x: T,
    pub y: T,
    pub z_add: T,
    pub z_xor: T,
}

pub const NUM_BITWISE_OP_LOOKUP_COLS: usize = size_of::<BitwiseOperationLookupCols<u8>>();
pub const NUM_BITWISE_OP_LOOKUP_PREPROCESSED_COLS: usize =
    size_of::<BitwiseOperationLookupPreprocessedCols<u8>>();

#[derive(Copy, Clone, PartialEq)]
pub enum BitwiseOperationLookupOpcode {
    ADD = 0,
    XOR = 1,
}

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
                        F::from_canonical_u32((x + y) % (1 << NUM_BITS)),
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
            .receive(
                prep_local.x,
                prep_local.y,
                prep_local.z_add,
                AB::Expr::from_canonical_u8(BitwiseOperationLookupOpcode::ADD as u8),
            )
            .eval(builder, local.mult_add);
        self.bus
            .receive(
                prep_local.x,
                prep_local.y,
                prep_local.z_xor,
                AB::Expr::from_canonical_u8(BitwiseOperationLookupOpcode::XOR as u8),
            )
            .eval(builder, local.mult_xor);
    }
}

// Lookup chip for operations on size NUM_BITS integers. Currently has pre-processed columns
// for (x + y) % 2^NUM_BITS and x ^ y. Interactions are of form [x, y, z, op], where x and y
// are integers, op is an opcode (see BitwiseOperationLookupOpcode in air.rs), and z is x op y.

#[derive(Debug)]
pub struct BitwiseOperationLookupChip<const NUM_BITS: usize> {
    pub air: BitwiseOperationLookupAir<NUM_BITS>,
    count_add: Vec<AtomicU32>,
    count_xor: Vec<AtomicU32>,
}

impl<const NUM_BITS: usize> BitwiseOperationLookupChip<NUM_BITS> {
    pub fn new(bus: BitwiseOperationLookupBus) -> Self {
        let num_rows = (1 << NUM_BITS) * (1 << NUM_BITS);
        let count_add = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        let count_xor = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        Self {
            air: BitwiseOperationLookupAir::new(bus),
            count_add,
            count_xor,
        }
    }

    pub fn bus(&self) -> BitwiseOperationLookupBus {
        self.air.bus
    }

    pub fn air_width(&self) -> usize {
        NUM_BITWISE_OP_LOOKUP_COLS
    }

    pub fn add_count(&self, x: u32, y: u32, op: BitwiseOperationLookupOpcode) {
        let idx = (x as usize) * (1 << NUM_BITS) + (y as usize);
        assert!(
            idx < self.count_add.len(),
            "range exceeded: {} >= {}",
            idx,
            self.count_add.len()
        );
        let val_atomic = match op {
            BitwiseOperationLookupOpcode::ADD => &self.count_add[idx],
            BitwiseOperationLookupOpcode::XOR => &self.count_xor[idx],
        };
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for i in 0..self.count_add.len() {
            self.count_add[i].store(0, std::sync::atomic::Ordering::Relaxed);
            self.count_xor[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![F::zero(); self.count_add.len() * NUM_BITWISE_OP_LOOKUP_COLS];
        for (n, row) in rows.chunks_mut(NUM_BITWISE_OP_LOOKUP_COLS).enumerate() {
            let cols: &mut BitwiseOperationLookupCols<F> = row.borrow_mut();
            cols.mult_add =
                F::from_canonical_u32(self.count_add[n].load(std::sync::atomic::Ordering::SeqCst));
            cols.mult_xor =
                F::from_canonical_u32(self.count_xor[n].load(std::sync::atomic::Ordering::SeqCst));
        }
        RowMajorMatrix::new(rows, NUM_BITWISE_OP_LOOKUP_COLS)
    }
}
