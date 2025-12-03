use core::borrow::BorrowMut;
use std::sync::{LazyLock, Mutex};

use p3_field::{Field, FieldAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{F, poly_common::Squarable};

use super::air::ExpBitsLenCols;

pub(crate) const NUM_BITS_MAX_PLUS_ONE: usize = 32;

/// Lookup table for inverses of all num_bits in [0, NUM_BITS_MAX_PLUS_ONE]
/// Note: 0^(-1) is stored as 0
static NUM_BITS_INV_TABLE: LazyLock<[F; NUM_BITS_MAX_PLUS_ONE]> = LazyLock::new(|| {
    std::array::from_fn(|idx| {
        if idx == 0 {
            F::ZERO
        } else {
            let value = F::from_canonical_u32(idx as u32);
            value
                .try_inverse()
                .expect("non-zero num_bits value should always be invertible")
        }
    })
});

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ExpBitsLenRecord {
    pub num_bits: u8,
    pub base: F,
    pub bit_src: F,
    pub row_offset: u32,
}

impl ExpBitsLenRecord {
    pub(crate) fn new(base: F, bit_src: F, num_bits: usize, row_offset: u32) -> Self {
        debug_assert!(num_bits < NUM_BITS_MAX_PLUS_ONE);
        Self {
            base,
            bit_src,
            num_bits: u8::try_from(num_bits)
                .expect("num_bits fits in NUM_BITS_MAX_PLUS_ONE (< 256)"),
            row_offset,
        }
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.num_bits as usize + 1
    }

    pub(crate) fn end_row(&self) -> usize {
        self.row_offset as usize + self.num_rows()
    }
}

#[derive(Debug, Default)]
pub struct ExpBitsLenCpuTraceGenerator {
    pub requests: Mutex<Vec<ExpBitsLenRecord>>,
}

impl ExpBitsLenCpuTraceGenerator {
    pub fn add_request(&self, base: F, bit_src: F, num_bits: usize) {
        self.add_requests([(base, bit_src, num_bits)]);
    }

    pub fn add_requests<I>(&self, batch: I)
    where
        I: IntoIterator<Item = (F, F, usize)>,
    {
        let mut records = self.requests.lock().unwrap();
        let mut next_row_offset = records.last().map(|record| record.end_row()).unwrap_or(0);
        for (base, bit_src, num_bits) in batch {
            let row_offset = u32::try_from(next_row_offset).expect("row offset should fit in u32");
            let record = ExpBitsLenRecord::new(base, bit_src, num_bits, row_offset);
            next_row_offset += record.num_rows();
            records.push(record);
        }
    }

    #[tracing::instrument(name = "generate_trace", skip_all)]
    pub fn generate_trace_row_major(self) -> RowMajorMatrix<F> {
        let records = self.requests.into_inner().unwrap();
        let num_valid_rows = records.last().map(|record| record.end_row()).unwrap_or(0);
        let width = ExpBitsLenCols::<F>::width();

        let padded_rows = num_valid_rows.next_power_of_two();
        let mut trace = vec![F::ZERO; padded_rows * width];

        // Split trace into chunks for each request
        let (data_slice, padding_slice) = trace.split_at_mut(num_valid_rows * width);
        let mut trace_slices: Vec<&mut [F]> = Vec::with_capacity(records.len());
        let mut remaining = data_slice;

        for record in &records {
            let chunk_size = record.num_rows() * width;
            let (chunk, rest) = remaining.split_at_mut(chunk_size);
            trace_slices.push(chunk);
            remaining = rest;
        }

        // Fill valid rows in parallel across requests (serial within each request)
        tracing::info_span!("fill_valid_rows").in_scope(|| {
            trace_slices
                .par_iter_mut()
                .zip(records.par_iter())
                .for_each(|(trace_slice, request)| {
                    fill_valid_rows(
                        request.base,
                        request.bit_src.as_canonical_u32(),
                        request.num_bits,
                        trace_slice,
                        width,
                    );
                });
        });

        // Fill padding rows: 0^0 = 1
        tracing::info_span!("fill_padding_rows").in_scope(|| {
            padding_slice.par_chunks_exact_mut(width).for_each(|row| {
                let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
                cols.result = F::ONE;
            });
        });

        RowMajorMatrix::new(trace, width)
    }
}

pub(crate) fn fill_valid_rows(base: F, bit_src: u32, n: u8, trace_slice: &mut [F], width: usize) {
    let bases: Vec<_> = base.exp_powers_of_2().take(n as usize + 1).collect();
    let mut acc = F::ONE;

    for i in (0..=n).rev() {
        let remaining = n - i;
        let shifted = bit_src >> i;

        let (result, sub_result) = if i == n {
            (F::ONE, F::ONE)
        } else {
            let sub = acc;
            if (bit_src >> i) & 1 == 1 {
                acc *= bases[i as usize];
            }
            (acc, sub)
        };

        let row_offset = remaining as usize * width;
        let row = &mut trace_slice[row_offset..row_offset + width];
        let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
        cols.base = bases[i as usize];
        cols.bit_src = F::from_canonical_u32(shifted);
        cols.result = result;
        cols.sub_result = sub_result;
        cols.num_bits = F::from_canonical_u8(remaining);
        cols.is_valid = F::ONE;
        cols.bit_src_div_2 = F::from_canonical_u32(shifted / 2);
        cols.bit_src_mod_2 = F::from_canonical_u32(shifted % 2);
        cols.num_bits_inv = NUM_BITS_INV_TABLE[remaining as usize];
    }
}
