use core::borrow::BorrowMut;
use std::sync::Mutex;

use openvm_stark_backend::poly_common::Squarable;
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use super::air::ExpBitsLenCols;

pub(crate) const NUM_BITS_MAX_PLUS_ONE: usize = 32;
pub(crate) const LOW_BITS_COUNT: usize = 27;

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ExpBitsLenRecord {
    pub num_bits: u8,
    pub base: F,
    pub bit_src: F,
    pub row_offset: u32,
    pub shift_bits: u8,
    pub shift_mult: u32,
}

impl ExpBitsLenRecord {
    pub(crate) fn new(
        base: F,
        bit_src: F,
        num_bits: usize,
        row_offset: u32,
        shift_bits: usize,
        shift_mult: u32,
    ) -> Self {
        debug_assert!(num_bits < NUM_BITS_MAX_PLUS_ONE);
        Self {
            base,
            bit_src,
            num_bits: u8::try_from(num_bits)
                .expect("num_bits fits in NUM_BITS_MAX_PLUS_ONE (< 256)"),
            row_offset,
            shift_bits: u8::try_from(shift_bits)
                .expect("shift_bits fits in NUM_BITS_MAX_PLUS_ONE (< 256)"),
            shift_mult,
        }
    }

    pub(crate) fn num_rows(&self) -> usize {
        NUM_BITS_MAX_PLUS_ONE
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
        self.add_requests_with_shift(batch.into_iter().map(|(x, y, z)| (x, y, z, 0, 0)));
    }

    pub fn add_requests_with_shift<I>(&self, batch: I)
    where
        I: IntoIterator<Item = (F, F, usize, usize, u32)>,
    {
        let mut records = self.requests.lock().unwrap();
        let mut next_row_offset = records.last().map(|record| record.end_row()).unwrap_or(0);
        for (base, bit_src, num_bits, shift_bits, shift_mult) in batch {
            let row_offset = u32::try_from(next_row_offset).expect("row offset should fit in u32");
            let record =
                ExpBitsLenRecord::new(base, bit_src, num_bits, row_offset, shift_bits, shift_mult);
            next_row_offset += record.num_rows();
            records.push(record);
        }
    }

    #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
    pub fn generate_trace_row_major(
        self,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let records = self.requests.into_inner().unwrap();
        let num_valid_rows = records.last().map(|record| record.end_row()).unwrap_or(0);
        let width = ExpBitsLenCols::<F>::width();

        let padded_rows = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padded_rows * width];

        let (data_slice, padding_slice) = trace.split_at_mut(num_valid_rows * width);
        let mut trace_slices: Vec<&mut [F]> = Vec::with_capacity(records.len());
        let mut remaining = data_slice;

        for record in &records {
            let chunk_size = record.num_rows() * width;
            let (chunk, rest) = remaining.split_at_mut(chunk_size);
            trace_slices.push(chunk);
            remaining = rest;
        }

        tracing::info_span!("fill_valid_rows").in_scope(|| {
            trace_slices
                .par_iter_mut()
                .zip(records.par_iter())
                .for_each(|(trace_slice, request)| {
                    fill_valid_rows(
                        request.base,
                        request.bit_src.as_canonical_u32(),
                        request.num_bits,
                        request.shift_bits,
                        request.shift_mult,
                        trace_slice,
                        width,
                    );
                });
        });

        tracing::info_span!("fill_padding_rows").in_scope(|| {
            padding_slice.par_chunks_exact_mut(width).for_each(|row| {
                let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
                cols.result = F::ONE;
                cols.result_multiplier = F::ONE;
            });
        });

        Some(RowMajorMatrix::new(trace, width))
    }
}

pub(crate) fn fill_valid_rows(
    base: F,
    bit_src: u32,
    n: u8,
    shift_bits: u8,
    shift_mult: u32,
    trace_slice: &mut [F],
    width: usize,
) {
    fill_valid_rows_with_decomp_src(base, bit_src, n, shift_bits, shift_mult, trace_slice, width);
}

pub(crate) fn fill_valid_rows_with_decomp_src(
    base: F,
    decomp_src: u32,
    n: u8,
    shift_bits: u8,
    shift_mult: u32,
    trace_slice: &mut [F],
    width: usize,
) {
    debug_assert!(n < NUM_BITS_MAX_PLUS_ONE as u8);
    debug_assert_eq!(trace_slice.len(), NUM_BITS_MAX_PLUS_ONE * width);
    debug_assert!(decomp_src < (1u32 << (NUM_BITS_MAX_PLUS_ONE - 1)));

    let bases: Vec<_> = base.exp_powers_of_2().take(NUM_BITS_MAX_PLUS_ONE).collect();
    let mut results = [F::ONE; NUM_BITS_MAX_PLUS_ONE];
    let mut acc = F::ONE;
    for step in (0..NUM_BITS_MAX_PLUS_ONE - 1).rev() {
        if step < n as usize && ((decomp_src >> step) & 1) == 1 {
            acc *= bases[step];
        }
        results[step] = acc;
    }

    let mut low_bits_are_zero = true;
    let mut high_bits_all_one = false;
    for step in 0..NUM_BITS_MAX_PLUS_ONE {
        if step == LOW_BITS_COUNT {
            high_bits_all_one = true;
        }

        let shifted = decomp_src >> step;
        let num_bits = (n as usize).saturating_sub(step);
        let low_bits_left = LOW_BITS_COUNT.saturating_sub(step);

        let row_offset = step * width;
        let row = &mut trace_slice[row_offset..row_offset + width];
        let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(step == 0);
        cols.bit_idx = F::from_usize(step);
        cols.base = bases[step];
        cols.bit_src = F::from_u32(shifted);
        cols.num_bits = F::from_usize(num_bits);
        cols.apply_bit = F::from_bool(num_bits != 0);
        cols.low_bits_left = F::from_usize(low_bits_left);
        cols.in_low_region = F::from_bool(low_bits_left != 0);
        cols.result = results[step];
        cols.result_multiplier = if num_bits != 0 && (shifted & 1) == 1 {
            bases[step]
        } else {
            F::ONE
        };
        cols.bit_src_mod_2 = F::from_bool((shifted & 1) == 1);
        cols.low_bits_are_zero = F::from_bool(low_bits_are_zero);
        cols.high_bits_all_one = F::from_bool(high_bits_all_one);
        cols.bit_src_original = F::from_u32(decomp_src);
        cols.shift_mult = if shift_bits as usize == step {
            F::from_u32(shift_mult)
        } else {
            F::ZERO
        };

        if step < LOW_BITS_COUNT {
            low_bits_are_zero &= (shifted & 1) == 0;
        } else if step + 1 < NUM_BITS_MAX_PLUS_ONE {
            high_bits_all_one &= (shifted & 1) == 1;
        }
    }
}
