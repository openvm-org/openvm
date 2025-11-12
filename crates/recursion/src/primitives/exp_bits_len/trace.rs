use core::borrow::BorrowMut;
use std::sync::Mutex;

use p3_field::{Field, FieldAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{F, poly_common::Squarable};

use super::air::ExpBitsLenCols;

#[derive(Clone, Debug)]
struct ExpBitsLenRequest {
    base: F,
    bit_src: F,
    num_bits: usize,
}

#[derive(Clone, Debug)]
struct ExpBitsLenRecord {
    base: F,
    bit_src: F,
    num_bits: usize,
    result: F,
    sub_result: F,
}

#[derive(Debug, Default)]
pub struct ExpBitsLenCpuTraceGenerator {
    requests: Mutex<Vec<ExpBitsLenRequest>>,
}

impl ExpBitsLenCpuTraceGenerator {
    pub fn add_exp_bits_len(&self, base: F, bit_src: F, num_bits: usize) {
        let request = ExpBitsLenRequest {
            base,
            bit_src,
            num_bits,
        };
        self.requests.lock().unwrap().push(request);
    }

    #[tracing::instrument(name = "generate_trace(ExpBitsLenAir)", skip_all)]
    pub fn generate_trace_row_major(&self) -> RowMajorMatrix<F> {
        let requests = self.requests.lock().unwrap();
        let width = ExpBitsLenCols::<F>::width();

        let num_valid_rows = requests.iter().map(|r| r.num_bits + 1).sum();

        let mut records = Vec::with_capacity(num_valid_rows);
        // This can be done in parallel by reserving the right region of rows
        // (offset determined by prefix sum).
        for request in requests.iter() {
            records.extend(build_records(
                request.base,
                request.bit_src.as_canonical_u32(),
                request.num_bits,
            ));
        }

        let num_rows = num_valid_rows.next_power_of_two();
        let mut trace = vec![F::ZERO; num_rows * width];

        trace
            .par_chunks_exact_mut(width)
            .zip(records.par_iter())
            .for_each(|(row, record)| {
                let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();

                let num_bits_f = F::from_canonical_usize(record.num_bits);
                let bit_src_u32 = record.bit_src.as_canonical_u32();

                cols.is_valid = F::ONE;
                cols.base = record.base;
                cols.bit_src = record.bit_src;
                cols.bit_src_div_2 = F::from_canonical_u32(bit_src_u32 / 2);
                cols.bit_src_mod_2 = F::from_canonical_u32(bit_src_u32 % 2);
                cols.num_bits = num_bits_f;
                cols.num_bits_inv = num_bits_f.try_inverse().unwrap_or(F::ZERO);
                cols.result = record.result;
                cols.sub_result = record.sub_result;
            });

        // dummy rows; 0^0 = 1
        trace
            .par_chunks_exact_mut(width)
            .skip(num_valid_rows)
            .for_each(|row| {
                let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
                cols.result = F::ONE;
            });

        RowMajorMatrix::new(trace, ExpBitsLenCols::<F>::width())
    }

    pub fn clear(&self) {
        self.requests.lock().unwrap().clear();
    }
}

fn build_records(base: F, bit_src: u32, n: usize) -> Vec<ExpBitsLenRecord> {
    let bases: Vec<_> = base.exp_powers_of_2().take(n + 1).collect();

    let mut rows = Vec::with_capacity(n + 1);
    let mut acc = F::ONE;

    for i in (0..=n).rev() {
        let remaining = n - i;
        let shifted = bit_src >> i;

        let (result, sub_result) = if i == n {
            (F::ONE, F::ONE)
        } else {
            let sub = acc;
            if (bit_src >> i) & 1 == 1 {
                acc *= bases[i];
            }
            (acc, sub)
        };

        rows.push(ExpBitsLenRecord {
            base: bases[i],
            bit_src: F::from_canonical_u32(shifted),
            num_bits: remaining,
            result,
            sub_result,
        });
    }

    rows
}
