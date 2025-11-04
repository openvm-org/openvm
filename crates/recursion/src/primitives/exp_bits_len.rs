use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{F, poly_common::Squarable};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::bus::{ExpBitsLenBus, ExpBitsLenMessage};

#[repr(C)]
#[derive(AlignedBorrow)]
struct ExpBitsLenCols<T> {
    is_valid: T,
    base: T,
    bit_src: T,
    num_bits: T,
    num_bits_inv: T,
    result: T,
    sub_result: T,
    // bit_src = 2 * q_mod_2 + r_mod_2
    bit_src_div_2: T,
    bit_src_mod_2: T,
}

#[derive(Debug)]
pub struct ExpBitsLenAir {
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub(crate) requests: std::sync::Mutex<Vec<ExpBitsLenRequest>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ExpBitsLenRequest {
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

impl BaseAirWithPublicValues<F> for ExpBitsLenAir {}
impl PartitionedBaseAir<F> for ExpBitsLenAir {}

impl<F> BaseAir<F> for ExpBitsLenAir {
    fn width(&self) -> usize {
        ExpBitsLenCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpBitsLenAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &ExpBitsLenCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.bit_src_mod_2);
        builder.assert_eq(
            local.bit_src,
            local.bit_src_div_2 * AB::Expr::TWO + local.bit_src_mod_2,
        );
        builder.assert_eq(
            local.num_bits,
            local.num_bits * local.num_bits * local.num_bits_inv,
        );
        builder.when(local.num_bits).assert_eq(
            local.result,
            local.sub_result
                * (local.bit_src_mod_2 * local.base + AB::Expr::ONE - local.bit_src_mod_2),
        );

        let is_num_bits_nonzero = local.num_bits * local.num_bits_inv;
        self.exp_bits_len_bus.add_key_with_lookups(
            builder,
            ExpBitsLenMessage {
                base: local.base,
                bit_src: local.bit_src,
                num_bits: local.num_bits,
                result: local.result,
            },
            local.is_valid,
        );
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.base * local.base,
                bit_src: local.bit_src_div_2.into(),
                num_bits: local.num_bits - AB::Expr::ONE,
                result: local.sub_result.into(),
            },
            is_num_bits_nonzero.clone(),
        );
        builder
            .when(AB::Expr::ONE - is_num_bits_nonzero)
            .assert_one(local.result);
    }
}

impl ExpBitsLenAir {
    pub fn new(exp_bits_len_bus: ExpBitsLenBus) -> Self {
        Self {
            exp_bits_len_bus,
            requests: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn add_exp_bits_len(&self, base: F, bit_src: F, num_bits: usize) {
        let request = ExpBitsLenRequest {
            base,
            bit_src,
            num_bits,
        };
        self.requests.lock().unwrap().push(request);
    }

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

        RowMajorMatrix::new(trace, BaseAir::<F>::width(self))
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
