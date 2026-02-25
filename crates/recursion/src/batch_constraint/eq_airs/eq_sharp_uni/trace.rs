use std::borrow::BorrowMut;

use openvm_stark_backend::{
    keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
    poly_common::Squarable,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, EF, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    batch_constraint::eq_airs::eq_sharp_uni::air::{EqSharpUniCols, EqSharpUniReceiverCols},
    system::Preflight,
    tracegen::RowMajorChip,
    utils::MultiProofVecVec,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct EqSharpUniRecord {
    xi: EF,
    product: EF,

    root: F,
    root_pow: F,

    // PERF: we can only store `xi_idx: u32` and later take `xi` from the transcript
    xi_idx: u32,
    root_half_order: u32,
    iter_idx: u32,
}

#[derive(Debug, Clone)]
pub struct EqSharpUniBlob {
    records: MultiProofVecVec<EqSharpUniRecord>,
    final_products: MultiProofVecVec<EF>,
    rs: Vec<EF>,
}

impl EqSharpUniBlob {
    fn new(l_skip: usize, num_proofs: usize) -> Self {
        Self {
            records: MultiProofVecVec::with_capacity(num_proofs * ((1 << l_skip) - 1)),
            final_products: MultiProofVecVec::with_capacity(num_proofs << l_skip),
            rs: Vec::with_capacity(num_proofs),
        }
    }
}

pub fn generate_eq_sharp_uni_blob(
    vk: &MultiStarkVerifyingKey0<BabyBearPoseidon2Config>,
    preflights: &[&Preflight],
) -> EqSharpUniBlob {
    let l_skip = vk.params.l_skip;
    let mut blob = EqSharpUniBlob::new(l_skip, preflights.len());
    let mut products = vec![EF::ONE; 1 << l_skip];
    let roots = F::two_adic_generator(l_skip)
        .inverse()
        .exp_powers_of_2()
        .take(l_skip)
        .collect::<Vec<_>>();
    for preflight in preflights.iter() {
        products[0] = EF::ONE;
        for i in 0..l_skip {
            let xi_idx = l_skip - 1 - i;
            let xi = preflight.gkr.xi[xi_idx].1;
            let root = roots[l_skip - 1 - i];
            let mut root_pow = F::ONE;
            for (j, &product) in products.iter().take(1 << i).enumerate() {
                blob.records.push(EqSharpUniRecord {
                    xi,
                    product,
                    root,
                    root_pow,
                    xi_idx: xi_idx as u32,
                    root_half_order: 1 << i,
                    iter_idx: j as u32,
                });
                root_pow *= root;
            }
            for j in (0..(1 << i)).rev() {
                let value = products[j];
                let root_pow = blob.records.data()[blob.records.len() - (1 << i) + j].root_pow;
                let second = xi * root_pow;
                products[j + (1 << i)] = value * (EF::ONE - xi - second);
                products[j] = value * (EF::ONE - xi + second);
            }
        }
        blob.records.end_proof();
        blob.final_products.extend_from_slice(&products);
        blob.final_products.end_proof();
        blob.rs.push(preflight.batch_constraint.sumcheck_rnd[0]);
    }
    blob
}

pub struct EqSharpUniTraceGenerator;

impl RowMajorChip<F> for EqSharpUniTraceGenerator {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a EqSharpUniBlob,
        &'a [&'a Preflight],
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (vk, blob, preflights) = ctx;
        let width = EqSharpUniCols::<F>::width();
        let l_skip = vk.inner.params.l_skip;
        let one_height = (1 << l_skip) - 1;
        let total_height = one_height * preflights.len();

        let padded_height = if let Some(height) = required_height {
            if height < total_height {
                return None;
            }
            height
        } else {
            total_height.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padded_height * width];

        for pidx in 0..preflights.len() {
            let records = &blob.records[pidx];
            trace[(pidx * one_height * width)..((pidx + 1) * one_height * width)]
                .par_chunks_exact_mut(width)
                .zip(records.par_iter())
                .enumerate()
                .for_each(|(i, (chunk, record))| {
                    let cols: &mut EqSharpUniCols<_> = chunk.borrow_mut();
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(i == 0);
                    cols.proof_idx = F::from_usize(pidx);
                    cols.is_first_iter = F::ONE;
                    cols.xi_idx = F::from_u32(record.xi_idx);
                    cols.xi
                        .copy_from_slice(record.xi.as_basis_coefficients_slice());
                    cols.iter_idx = F::from_u32(record.iter_idx);
                    cols.is_first_iter = F::from_bool(record.iter_idx == 0);
                    cols.product_before
                        .copy_from_slice(record.product.as_basis_coefficients_slice());
                    cols.root = record.root;
                    cols.root_pow = record.root_pow;
                    cols.root_half_order = F::from_u32(record.root_half_order);
                });
        }

        trace[total_height * width..]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut EqSharpUniCols<F> = chunk.borrow_mut();
                cols.proof_idx = F::from_usize(preflights.len() + i);
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}

pub struct EqSharpUniReceiverTraceGenerator;

impl RowMajorChip<F> for EqSharpUniReceiverTraceGenerator {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a EqSharpUniBlob,
        &'a [&'a Preflight],
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (vk, blob, preflights) = ctx;
        let l_skip = vk.inner.params.l_skip;

        let width = EqSharpUniReceiverCols::<F>::width();
        let one_height = 1 << l_skip;
        let total_height = one_height * preflights.len();
        let padded_height = if let Some(height) = required_height {
            if height < total_height {
                return None;
            }
            height
        } else {
            total_height.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padded_height * width];

        for pidx in 0..preflights.len() {
            let products = &blob.final_products[pidx];
            let r = blob.rs[pidx];
            trace[(pidx * one_height * width)..((pidx + 1) * one_height * width)]
                .par_chunks_exact_mut(width)
                .zip(products.par_iter())
                .enumerate()
                .for_each(|(i, (chunk, product))| {
                    let cols: &mut EqSharpUniReceiverCols<_> = chunk.borrow_mut();
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(i == 0);
                    cols.is_last = F::from_bool(i + 1 == one_height);
                    cols.proof_idx = F::from_usize(pidx);
                    cols.coeff
                        .copy_from_slice(product.as_basis_coefficients_slice());
                    cols.r.copy_from_slice(r.as_basis_coefficients_slice());
                    cols.idx = F::from_usize(i);
                });
            let mut cur_sum = EF::ZERO;
            trace[(pidx * one_height * width)..((pidx + 1) * one_height * width)]
                .chunks_exact_mut(width)
                .rev()
                .for_each(|chunk| {
                    let cols: &mut EqSharpUniReceiverCols<_> = chunk.borrow_mut();
                    cur_sum = cur_sum * r + EF::from_basis_coefficients_slice(&cols.coeff).unwrap();
                    cols.cur_sum
                        .copy_from_slice(cur_sum.as_basis_coefficients_slice());
                });
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}
