use std::borrow::BorrowMut;

use openvm_stark_backend::keygen::types::MultiStarkVerifyingKey;
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};

use crate::{
    batch_constraint::{eq_airs::eq_ns::air::EqNsColumns, SelectorCount},
    system::Preflight,
    tracegen::RowMajorChip,
    utils::MultiVecWithBounds,
};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct EqNsRecord {
    xi: EF,
    r: EF,
    eq_r_ones: EF,
    eq_r_zeroes: EF,
    r_prod: EF,
    eq: EF,
    eq_sharp: EF,
    sel_first_count: usize,
    sel_last_and_trans_count: usize,
    n_logup: usize,
    n_max: usize,
}

pub struct EqNsTraceGenerator;

impl RowMajorChip<F> for EqNsTraceGenerator {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a [&'a Preflight],
        &'a MultiVecWithBounds<SelectorCount, 1>,
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (vk, preflights, selector_counts) = ctx;
        let l_skip = vk.inner.params.l_skip;
        let records = preflights
            .iter()
            .enumerate()
            .map(|(pidx, preflight)| {
                let selector_counts = &selector_counts[[pidx]];
                let n_global = preflight.proof_shape.n_global();
                let n_max = preflight.proof_shape.n_max;
                let rs = &preflight.batch_constraint.sumcheck_rnd;
                let xi = &preflight.batch_constraint.xi;
                let mut res = Vec::with_capacity(n_global + 1);
                let mut eq_r_ones = EF::ONE;
                let mut eq_r_zeroes = EF::ONE;
                for i in 0..n_max {
                    let counts = selector_counts[i + l_skip];
                    res.push(EqNsRecord {
                        xi: xi[l_skip + i],
                        r: rs[1 + i],
                        r_prod: EF::ONE,
                        eq: preflight.batch_constraint.eq_ns[i],
                        eq_sharp: preflight.batch_constraint.eq_sharp_ns[i],
                        eq_r_ones,
                        eq_r_zeroes,
                        n_logup: preflight.proof_shape.n_logup,
                        n_max: preflight.proof_shape.n_max,
                        sel_first_count: counts.first,
                        sel_last_and_trans_count: counts.last + counts.transition,
                    });
                    eq_r_ones *= rs[1 + i];
                    eq_r_zeroes *= EF::ONE - rs[1 + i];
                }
                let counts = selector_counts[l_skip + n_max];
                for i in n_max..=n_global {
                    let (sel_first_count, sel_last_and_trans_count) = if i == n_max {
                        (counts.first, counts.last + counts.transition)
                    } else {
                        (0, 0)
                    };
                    let xi = if i == n_global {
                        EF::ZERO
                    } else {
                        xi[l_skip + i]
                    };
                    res.push(EqNsRecord {
                        xi,
                        r: EF::ONE,
                        r_prod: EF::ONE,
                        eq: preflight.batch_constraint.eq_ns[n_max],
                        eq_sharp: preflight.batch_constraint.eq_sharp_ns[n_max],
                        eq_r_ones,
                        eq_r_zeroes,
                        n_logup: preflight.proof_shape.n_logup,
                        n_max: preflight.proof_shape.n_max,
                        sel_first_count,
                        sel_last_and_trans_count,
                    });
                }
                for i in (0..n_global).rev() {
                    res[i].r_prod = res[i + 1].r_prod * res[i].r;
                }
                res
            })
            .collect::<Vec<_>>();

        let width = EqNsColumns::<F>::width();
        let total_height = records.iter().map(|rows| rows.len()).sum::<usize>();
        let padded_height = if let Some(height) = required_height {
            if height < total_height {
                return None;
            }
            height
        } else {
            total_height.next_power_of_two()
        };

        let mut trace = vec![F::ZERO; padded_height * width];
        let mut cur_height = 0;
        for (pidx, rows) in records.iter().enumerate() {
            trace[cur_height * width..(cur_height + rows.len()) * width]
                .par_chunks_exact_mut(width)
                .zip(rows.par_iter())
                .enumerate()
                .for_each(|(i, (chunk, record))| {
                    let cols: &mut EqNsColumns<_> = chunk.borrow_mut();
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(i == 0);
                    cols.proof_idx = F::from_usize(pidx);
                    cols.n = F::from_usize(i);
                    cols.n_logup = F::from_usize(record.n_logup);
                    cols.n_max = F::from_usize(record.n_max);
                    cols.n_less_than_n_logup = F::from_bool(i < record.n_logup);
                    cols.n_less_than_n_max = F::from_bool(i < record.n_max);
                    cols.is_transition_and_n_less_than_n_max =
                        F::from_bool(i + 1 < rows.len() && i < record.n_max);
                    cols.xi_n
                        .copy_from_slice(record.xi.as_basis_coefficients_slice());
                    cols.r_n
                        .copy_from_slice(record.r.as_basis_coefficients_slice());
                    cols.r_product
                        .copy_from_slice(record.r_prod.as_basis_coefficients_slice());
                    cols.r_pref_product
                        .copy_from_slice(record.eq_r_ones.as_basis_coefficients_slice());
                    cols.one_minus_r_pref_prod
                        .copy_from_slice(record.eq_r_zeroes.as_basis_coefficients_slice());
                    cols.eq
                        .copy_from_slice(record.eq.as_basis_coefficients_slice());
                    cols.eq_sharp
                        .copy_from_slice(record.eq_sharp.as_basis_coefficients_slice());
                    cols.sel_first_count = F::from_usize(record.sel_first_count);
                    cols.sel_last_and_trans_count = F::from_usize(record.sel_last_and_trans_count);
                });
            let mut num_n_lift_int = vec![0; preflights[pidx].proof_shape.n_logup + 1];
            let mut num_n_lift_con = vec![0; preflights[pidx].proof_shape.n_max + 1];
            for (air_idx, vdata) in preflights[pidx].proof_shape.sorted_trace_vdata.iter() {
                let num_interactions = vk.inner.per_air[*air_idx].num_interactions();
                let n_lift = vdata.log_height.saturating_sub(l_skip);
                num_n_lift_con[n_lift] += 1;
                if num_interactions > 0 {
                    num_n_lift_int[n_lift] += num_interactions;
                }
            }
            let mut xi_mult = 0;
            for (chunk, cnt) in trace[cur_height * width..]
                .chunks_mut(width)
                .zip(num_n_lift_int.into_iter())
            {
                xi_mult += cnt;
                let cols: &mut EqNsColumns<_> = chunk.borrow_mut();
                cols.xi_mult = F::from_usize(xi_mult);
            }
            for (chunk, cnt) in trace[cur_height * width..]
                .chunks_mut(width)
                .zip(num_n_lift_con.into_iter())
            {
                let cols: &mut EqNsColumns<_> = chunk.borrow_mut();
                cols.num_traces = F::from_usize(cnt);
            }
            cur_height += rows.len();
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}
