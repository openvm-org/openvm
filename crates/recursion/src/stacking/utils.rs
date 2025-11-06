use std::array::from_fn;

use itertools::Itertools;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use stark_backend_v2::{
    EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::{eval_eq_mle, eval_eq_prism, eval_in_uni, eval_rot_kernel_prism},
    proof::{Proof, TraceVData},
};

use crate::bus::ColumnClaimsMessage;

pub struct StackedSliceData {
    pub commit_idx: usize,
    pub col_idx: usize,
    pub row_idx: usize,
    pub n: isize,
    pub is_last_for_commit: bool,
}

pub fn get_stacked_slice_data(
    vk: &MultiStarkVerifyingKeyV2,
    sorted_trace_vdata: &[(usize, TraceVData)],
) -> Vec<StackedSliceData> {
    let mut res = Vec::new();

    let mut commit_idx = 0;
    let mut col_idx = 0;
    let mut row_idx = 0;

    let stacked_height = 1 << (vk.inner.params.l_skip + vk.inner.params.n_stack);

    let mut push_res = |log_height: usize, is_last_for_commit| {
        let n = log_height as isize - vk.inner.params.l_skip as isize;
        res.push(StackedSliceData {
            commit_idx,
            col_idx,
            row_idx,
            n,
            is_last_for_commit,
        });
        if is_last_for_commit {
            commit_idx += 1;
            col_idx = 0;
            row_idx = 0;
        } else {
            let col_height = 1 << (n.max(0) as usize + vk.inner.params.l_skip);
            debug_assert!(row_idx + col_height <= stacked_height);
            row_idx = (row_idx + col_height) % stacked_height;
            if row_idx == 0 {
                col_idx += 1;
            }
        }
    };

    for (sort_idx, (air_idx, vdata)) in sorted_trace_vdata.iter().enumerate() {
        let common_width = vk.inner.per_air[*air_idx].params.width.common_main;
        for trace_col_idx in 0..common_width {
            let last_air = sort_idx + 1 == sorted_trace_vdata.len();
            let last_col = trace_col_idx + 1 == common_width;
            push_res(vdata.log_height, last_air && last_col);
        }
    }

    for (air_idx, vdata) in sorted_trace_vdata {
        let trace_width = &vk.inner.per_air[*air_idx].params.width;
        let part_widths = trace_width
            .preprocessed
            .iter()
            .chain(trace_width.cached_mains.iter());
        for &part_width in part_widths {
            for part_col_idx in 0..part_width {
                let is_last = part_col_idx + 1 == part_width;
                push_res(vdata.log_height, is_last);
            }
        }
    }
    res
}

pub fn sorted_column_claims(proof: &Proof) -> Vec<ColumnClaimsMessage<F>> {
    let mut ret = Vec::new();
    let column_openings = &proof.batch_constraint_proof.column_openings;

    for (sort_idx, parts) in column_openings.iter().enumerate() {
        for (col_idx, (col_claim, rot_claim)) in parts[0].iter().enumerate() {
            let msg = ColumnClaimsMessage {
                sort_idx: F::from_canonical_usize(sort_idx),
                part_idx: F::ZERO,
                col_idx: F::from_canonical_usize(col_idx),
                col_claim: from_fn(|i| col_claim.as_base_slice()[i]),
                rot_claim: from_fn(|i| rot_claim.as_base_slice()[i]),
            };
            ret.push(msg);
        }
    }

    for (sort_idx, parts) in column_openings.iter().enumerate() {
        for (part_idx, cols) in parts.iter().enumerate().skip(1) {
            for (col_idx, (col_claim, rot_claim)) in cols.iter().enumerate() {
                let msg = ColumnClaimsMessage {
                    sort_idx: F::from_canonical_usize(sort_idx),
                    part_idx: F::from_canonical_usize(part_idx),
                    col_idx: F::from_canonical_usize(col_idx),
                    col_claim: from_fn(|i| col_claim.as_base_slice()[i]),
                    rot_claim: from_fn(|i| rot_claim.as_base_slice()[i]),
                };
                ret.push(msg);
            }
        }
    }
    ret
}

// Returns (coeff, (eq, k_rot, eq_bits))
#[allow(clippy::type_complexity)]
pub fn compute_coefficients(
    proof: &Proof,
    slice_data: &[StackedSliceData],
    u: &[EF],
    r: &[EF],
    lambda: &EF,
    l_skip: usize,
    n_stack: usize,
) -> (Vec<Vec<EF>>, Vec<(EF, EF, EF)>) {
    let mut coeffs = proof
        .stacking_proof
        .stacking_openings
        .iter()
        .map(|vec| vec![EF::ZERO; vec.len()])
        .collect_vec();
    let lambda_powers = lambda.powers().take(slice_data.len() * 2).collect_vec();
    let mut per_slice = Vec::with_capacity(slice_data.len());
    for (i, slice) in slice_data.iter().enumerate() {
        let n_lift = slice.n.max(0) as usize;
        let b = (l_skip + n_lift..l_skip + n_stack)
            .map(|j| F::from_bool((slice.row_idx >> j) & 1 == 1))
            .collect_vec();
        let eq_mle = eval_eq_mle(&u[n_lift + 1..], &b);
        let ind = eval_in_uni(l_skip, slice.n, u[0]);
        let (l, rs_n) = if slice.n.is_negative() {
            (
                l_skip.wrapping_add_signed(slice.n),
                &[r[0].exp_power_of_2(-slice.n as usize)] as &[_],
            )
        } else {
            (l_skip, &r[..=n_lift])
        };
        let eq_prism = eval_eq_prism(l, &u[..=n_lift], rs_n);
        let rot_kernel_prism = eval_rot_kernel_prism(l, &u[..=n_lift], rs_n);
        coeffs[slice.commit_idx][slice.col_idx] += eq_mle
            * (lambda_powers[2 * i] * eq_prism + lambda_powers[2 * i + 1] * rot_kernel_prism)
            * ind;
        per_slice.push((eq_prism * ind, rot_kernel_prism * ind, eq_mle));
    }
    (coeffs, per_slice)
}
