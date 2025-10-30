use core::iter::zip;
use std::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{SumcheckClaimBus, SumcheckClaimMessage},
    bus::{
        BatchConstraintModuleBus, BatchConstraintModuleMessage, TranscriptBus, TranscriptBusMessage,
    },
    system::Preflight,
};

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct FractionsFolderCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    airs_remaining: T,
    tidx_alpha_beta: T,
    gkr_post_tidx: T, // TODO: this number of tids is really annoying
    tidx: T,
    n_global: T,

    sum_claim_p: [T; D_EF],
    sum_claim_q: [T; D_EF],
    cur_p_sum: [T; D_EF],
    cur_q_sum: [T; D_EF],
    mu: [T; D_EF],
    cur_hash: [T; D_EF],
}

pub struct FractionFolderAir {
    pub transcript_bus: TranscriptBus,
    pub sumcheck_bus: SumcheckClaimBus,
    pub gkr_claim_bus: BatchConstraintModuleBus,
}

impl<F> BaseAirWithPublicValues<F> for FractionFolderAir {}
impl<F> PartitionedBaseAir<F> for FractionFolderAir {}

impl<F> BaseAir<F> for FractionFolderAir {
    fn width(&self) -> usize {
        FractionsFolderCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FractionFolderAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &FractionsFolderCols<AB::Var> = (*local).borrow();
        let _next: &FractionsFolderCols<AB::Var> = (*next).borrow();

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx,
                    value: local.sum_claim_p[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(D_EF + i) + local.tidx,
                    value: local.sum_claim_q[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(2 * D_EF + i) + local.tidx,
                    value: local.mu[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_last,
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx_alpha_beta,
                    value: local.cur_q_sum[i] - local.sum_claim_q[i],
                    is_sample: AB::Expr::ONE,
                },
                local.is_first,
            );
        }
        self.sumcheck_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: AB::Expr::ZERO,
                value: local.cur_hash.map(|x| x.into()),
            },
            local.is_first,
        );

        self.gkr_claim_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintModuleMessage {
                tidx_alpha_beta: local.tidx_alpha_beta.into(),
                tidx: local.gkr_post_tidx.into(),
                n_global: local.n_global.into(),
                gkr_input_layer_claim: [
                    local.cur_p_sum.map(|x| x.into()),
                    local.cur_q_sum.map(|x| x.into()),
                ],
            },
            local.is_last,
        );
    }
}

pub(crate) fn generate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = FractionsFolderCols::<F>::width();

    let height = proofs
        .iter()
        .map(|p| {
            let res = p.batch_constraint_proof.numerator_term_per_air.len();
            debug_assert!(res > 0);
            res
        })
        .sum::<usize>();
    let padded_height = height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];
    let mut cur_height = 0;

    for (pidx, (proof, preflight)) in zip(proofs, preflights).enumerate() {
        let (npa, dpa) = (
            &proof.batch_constraint_proof.numerator_term_per_air,
            &proof.batch_constraint_proof.denominator_term_per_air,
        );
        let height = npa.len();
        let mu_tidx = preflight.batch_constraint.tidx_before_univariate - D_EF;
        let mu_slice = &preflight.transcript.values()[mu_tidx..mu_tidx + D_EF];
        let tidx_alpha_beta = preflight.proof_shape.post_tidx + 2;
        let gkr_post_tidx = preflight.gkr.post_tidx;
        let n_global = preflight.proof_shape.n_global();

        let rows = &mut trace[cur_height * width..(cur_height + height) * width];
        rows.par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut FractionsFolderCols<F> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(i == 0);
                cols.is_last = F::from_bool(i + 1 == height);
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.airs_remaining = F::from_canonical_usize(height - 1 - i);
                cols.n_global = F::from_canonical_usize(n_global);
                cols.tidx_alpha_beta = F::from_canonical_usize(tidx_alpha_beta);
                cols.sum_claim_p.copy_from_slice(npa[i].as_base_slice());
                cols.sum_claim_q.copy_from_slice(dpa[i].as_base_slice());
                cols.gkr_post_tidx = F::from_canonical_usize(gkr_post_tidx);
                cols.mu.copy_from_slice(mu_slice);
                cols.tidx = F::from_canonical_usize(gkr_post_tidx + (1 + 2 * i) * D_EF);
            });

        let mut cur_p_sum = [F::ZERO; D_EF];
        let mut cur_q_sum: [_; D_EF] =
            core::array::from_fn(|i| preflight.transcript.values()[tidx_alpha_beta + i]);

        for chunk in rows.chunks_exact_mut(width) {
            let cols: &mut FractionsFolderCols<F> = chunk.borrow_mut();
            for j in 0..D_EF {
                cur_p_sum[j] += cols.sum_claim_p[j];
                cur_q_sum[j] += cols.sum_claim_q[j];
            }
            cols.cur_p_sum.copy_from_slice(&cur_p_sum);
            cols.cur_q_sum.copy_from_slice(&cur_q_sum);
        }

        cur_height += height;
    }
    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut FractionsFolderCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
