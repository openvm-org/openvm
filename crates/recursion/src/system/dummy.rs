// Utilities for dummy tracegen
use core::iter::zip;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use stark_backend_v2::{
    EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::Squarable,
    proof::{GkrLayerClaims, Proof},
};

use crate::{
    bus::{
        AirHeightsBusMessage, AirPartShapeBusMessage, AirShapeBusMessage,
        BatchConstraintModuleMessage, ColumnClaimsMessage, CommitmentsBusMessage,
        ConstraintSumcheckRandomness, StackingIndexMessage, TranscriptBusMessage,
        WhirModuleMessage, WhirOpeningPointMessage, XiRandomnessMessage,
    },
    system::Preflight,
};

impl Preflight {
    pub(crate) fn batch_constraint_module_msgs(
        &self,
        proof: &Proof,
    ) -> Vec<BatchConstraintModuleMessage<F>> {
        let gkr_input_layer_claim =
            if let Some(last_layer_claims) = proof.gkr_proof.claims_per_layer.last() {
                let &GkrLayerClaims {
                    p_xi_0,
                    p_xi_1,
                    q_xi_0,
                    q_xi_1,
                } = last_layer_claims;
                let rho = self.gkr.xi[0].1;
                let input_layer_p_claim = p_xi_0 + rho * (p_xi_1 - p_xi_0);
                let input_layer_q_claim = q_xi_0 + rho * (q_xi_1 - q_xi_0);
                [
                    input_layer_p_claim.as_base_slice().try_into().unwrap(),
                    input_layer_q_claim.as_base_slice().try_into().unwrap(),
                ]
            } else {
                [[F::ZERO; 4], [F::ZERO; 4]]
            };

        vec![BatchConstraintModuleMessage {
            // Skip grinding nonce observation and grinding challenge sampling
            tidx_alpha_beta: F::from_canonical_usize(self.proof_shape.post_tidx) + F::TWO,
            tidx: F::from_canonical_usize(self.gkr.post_tidx),
            n_max: F::from_canonical_usize(self.proof_shape.n_max),
            gkr_input_layer_claim,
        }]
    }

    pub(crate) fn xi_randomness_messages(&self) -> Vec<XiRandomnessMessage<F>> {
        self.gkr
            .xi
            .iter()
            .enumerate()
            .map(|(i, (_, xi))| XiRandomnessMessage {
                idx: F::from_canonical_usize(i),
                challenge: xi.as_base_slice().try_into().unwrap(),
            })
            .collect()
    }

    pub(crate) fn batch_constraint_sumcheck_randomness(
        &self,
    ) -> Vec<ConstraintSumcheckRandomness<F>> {
        (0..self.proof_shape.n_max + 1)
            .map(|i| ConstraintSumcheckRandomness {
                idx: F::from_canonical_usize(i),
                challenge: self.batch_constraint.sumcheck_rnd[i]
                    .as_base_slice()
                    .try_into()
                    .unwrap(),
            })
            .collect()
    }

    pub(crate) fn column_claims_messages(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
    ) -> Vec<ColumnClaimsMessage<F>> {
        let mut i = 0;
        let mut column_claims_bus_msgs = vec![];
        for (sort_idx, (air_id, _)) in self.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let vk = &vk.inner.per_air[*air_id];
            for col in 0..vk.params.width.common_main {
                let (col_claim, rot_claim) =
                    proof.batch_constraint_proof.column_openings[sort_idx][0][col];
                column_claims_bus_msgs.push(ColumnClaimsMessage {
                    idx: F::from_canonical_usize(i),
                    sort_idx: F::from_canonical_usize(sort_idx),
                    part_idx: F::ZERO,
                    col_idx: F::from_canonical_usize(col),
                    col_claim: col_claim.as_base_slice().try_into().unwrap(),
                    rot_claim: rot_claim.as_base_slice().try_into().unwrap(),
                });
                i += 1
            }
        }
        for (sort_idx, (air_id, _)) in self.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let vk = &vk.inner.per_air[*air_id];
            let width = &vk.params.width;
            let widths = width.preprocessed.iter().chain(width.cached_mains.iter());

            for (part, width) in widths.enumerate() {
                for col in 0..*width {
                    let (col_claim, rot_claim) =
                        proof.batch_constraint_proof.column_openings[sort_idx][part + 1][col];
                    column_claims_bus_msgs.push(ColumnClaimsMessage {
                        idx: F::from_canonical_usize(i),
                        sort_idx: F::from_canonical_usize(sort_idx),
                        part_idx: F::from_canonical_usize(part + 1),
                        col_idx: F::from_canonical_usize(col),
                        col_claim: col_claim.as_base_slice().try_into().unwrap(),
                        rot_claim: rot_claim.as_base_slice().try_into().unwrap(),
                    });
                    i += 1;
                }
            }
        }
        column_claims_bus_msgs
    }

    pub(crate) fn air_bus_msgs(&self, vk: &MultiStarkVerifyingKeyV2) -> Vec<AirShapeBusMessage<F>> {
        self.proof_shape
            .sorted_trace_vdata
            .iter()
            .enumerate()
            .map(|(sort_idx, (air_id, vdata))| {
                let vk = &vk.inner.per_air[*air_id];
                AirShapeBusMessage {
                    sort_idx: F::from_canonical_usize(sort_idx),
                    air_id: F::from_canonical_usize(*air_id),
                    hypercube_dim: F::from_canonical_usize(vdata.hypercube_dim),
                    has_preprocessed: F::from_bool(vk.preprocessed_data.is_some()),
                    num_main_parts: F::from_canonical_usize(1 + vdata.cached_commitments.len()),
                    num_interactions: F::from_canonical_usize(
                        vk.symbolic_constraints.interactions.len(),
                    ),
                }
            })
            .collect()
    }

    pub(crate) fn air_part_bus_msgs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
    ) -> Vec<AirPartShapeBusMessage<F>> {
        self.proof_shape
            .sorted_trace_vdata
            .iter()
            .enumerate()
            .flat_map(|(sort_idx, (air_id, _))| {
                let vk = &vk.inner.per_air[*air_id];

                let mut parts = vec![];
                let mut part = F::ZERO;

                parts.push(AirPartShapeBusMessage {
                    idx: F::from_canonical_usize(sort_idx),
                    part,
                    width: F::from_canonical_usize(vk.params.width.common_main),
                });
                if let Some(width) = &vk.params.width.preprocessed {
                    part += F::ONE;
                    parts.push(AirPartShapeBusMessage {
                        idx: F::from_canonical_usize(sort_idx),
                        part,
                        width: F::from_canonical_usize(*width),
                    });
                }
                for width in &vk.params.width.cached_mains {
                    part += F::ONE;
                    parts.push(AirPartShapeBusMessage {
                        idx: F::from_canonical_usize(sort_idx),
                        part,
                        width: F::from_canonical_usize(*width),
                    });
                }
                parts
            })
            .collect()
    }

    pub(crate) fn air_heights_bus_msgs_and_widths(
        &self,
        mvk: &MultiStarkVerifyingKeyV2,
    ) -> Vec<(AirHeightsBusMessage<F>, usize)> {
        self.proof_shape
            .sorted_trace_vdata
            .iter()
            .enumerate()
            .map(|(sort_idx, (air_id, vdata))| {
                let vk = &mvk.inner.per_air[*air_id];
                let log_height = vdata.hypercube_dim + mvk.inner.params.l_skip;
                let mut total_width = vk.params.width.common_main;

                for width in vk
                    .params
                    .width
                    .preprocessed
                    .iter()
                    .chain(&vk.params.width.cached_mains)
                {
                    total_width += *width;
                }

                (
                    AirHeightsBusMessage {
                        sort_idx: F::from_canonical_usize(sort_idx),
                        log_height: F::from_canonical_usize(log_height),
                        height: F::from_canonical_usize(1 << log_height),
                    },
                    total_width,
                )
            })
            .collect()
    }

    pub fn whir_module_msg(&self, proof: &Proof) -> WhirModuleMessage<F> {
        let mu = self.stacking.stacking_batching_challenge;
        let mut mu_pows = mu.powers();
        let mut claim = EF::ZERO;
        for openings in &proof.stacking_proof.stacking_openings {
            for &opening in openings {
                claim += mu_pows.next().unwrap() * opening;
            }
        }
        WhirModuleMessage {
            tidx: F::from_canonical_usize(self.stacking.post_tidx),
            mu: mu.as_base_slice().try_into().unwrap(),
            claim: claim.as_base_slice().try_into().unwrap(),
        }
    }

    pub fn whir_commitments_msgs(&self, proof: &Proof) -> Vec<CommitmentsBusMessage<F>> {
        let mut messages = vec![];
        for (i, commit) in proof.whir_proof.codeword_commits.iter().enumerate() {
            messages.push(CommitmentsBusMessage {
                major_idx: F::from_canonical_usize(i + 1),
                minor_idx: F::ZERO,
                commitment: *commit,
            });
        }
        messages
    }

    pub fn stacking_commitments_msgs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
    ) -> Vec<CommitmentsBusMessage<F>> {
        let mut messages = vec![];
        messages.push(CommitmentsBusMessage {
            major_idx: F::ZERO,
            minor_idx: F::ZERO,
            commitment: proof.common_main_commit,
        });
        let mut commit_idx = F::ONE;
        for (air_id, vdata) in &self.proof_shape.sorted_trace_vdata {
            let vk = &vk.inner.per_air[*air_id];
            let commits = vk
                .preprocessed_data
                .as_ref()
                .into_iter()
                .map(|p| p.commit)
                .chain(vdata.cached_commitments.iter().cloned());

            for commit in commits {
                messages.push(CommitmentsBusMessage {
                    major_idx: F::ZERO,
                    minor_idx: commit_idx,
                    commitment: commit,
                });
                commit_idx += F::ONE;
            }
        }
        messages
    }

    pub fn stacking_indices_bus_msgs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
    ) -> Vec<StackingIndexMessage<F>> {
        let l_skip = vk.inner.params.l_skip;
        let stacking_height = 1 << (l_skip + vk.inner.params.n_stack);
        let mut messages = vec![];
        for i in 0..self.proof_shape.stacked_common_width {
            messages.push(StackingIndexMessage {
                commit_idx: F::ZERO,
                col_idx: F::from_canonical_usize(i),
            });
        }
        let mut commit_idx = F::ONE;
        for (air_id, vdata) in &self.proof_shape.sorted_trace_vdata {
            let width = &vk.inner.per_air[*air_id].params.width;
            let widths = width.preprocessed.iter().chain(width.cached_mains.iter());

            for width in widths {
                let cells = width * (1 << (vdata.hypercube_dim + l_skip));
                let stacking_width = cells.div_ceil(stacking_height);
                for i in 0..stacking_width {
                    messages.push(StackingIndexMessage {
                        commit_idx,
                        col_idx: F::from_canonical_usize(i),
                    });
                }
                commit_idx += F::ONE;
            }
        }
        messages
    }

    pub(crate) fn whir_opening_point_messages(
        &self,
        l_skip: usize,
    ) -> Vec<WhirOpeningPointMessage<F>> {
        let rnd = &self.stacking.sumcheck_rnd;
        rnd[0]
            .exp_powers_of_2()
            .take(l_skip)
            .chain(rnd[1..].iter().copied())
            .enumerate()
            .map(|(i, value)| WhirOpeningPointMessage {
                idx: F::from_canonical_usize(i),
                value: value.as_base_slice().try_into().unwrap(),
            })
            .collect()
    }

    pub(crate) fn transcript_msgs(&self, from: usize, to: usize) -> Vec<TranscriptBusMessage<F>> {
        let values = &self.transcript[from..to];
        let sample_flags = &self.transcript.samples()[from..to];
        zip(values, sample_flags)
            .enumerate()
            .map(|(i, (v, is_sample))| TranscriptBusMessage {
                tidx: F::from_canonical_usize(from + i),
                value: *v,
                is_sample: F::from_bool(*is_sample),
            })
            .collect()
    }
}
