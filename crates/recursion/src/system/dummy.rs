// Utilities for dummy tracegen
use core::iter::zip;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{GkrLayerClaims, Proof},
};

use crate::{
    bus::{
        AirPartShapeBusMessage, AirShapeBusMessage, BatchConstraintModuleMessage,
        ColumnClaimsMessage, CommitmentsBusMessage, ConstraintSumcheckRandomness,
        StackingIndexMessage, StackingSumcheckRandomnessMessage, TranscriptBusMessage,
        WhirModuleMessage, XiRandomnessMessage,
    },
    system::Preflight,
};

impl<TS: FiatShamirTranscript> Preflight<TS> {
    pub(crate) fn batch_constraint_module_msgs(
        &self,
        proof: &Proof,
    ) -> Vec<BatchConstraintModuleMessage<F>> {
        let tidx = self.proof_shape.post_tidx;
        let alpha_logup: [F; D_EF] = self.transcript.data[tidx..tidx + D_EF].try_into().unwrap();
        let beta_logup: [F; D_EF] = self.transcript.data[tidx + D_EF..tidx + 2 * D_EF]
            .try_into()
            .unwrap();

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
            tidx: F::from_canonical_usize(self.gkr.post_tidx),
            alpha_logup,
            beta_logup,
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
                        proof.batch_constraint_proof.column_openings[sort_idx][part][col];
                    column_claims_bus_msgs.push(ColumnClaimsMessage {
                        idx: F::from_canonical_usize(i),
                        sort_idx: F::from_canonical_usize(sort_idx),
                        part_idx: F::from_canonical_usize(part),
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
                    num_main_parts: F::from_canonical_usize(
                        1 + vdata.cached_commitments.len()
                            + vk.preprocessed_data.is_some() as usize,
                    ),
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

    pub fn stacking_widths_bus_msgs(
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

    pub(crate) fn stacking_randomness_msgs(&self) -> Vec<StackingSumcheckRandomnessMessage<F>> {
        self.stacking
            .sumcheck_rnd
            .iter()
            .enumerate()
            .map(|(i, challenge)| StackingSumcheckRandomnessMessage {
                idx: F::from_canonical_usize(i),
                challenge: challenge.as_base_slice().try_into().unwrap(),
            })
            .collect()
    }

    pub(crate) fn transcript_msgs(&self, from: usize, to: usize) -> Vec<TranscriptBusMessage<F>> {
        let values = &self.transcript.data[from..to];
        let is_samples = &self.transcript.is_sample[from..to];
        zip(values, is_samples)
            .enumerate()
            .map(|(i, (v, is_sample))| TranscriptBusMessage {
                tidx: F::from_canonical_usize(from + i),
                value: *v,
                is_sample: F::from_bool(*is_sample),
            })
            .collect()
    }
}
