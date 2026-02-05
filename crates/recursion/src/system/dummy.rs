// Utilities for dummy tracegen
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use stark_backend_v2::{EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    bus::{CommitmentsBusMessage, ConstraintSumcheckRandomness, WhirModuleMessage},
    system::Preflight,
};

impl Preflight {
    pub(crate) fn batch_constraint_sumcheck_randomness(
        &self,
    ) -> Vec<ConstraintSumcheckRandomness<F>> {
        self.batch_constraint
            .sumcheck_rnd
            .iter()
            .enumerate()
            .map(|(i, r)| ConstraintSumcheckRandomness {
                idx: F::from_usize(i),
                challenge: r.as_basis_coefficients_slice().try_into().unwrap(),
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
            tidx: F::from_usize(self.stacking.post_tidx),
            mu: mu.as_basis_coefficients_slice().try_into().unwrap(),
            claim: claim.as_basis_coefficients_slice().try_into().unwrap(),
        }
    }

    pub fn whir_commitments_msgs(&self, proof: &Proof) -> Vec<CommitmentsBusMessage<F>> {
        let mut messages = vec![];
        for (i, commit) in proof.whir_proof.codeword_commits.iter().enumerate() {
            messages.push(CommitmentsBusMessage {
                major_idx: F::from_usize(i + 1),
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
}
