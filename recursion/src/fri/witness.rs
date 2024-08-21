use afs_compiler::ir::{Builder, Witness};
use p3_bn254_fr::Bn254Fr;

use super::types::{BatchOpeningVariable, TwoAdicPcsProofVariable};
use crate::{
    config::outer::{
        OuterBatchOpening, OuterCommitPhaseStep, OuterConfig, OuterFriProof, OuterPcsProof,
        OuterQueryProof,
    },
    digest::DigestVal,
    fri::types::{FriCommitPhaseProofStepVariable, FriProofVariable, FriQueryProofVariable},
    witness::{VectorWitnessable, Witnessable},
};

type C = OuterConfig;

fn to_digest_val_vec(v: &[[Bn254Fr; 1]]) -> Vec<DigestVal<C>> {
    v.iter().map(|x| DigestVal::N(x.to_vec())).collect()
}

impl Witnessable<C> for OuterCommitPhaseStep {
    type WitnessVariable = FriCommitPhaseProofStepVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let sibling_value = self.sibling_value.read(builder);
        let opening_proof = to_digest_val_vec(&self.opening_proof).read(builder);
        Self::WitnessVariable {
            sibling_value,
            opening_proof,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.sibling_value.write(witness);
        to_digest_val_vec(&self.opening_proof).write(witness);
    }
}

impl VectorWitnessable<C> for OuterCommitPhaseStep {}

impl Witnessable<C> for OuterQueryProof {
    type WitnessVariable = FriQueryProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let commit_phase_openings = self.commit_phase_openings.read(builder);
        Self::WitnessVariable {
            commit_phase_openings,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.commit_phase_openings.write(witness);
    }
}

impl VectorWitnessable<C> for OuterQueryProof {}

impl Witnessable<C> for OuterFriProof {
    type WitnessVariable = FriProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let commit_phase_commits = self.commit_phase_commits.read(builder);
        let query_proofs = self.query_proofs.read(builder);
        let final_poly = self.final_poly.read(builder);
        let pow_witness = self.pow_witness.read(builder);
        Self::WitnessVariable {
            commit_phase_commits,
            query_proofs,
            final_poly,
            pow_witness,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.commit_phase_commits.write(witness);
        <Vec<_> as Witnessable<C>>::write(&self.query_proofs, witness);
        self.final_poly.write(witness);
        self.pow_witness.write(witness);
    }
}

impl Witnessable<C> for OuterBatchOpening {
    type WitnessVariable = BatchOpeningVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let opened_values = self.opened_values.read(builder);
        let opening_proof = to_digest_val_vec(&self.opening_proof).read(builder);
        Self::WitnessVariable {
            opened_values,
            opening_proof,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.opened_values.write(witness);
        to_digest_val_vec(&self.opening_proof).write(witness);
    }
}

impl VectorWitnessable<C> for OuterBatchOpening {}
impl VectorWitnessable<C> for Vec<OuterBatchOpening> {}

impl Witnessable<C> for OuterPcsProof {
    type WitnessVariable = TwoAdicPcsProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let fri_proof = self.fri_proof.read(builder);
        let query_openings = self.query_openings.read(builder);
        Self::WitnessVariable {
            fri_proof,
            query_openings,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.fri_proof.write(witness);
        self.query_openings.write(witness);
    }
}
