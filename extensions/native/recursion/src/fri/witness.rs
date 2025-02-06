use openvm_native_compiler::{
    ir::{Builder, Witness, WitnessRef},
    prelude::*,
};

use super::types::BatchOpeningVariable;
use crate::{
    config::outer::{
        OuterBatchOpening, OuterCommitPhaseStep, OuterConfig, OuterDigest, OuterFriProof,
        OuterQueryProof,
    },
    fri::types::{FriCommitPhaseProofStepVariable, FriProofVariable, FriQueryProofVariable},
    vars::HintSlice,
    witness::{VectorWitnessable, Witnessable},
};

type C = OuterConfig;

impl Witnessable<C> for OuterCommitPhaseStep {
    type WitnessVariable = FriCommitPhaseProofStepVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let sibling_value = self.sibling_value.read(builder);
        let opening_proof = read_opening_proof(builder, &self.opening_proof);
        Self::WitnessVariable {
            sibling_value,
            opening_proof,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.sibling_value.write(witness);
        write_opening_proof(witness, &self.opening_proof);
    }
}

impl VectorWitnessable<C> for OuterCommitPhaseStep {}

impl Witnessable<C> for OuterQueryProof {
    type WitnessVariable = FriQueryProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let input_proof = self.input_proof.read(builder);
        let commit_phase_openings = self.commit_phase_openings.read(builder);
        Self::WitnessVariable {
            input_proof,
            commit_phase_openings,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.input_proof.write(witness);
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
        let opening_proof = read_opening_proof(builder, &self.opening_proof);
        Self::WitnessVariable {
            opened_values,
            opening_proof,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.opened_values.write(witness);
        write_opening_proof(witness, &self.opening_proof);
    }
}

impl VectorWitnessable<C> for OuterBatchOpening {}
impl VectorWitnessable<C> for Vec<OuterBatchOpening> {}

fn read_opening_proof(builder: &mut Builder<C>, opening_proof: &[OuterDigest]) -> HintSlice<C> {
    let opening_proof: Vec<WitnessRef> = opening_proof
        .iter()
        .flatten()
        .map(|x| x.read(builder).into())
        .collect();
    let length = Usize::from(opening_proof.len());
    let id = builder.witness_load(opening_proof);
    HintSlice { length, id }
}
fn write_opening_proof(witness: &mut Witness<OuterConfig>, opening_proof: &[OuterDigest]) {
    let opening_proof: Vec<_> = opening_proof.iter().flat_map(|op| op.to_vec()).collect();
    opening_proof.write(witness);
}
