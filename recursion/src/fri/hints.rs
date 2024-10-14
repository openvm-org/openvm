use afs_compiler::{
    asm::AsmConfig,
    ir::{Builder, Config, DIGEST_SIZE},
};

use super::types::{BatchOpeningVariable, TwoAdicPcsProofVariable};
use crate::{
    digest::DigestVariable,
    fri::types::{FriCommitPhaseProofStepVariable, FriProofVariable, FriQueryProofVariable},
    hints::{
        Hintable, InnerBatchOpening, InnerChallenge, InnerCommitPhaseStep, InnerDigest,
        InnerFriProof, InnerPcsProof, InnerQueryProof, InnerVal, VecAutoHintable,
    },
    types::InnerConfig,
};

type C = InnerConfig;

impl Hintable<C> for InnerDigest {
    type HintVariable = DigestVariable<C>;

    fn read(builder: &mut Builder<AsmConfig<InnerVal, InnerChallenge>>) -> Self::HintVariable {
        DigestVariable::Felt(builder.hint_felts())
    }

    fn write(&self) -> Vec<Vec<InnerVal>> {
        let h: [InnerVal; DIGEST_SIZE] = *self;
        vec![h.to_vec()]
    }
}

impl VecAutoHintable for InnerDigest {}

impl Hintable<C> for InnerCommitPhaseStep {
    type HintVariable = FriCommitPhaseProofStepVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let sibling_value = builder.hint_ext();
        let opening_proof = Vec::<InnerDigest>::read(builder);
        Self::HintVariable {
            sibling_value,
            opening_proof,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(Hintable::<C>::write(&vec![self.sibling_value]));
        stream.extend(Vec::<InnerDigest>::write(&self.opening_proof));

        stream
    }
}

impl VecAutoHintable for InnerCommitPhaseStep {}

impl Hintable<C> for InnerQueryProof {
    type HintVariable = FriQueryProofVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let commit_phase_openings = Vec::<InnerCommitPhaseStep>::read(builder);
        Self::HintVariable {
            commit_phase_openings,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(Vec::<InnerCommitPhaseStep>::write(
            &self.commit_phase_openings,
        ));

        stream
    }
}

impl VecAutoHintable for InnerQueryProof {}

impl Hintable<C> for InnerFriProof {
    type HintVariable = FriProofVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let commit_phase_commits = Vec::<InnerDigest>::read(builder);
        let query_proofs = Vec::<InnerQueryProof>::read(builder);
        let final_poly = builder.hint_ext();
        let pow_witness = builder.hint_felt();
        Self::HintVariable {
            commit_phase_commits,
            query_proofs,
            final_poly,
            pow_witness,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(Vec::<InnerDigest>::write(
            &self
                .commit_phase_commits
                .iter()
                .map(|x| (*x).into())
                .collect(),
        ));
        stream.extend(Vec::<InnerQueryProof>::write(&self.query_proofs));
        stream.extend(vec![self.final_poly].write());
        stream.push(vec![self.pow_witness]);

        stream
    }
}

impl Hintable<C> for InnerBatchOpening {
    type HintVariable = BatchOpeningVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let opened_values = Vec::<Vec<InnerVal>>::read(builder);
        let opening_proof = Vec::<InnerDigest>::read(builder);
        Self::HintVariable {
            opened_values,
            opening_proof,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();
        stream.extend(Vec::<Vec<InnerVal>>::write(&self.opened_values));
        stream.extend(Vec::<InnerDigest>::write(&self.opening_proof));
        stream
    }
}

impl VecAutoHintable for InnerBatchOpening {}
impl VecAutoHintable for Vec<InnerBatchOpening> {}

impl Hintable<C> for InnerPcsProof {
    type HintVariable = TwoAdicPcsProofVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let fri_proof = InnerFriProof::read(builder);
        let query_openings = Vec::<Vec<InnerBatchOpening>>::read(builder);
        Self::HintVariable {
            fri_proof,
            query_openings,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();
        stream.extend(self.fri_proof.write());
        stream.extend(self.query_openings.write());
        stream
    }
}
