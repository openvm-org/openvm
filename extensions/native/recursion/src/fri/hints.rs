use openvm_native_compiler::{
    asm::AsmConfig,
    ir::{Builder, Config, Usize, DIGEST_SIZE},
};
use openvm_stark_backend::p3_field::FieldAlgebra;

use super::types::BatchOpeningVariable;
use crate::{
    digest::DigestVariable,
    fri::types::{FriCommitPhaseProofStepVariable, FriProofVariable, FriQueryProofVariable},
    hints::{
        Hintable, InnerBatchOpening, InnerChallenge, InnerCommitPhaseStep, InnerDigest,
        InnerFriProof, InnerQueryProof, InnerVal, VecAutoHintable,
    },
    types::InnerConfig,
    vars::HintSlice,
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
        let opening_proof = read_opening_proof(builder);
        Self::HintVariable {
            sibling_value,
            opening_proof,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(Hintable::<C>::write(&vec![self.sibling_value]));
        stream.extend(write_opening_proof(&self.opening_proof));

        stream
    }
}

impl VecAutoHintable for InnerCommitPhaseStep {}

impl Hintable<C> for InnerQueryProof {
    type HintVariable = FriQueryProofVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let input_proof = Vec::<InnerBatchOpening>::read(builder);
        let commit_phase_openings = Vec::<InnerCommitPhaseStep>::read(builder);
        Self::HintVariable {
            input_proof,
            commit_phase_openings,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(self.input_proof.write());
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
        let final_poly = builder.hint_exts();
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
        stream.extend(self.final_poly.write());
        stream.push(vec![self.pow_witness]);

        stream
    }
}

impl Hintable<C> for InnerBatchOpening {
    type HintVariable = BatchOpeningVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        builder.cycle_tracker_start("HintOpenedValues");
        let opened_values = Vec::<Vec<InnerVal>>::read(builder);
        builder.cycle_tracker_end("HintOpenedValues");
        builder.cycle_tracker_start("HintOpeningProof");
        let opening_proof = read_opening_proof(builder);
        builder.cycle_tracker_end("HintOpeningProof");
        Self::HintVariable {
            opened_values,
            opening_proof,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();
        stream.extend(Vec::<Vec<InnerVal>>::write(&self.opened_values));
        stream.extend(write_opening_proof(&self.opening_proof));
        stream
    }
}

impl VecAutoHintable for InnerBatchOpening {}
impl VecAutoHintable for Vec<InnerBatchOpening> {}

fn read_opening_proof(builder: &mut Builder<C>) -> HintSlice<C> {
    let length = Usize::from(builder.hint_var());
    let id = Usize::from(builder.hint_load());
    HintSlice { length, id }
}

fn write_opening_proof(opening_proof: &[InnerDigest]) -> Vec<Vec<InnerVal>> {
    vec![
        vec![InnerVal::from_canonical_usize(opening_proof.len())],
        opening_proof.iter().flatten().copied().collect(),
    ]
}
