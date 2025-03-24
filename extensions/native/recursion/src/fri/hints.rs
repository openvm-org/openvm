use openvm_native_compiler::{
    asm::AsmConfig,
    ir::{Builder, Config, Usize, DIGEST_SIZE},
};
use openvm_stark_backend::{
    p3_commit::Mmcs,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use p3_fri::{BatchOpening, CommitPhaseProofStep, QueryProof};

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

// impl<C: Config> Hintable<C> for [C::F; DIGEST_SIZE]
// where
//     C::F: Copy,
// {
//     type HintVariable = DigestVariable<C>;

//     fn read(builder: &mut Builder<C>) -> Self::HintVariable {
//         let digest = builder.hint_felts_fixed(DIGEST_SIZE);
//         DigestVariable::Felt(digest)
//     }

//     fn write(&self) -> Vec<Vec<C::F>> {
//         let h: [C::F; DIGEST_SIZE] = *self;
//         h.map(|x| vec![x]).to_vec()
//     }
// }

impl<F> VecAutoHintable for [F; DIGEST_SIZE] {}

impl<F: Field, C: Config<F = F, N = F>, M: Mmcs<C::EF, Proof = Vec<[C::EF; DIGEST_SIZE]>>>
    Hintable<C> for CommitPhaseProofStep<C::EF, M>
{
    type HintVariable = FriCommitPhaseProofStepVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let sibling_value = builder.hint_ext();
        let opening_proof = read_hint_slice(builder);
        Self::HintVariable {
            sibling_value,
            opening_proof,
        }
    }

    fn write(&self) -> Vec<Vec<F>> {
        let mut stream = Vec::new();

        stream.extend(Hintable::<C>::write(&self.sibling_value));
        stream.extend(write_opening_proof(&self.opening_proof));

        stream
    }
}

impl<F: Field, M: Mmcs<F>> VecAutoHintable for CommitPhaseProofStep<F, M> {}

impl<F: PrimeField32, C: Config<F = F, N = F>, M: Mmcs<C::EF, Proof = Vec<[F; DIGEST_SIZE]>>>
    Hintable<C> for QueryProof<C::EF, M, Vec<BatchOpening<C::EF, M>>>
{
    type HintVariable = FriQueryProofVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let input_proof = Vec::<InnerBatchOpening>::read(builder);
        let commit_phase_openings = Vec::<CommitPhaseProofStep<F, M>>::read(builder);
        Self::HintVariable {
            input_proof,
            commit_phase_openings,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(self.input_proof.write());
        stream.extend(Vec::<CommitPhaseProofStep<F, M>>::write(
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
        let opened_values = read_hint_slice(builder);
        builder.cycle_tracker_end("HintOpenedValues");
        builder.cycle_tracker_start("HintOpeningProof");
        let opening_proof = read_hint_slice(builder);
        builder.cycle_tracker_end("HintOpeningProof");
        Self::HintVariable {
            opened_values,
            opening_proof,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::F>> {
        let mut stream = Vec::new();
        let flat_opened_values: Vec<_> = self.opened_values.iter().flatten().copied().collect();
        stream.extend(vec![
            vec![InnerVal::from_canonical_usize(flat_opened_values.len())],
            flat_opened_values,
        ]);
        stream.extend(write_opening_proof(&self.opening_proof));
        stream
    }
}

impl VecAutoHintable for InnerBatchOpening {}
impl VecAutoHintable for Vec<InnerBatchOpening> {}

fn read_hint_slice<C: Config>(builder: &mut Builder<C>) -> HintSlice<C> {
    let length = Usize::from(builder.hint_var());
    let id = Usize::from(builder.hint_load());
    HintSlice { length, id }
}

fn write_opening_proof<F: FieldAlgebra + Copy>(opening_proof: &[[F; DIGEST_SIZE]]) -> Vec<Vec<F>> {
    vec![
        vec![F::from_canonical_usize(opening_proof.len())],
        opening_proof.iter().flatten().copied().collect(),
    ]
}
