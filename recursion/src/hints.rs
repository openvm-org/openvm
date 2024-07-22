use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, Field};
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof, TwoAdicFriPcsProof};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use afs_compiler::ir::{Array, Builder, Config, Ext, Felt, MemVariable, Var, DIGEST_SIZE};
use afs_stark_backend::keygen::types::TraceWidth;
use afs_stark_backend::prover::opener::{AdjacentOpenedValues, OpenedValues, OpeningProof};
use afs_stark_backend::prover::types::{Commitments, Proof};
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::types::{
    AdjacentOpenedValuesVariable, AxiomCommitmentsVariable, AxiomMemoryLayout,
    AxiomMemoryLayoutVariable, AxiomProofVariable, InnerConfig, OpenedValuesVariable,
    OpeningProofVariable, TraceWidthVariable,
};

pub type InnerVal = BabyBear;
pub type InnerChallenge = BinomialExtensionField<InnerVal, 4>;
pub type InnerPerm =
    Poseidon2<InnerVal, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
pub type InnerHash = PaddingFreeSponge<InnerPerm, 16, 8, 8>;
pub type InnerDigest = [InnerVal; DIGEST_SIZE];
pub type InnerCompress = TruncatedPermutation<InnerPerm, 2, 8, 16>;
pub type InnerValMmcs = FieldMerkleTreeMmcs<
    <InnerVal as Field>::Packing,
    <InnerVal as Field>::Packing,
    InnerHash,
    InnerCompress,
    8,
>;
pub type InnerChallengeMmcs = ExtensionMmcs<InnerVal, InnerChallenge, InnerValMmcs>;
pub type InnerQueryProof = QueryProof<InnerChallenge, InnerChallengeMmcs>;
pub type InnerCommitPhaseStep = CommitPhaseProofStep<InnerChallenge, InnerChallengeMmcs>;
pub type InnerFriProof = FriProof<InnerChallenge, InnerChallengeMmcs, InnerVal>;
pub type InnerBatchOpening = BatchOpening<InnerVal, InnerValMmcs>;
pub type InnerPcsProof =
    TwoAdicFriPcsProof<InnerVal, InnerChallenge, InnerValMmcs, InnerChallengeMmcs>;

pub trait Hintable<C: Config> {
    type HintVariable: MemVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable;

    fn write(&self) -> Vec<Vec<C::F>>;

    fn witness(variable: &Self::HintVariable, builder: &mut Builder<C>) {
        let target = Self::read(builder);
        builder.assign(variable.clone(), target);
    }
}

impl Hintable<InnerConfig> for usize {
    type HintVariable = Var<InnerVal>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_var()
    }

    fn write(&self) -> Vec<Vec<InnerVal>> {
        vec![vec![InnerVal::from_canonical_usize(*self)]]
    }
}

impl Hintable<InnerConfig> for InnerVal {
    type HintVariable = Felt<InnerVal>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_felt()
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        vec![vec![*self]]
    }
}

impl Hintable<InnerConfig> for InnerChallenge {
    type HintVariable = Ext<InnerVal, InnerChallenge>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_ext()
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        vec![self.as_base_slice().to_vec()]
    }
}

pub trait VecAutoHintable<C: Config>: Hintable<C> {}

impl VecAutoHintable<InnerConfig> for Vec<usize> {}

impl VecAutoHintable<InnerConfig> for Vec<InnerVal> {}

impl VecAutoHintable<InnerConfig> for Vec<Vec<InnerChallenge>> {}

impl VecAutoHintable<InnerConfig> for AdjacentOpenedValues<InnerChallenge> {}

impl VecAutoHintable<InnerConfig> for Vec<AdjacentOpenedValues<InnerChallenge>> {}

impl VecAutoHintable<InnerConfig> for Vec<Vec<AdjacentOpenedValues<InnerChallenge>>> {}

impl<I: VecAutoHintable<InnerConfig>> Hintable<InnerConfig> for Vec<I> {
    type HintVariable = Array<InnerConfig, I::HintVariable>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let len = builder.hint_var();
        let mut arr = builder.dyn_array(len);
        builder.range(0, len).for_each(|i, builder| {
            let hint = I::read(builder);
            builder.set(&mut arr, i, hint);
        });
        arr
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        let mut stream = Vec::new();

        let len = InnerVal::from_canonical_usize(self.len());
        stream.push(vec![len]);

        self.iter().for_each(|i| {
            let comm = I::write(i);
            stream.extend(comm);
        });

        stream
    }
}

impl Hintable<InnerConfig> for AxiomMemoryLayout<BabyBearPoseidon2Config> {
    type HintVariable = AxiomMemoryLayoutVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let proof = Proof::<BabyBearPoseidon2Config>::read(builder);
        let log_degree_per_air = Vec::<usize>::read(builder);
        let public_values = Vec::<Vec<InnerVal>>::read(builder);

        AxiomMemoryLayoutVariable {
            proof,
            log_degree_per_air,
            public_values,
        }
    }

    fn write(&self) -> Vec<Vec<InnerVal>> {
        let mut stream = Vec::new();

        stream.extend(self.proof.write());
        stream.extend(self.log_degree_per_air.write());
        stream.extend(self.public_values.write());

        stream
    }
}

impl Hintable<InnerConfig> for Vec<usize> {
    type HintVariable = Array<InnerConfig, Var<InnerVal>>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_vars()
    }

    fn write(&self) -> Vec<Vec<InnerVal>> {
        vec![self
            .iter()
            .map(|x| InnerVal::from_canonical_usize(*x))
            .collect()]
    }
}

impl Hintable<InnerConfig> for Vec<InnerVal> {
    type HintVariable = Array<InnerConfig, Felt<InnerVal>>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_felts()
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        vec![self.clone()]
    }
}

impl Hintable<InnerConfig> for Vec<InnerChallenge> {
    type HintVariable = Array<InnerConfig, Ext<InnerVal, InnerChallenge>>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_exts()
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        vec![
            vec![InnerVal::from_canonical_usize(self.len())],
            self.iter()
                .flat_map(|x| (*x).as_base_slice().to_vec())
                .collect(),
        ]
    }
}

impl Hintable<InnerConfig> for Vec<Vec<InnerChallenge>> {
    type HintVariable = Array<InnerConfig, Array<InnerConfig, Ext<InnerVal, InnerChallenge>>>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let len = builder.hint_var();
        let mut arr = builder.dyn_array(len);
        builder.range(0, len).for_each(|i, builder| {
            let hint = Vec::<InnerChallenge>::read(builder);
            builder.set(&mut arr, i, hint);
        });
        arr
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        let mut stream = Vec::new();

        let len = InnerVal::from_canonical_usize(self.len());
        stream.push(vec![len.into()]);

        self.iter().for_each(|arr| {
            let comm = Vec::<InnerChallenge>::write(arr);
            stream.extend(comm);
        });

        stream
    }
}

impl Hintable<InnerConfig> for TraceWidth {
    type HintVariable = TraceWidthVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let preprocessed = Vec::<usize>::read(builder);
        let partitioned_main = Vec::<usize>::read(builder);
        let after_challenge = Vec::<usize>::read(builder);

        TraceWidthVariable {
            preprocessed,
            partitioned_main,
            after_challenge,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(self.preprocessed.into_iter().collect::<Vec<_>>().write());
        stream.extend(self.partitioned_main.write());
        stream.extend(self.after_challenge.write());

        stream
    }
}

impl Hintable<InnerConfig> for Proof<BabyBearPoseidon2Config> {
    type HintVariable = AxiomProofVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let commitments = Commitments::<BabyBearPoseidon2Config>::read(builder);
        let opening = OpeningProof::<BabyBearPoseidon2Config>::read(builder);
        let exposed_values_after_challenge = Vec::<Vec<Vec<InnerChallenge>>>::read(builder);

        AxiomProofVariable {
            commitments,
            opening,
            exposed_values_after_challenge,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(self.commitments.write());
        stream.extend(self.opening.write());
        stream.extend(self.exposed_values_after_challenge.write());

        stream
    }
}

impl Hintable<InnerConfig>
    for afs_stark_backend::prover::opener::OpeningProof<BabyBearPoseidon2Config>
{
    type HintVariable = OpeningProofVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let proof = InnerPcsProof::read(builder);
        let values = OpenedValues::read(builder);

        OpeningProofVariable { proof, values }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(self.proof.write());
        stream.extend(self.values.write());

        stream
    }
}

impl Hintable<InnerConfig> for OpenedValues<InnerChallenge> {
    type HintVariable = OpenedValuesVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let preprocessed = Vec::<AdjacentOpenedValues<InnerChallenge>>::read(builder);
        let main = Vec::<Vec<AdjacentOpenedValues<InnerChallenge>>>::read(builder);
        let quotient = Vec::<Vec<Vec<InnerChallenge>>>::read(builder);
        let after_challenge = Vec::<Vec<AdjacentOpenedValues<InnerChallenge>>>::read(builder);

        OpenedValuesVariable {
            preprocessed,
            main,
            quotient,
            after_challenge,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(self.preprocessed.write());
        stream.extend(self.main.write());
        stream.extend(self.quotient.write());
        stream.extend(self.after_challenge.write());

        stream
    }
}

impl Hintable<InnerConfig> for AdjacentOpenedValues<InnerChallenge> {
    type HintVariable = AdjacentOpenedValuesVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let local = Vec::<InnerChallenge>::read(builder);
        let next = Vec::<InnerChallenge>::read(builder);
        AdjacentOpenedValuesVariable { local, next }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        let mut stream = Vec::new();
        stream.extend(self.local.write());
        stream.extend(self.next.write());
        stream
    }
}

impl Hintable<InnerConfig> for Commitments<BabyBearPoseidon2Config> {
    type HintVariable = AxiomCommitmentsVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let main_trace = Vec::<InnerDigest>::read(builder);
        let after_challenge = Vec::<InnerDigest>::read(builder);
        let quotient = InnerDigest::read(builder);

        AxiomCommitmentsVariable {
            main_trace,
            after_challenge,
            quotient,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::F>> {
        let mut stream = Vec::new();

        stream.extend(Vec::<InnerDigest>::write(
            &self.main_trace.iter().map(|&x| x.into()).collect(),
        ));
        stream.extend(Vec::<InnerDigest>::write(
            &self.after_challenge.iter().map(|&x| x.into()).collect(),
        ));
        let h: InnerDigest = self.quotient.into();
        stream.extend(h.write());

        stream
    }
}

#[cfg(test)]
mod test {
    use crate::hints::{Hintable, InnerChallenge, InnerVal};
    use afs_compiler::asm::AsmBuilder;
    use afs_compiler::ir::{Ext, Felt, Var};
    use afs_compiler::util::execute_program;
    use p3_field::AbstractField;

    #[test]
    fn test_var_array() {
        let x = vec![
            InnerVal::from_canonical_usize(1),
            InnerVal::from_canonical_usize(2),
            InnerVal::from_canonical_usize(3),
        ];
        let stream = Vec::<InnerVal>::write(&x);
        assert_eq!(stream, vec![x.clone()]);

        let mut builder = AsmBuilder::<InnerVal, InnerChallenge>::default();
        let arr = Vec::<InnerVal>::read(&mut builder);

        let expected: Var<_> = builder.constant(InnerVal::from_canonical_usize(3));
        builder.assert_var_eq(arr.len(), expected);

        for (i, &val) in x.iter().enumerate() {
            let actual = builder.get(&arr, i);
            let expected: Felt<InnerVal> = builder.constant(val);
            builder.assert_felt_eq(actual, expected);
        }

        builder.halt();

        let program = builder.compile_isa::<1>();
        execute_program::<1, _>(program, stream);
    }

    #[test]
    fn test_ext_array() {
        let x = vec![
            InnerChallenge::from_canonical_usize(1),
            InnerChallenge::from_canonical_usize(2),
            InnerChallenge::from_canonical_usize(3),
        ];
        let stream = Vec::<InnerChallenge>::write(&x);
        assert_eq!(
            stream,
            vec![
                vec![InnerVal::from_canonical_usize(x.len())],
                vec![
                    InnerVal::from_canonical_usize(1),
                    InnerVal::from_canonical_usize(0),
                    InnerVal::from_canonical_usize(0),
                    InnerVal::from_canonical_usize(0),
                    InnerVal::from_canonical_usize(2),
                    InnerVal::from_canonical_usize(0),
                    InnerVal::from_canonical_usize(0),
                    InnerVal::from_canonical_usize(0),
                    InnerVal::from_canonical_usize(3),
                    InnerVal::from_canonical_usize(0),
                    InnerVal::from_canonical_usize(0),
                    InnerVal::from_canonical_usize(0),
                ],
            ]
        );

        let mut builder = AsmBuilder::<InnerVal, InnerChallenge>::default();
        let arr = Vec::<InnerChallenge>::read(&mut builder);

        let expected: Var<_> = builder.constant(InnerVal::from_canonical_usize(3));
        builder.assert_var_eq(arr.len(), expected);

        for (i, &val) in x.iter().enumerate() {
            let actual = builder.get(&arr, i);
            let expected: Ext<InnerVal, InnerChallenge> = builder.constant(val);
            builder.assert_ext_eq(actual, expected);
        }

        builder.halt();

        let program = builder.compile_isa::<1>();
        execute_program::<1, _>(program, stream);
    }
}
