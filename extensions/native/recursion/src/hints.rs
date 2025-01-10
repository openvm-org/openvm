use std::cmp::Reverse;

use itertools::Itertools;
use openvm_native_compiler::ir::{
    unsafe_array_transmute, Array, ArrayLike, Builder, Config, Ext, Felt, MemVariable, Usize, Var,
    DIGEST_SIZE,
};
use openvm_native_compiler_derive::compile_zip;
use openvm_stark_backend::{
    keygen::types::TraceWidth,
    p3_commit::ExtensionMmcs,
    p3_field::{extension::BinomialExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra},
    p3_util::log2_strict_usize,
    prover::{
        opener::{AdjacentOpenedValues, OpenedValues, OpeningProof},
        types::{AirProofData, Commitments, Proof},
    },
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    p3_baby_bear::{BabyBear, Poseidon2BabyBear},
};
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use crate::{
    types::{InnerConfig, VerifierInput},
    vars::{
        AdjacentOpenedValuesVariable, AirProofDataVariable, CommitmentsVariable,
        OpenedValuesVariable, OpeningProofVariable, StarkProofVariable, TraceWidthVariable,
        VerifierInputVariable,
    },
};

pub type InnerVal = BabyBear;
pub type InnerChallenge = BinomialExtensionField<InnerVal, 4>;
pub type InnerPerm = Poseidon2BabyBear<16>;
pub type InnerHash = PaddingFreeSponge<InnerPerm, 16, 8, 8>;
pub type InnerDigest = [InnerVal; DIGEST_SIZE];
pub type InnerCompress = TruncatedPermutation<InnerPerm, 2, 8, 16>;
pub type InnerValMmcs = MerkleTreeMmcs<
    <InnerVal as Field>::Packing,
    <InnerVal as Field>::Packing,
    InnerHash,
    InnerCompress,
    8,
>;
pub type InnerChallengeMmcs = ExtensionMmcs<InnerVal, InnerChallenge, InnerValMmcs>;
pub type InnerInputProof = Vec<InnerBatchOpening>;
pub type InnerQueryProof = QueryProof<InnerChallenge, InnerChallengeMmcs, InnerInputProof>;
pub type InnerCommitPhaseStep = CommitPhaseProofStep<InnerChallenge, InnerChallengeMmcs>;
pub type InnerFriProof = FriProof<InnerChallenge, InnerChallengeMmcs, InnerVal, InnerInputProof>;
pub type InnerBatchOpening = BatchOpening<InnerVal, InnerValMmcs>;

pub trait Hintable<C: Config> {
    type HintVariable: MemVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable;

    fn write(&self) -> Vec<Vec<C::N>>;

    fn witness(variable: &Self::HintVariable, builder: &mut Builder<C>) {
        let target = Self::read(builder);
        builder.assign(variable, target);
    }
}

impl<C: Config> Hintable<C> for usize {
    type HintVariable = Var<C::N>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        builder.hint_var()
    }

    fn write(&self) -> Vec<Vec<C::N>> {
        vec![vec![FieldAlgebra::from_canonical_usize(*self)]]
    }
}

// Assumes F = N
impl Hintable<InnerConfig> for InnerVal {
    type HintVariable = Felt<InnerVal>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_felt()
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        vec![vec![*self]]
    }
}

// Assumes F = N
impl Hintable<InnerConfig> for InnerChallenge {
    type HintVariable = Ext<InnerVal, InnerChallenge>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_ext()
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        vec![self.as_base_slice().to_vec()]
    }
}

/// Implement this on a type `T` that also implements `Hintable<C: Config>`
/// so that `Hintable<C>` is auto implemented on `Vec<T>`
pub trait VecAutoHintable {}

impl VecAutoHintable for Vec<usize> {}

impl VecAutoHintable for Vec<InnerVal> {}

impl VecAutoHintable for Vec<Vec<InnerChallenge>> {}

impl VecAutoHintable for AdjacentOpenedValues<InnerChallenge> {}

impl VecAutoHintable for Vec<AdjacentOpenedValues<InnerChallenge>> {}

impl VecAutoHintable for Vec<Vec<AdjacentOpenedValues<InnerChallenge>>> {}

impl VecAutoHintable for AirProofData<BabyBearPoseidon2Config> {}
impl VecAutoHintable for Proof<BabyBearPoseidon2Config> {}

impl<C: Config, I: VecAutoHintable + Hintable<C>> Hintable<C> for Vec<I> {
    type HintVariable = Array<C, I::HintVariable>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let len = builder.hint_var();
        let arr = builder.dyn_array(len);
        compile_zip!(builder, arr).for_each(|idx_vec, builder| {
            let hint = I::read(builder);
            let ptr = idx_vec[0];
            builder.iter_ptr_set(&arr, ptr, hint);
        });
        arr
    }

    fn write(&self) -> Vec<Vec<<C as Config>::N>> {
        let mut stream = Vec::new();

        let len = C::N::from_canonical_usize(self.len());
        stream.push(vec![len]);

        self.iter().for_each(|i| {
            let comm = I::write(i);
            stream.extend(comm);
        });

        stream
    }
}

impl Hintable<InnerConfig> for VerifierInput<BabyBearPoseidon2Config> {
    type HintVariable = VerifierInputVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let proof = Proof::<BabyBearPoseidon2Config>::read(builder);

        let raw_log_degree_per_air = Vec::<usize>::read(builder);
        // A hacky way to cast ptr.
        let log_degree_per_air = if let Array::Dyn(ptr, len) = raw_log_degree_per_air {
            Array::Dyn(ptr, len)
        } else {
            unreachable!();
        };

        let raw_air_perm_by_height = Vec::<usize>::read(builder);
        // A hacky way to transmute from Array of Var to Array of Usize.
        let air_perm_by_height = if let Array::Dyn(ptr, len) = raw_air_perm_by_height {
            Array::Dyn(ptr, len)
        } else {
            unreachable!();
        };

        VerifierInputVariable {
            proof,
            log_degree_per_air,
            air_perm_by_height,
        }
    }

    fn write(&self) -> Vec<Vec<InnerVal>> {
        let mut stream = Vec::new();
        // Explict enforce consistency when matrixs have the same height.
        let air_perm_by_height: Vec<_> = (0..self.log_degree_per_air.len())
            .sorted_by_key(|i| Reverse(self.log_degree_per_air[*i]))
            .collect();

        stream.extend(self.proof.write());
        stream.extend(self.log_degree_per_air.write());
        stream.extend(air_perm_by_height.write());

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

impl<C: Config> Hintable<C> for Vec<u8> {
    type HintVariable = Array<C, Var<C::N>>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        builder.hint_vars()
    }

    fn write(&self) -> Vec<Vec<C::N>> {
        vec![self
            .iter()
            .map(|x| FieldAlgebra::from_canonical_u8(*x))
            .collect()]
    }
}

impl Hintable<InnerConfig> for Vec<InnerVal> {
    type HintVariable = Array<InnerConfig, Felt<InnerVal>>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_felts()
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        vec![self.clone()]
    }
}

impl Hintable<InnerConfig> for Vec<InnerChallenge> {
    type HintVariable = Array<InnerConfig, Ext<InnerVal, InnerChallenge>>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        builder.hint_exts()
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
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
        let arr = builder.dyn_array(len);
        compile_zip!(builder, arr).for_each(|idx_vec, builder| {
            let hint = Vec::<InnerChallenge>::read(builder);
            builder.iter_ptr_set(&arr, idx_vec[0], hint);
        });
        arr
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();

        let len = InnerVal::from_canonical_usize(self.len());
        stream.push(vec![len]);

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
        let preprocessed = unsafe_array_transmute(Vec::<usize>::read(builder));
        let cached_mains = unsafe_array_transmute(Vec::<usize>::read(builder));
        let common_main = Usize::Var(usize::read(builder));
        let after_challenge = unsafe_array_transmute(Vec::<usize>::read(builder));

        TraceWidthVariable {
            preprocessed,
            cached_mains,
            common_main,
            after_challenge,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();

        stream.extend(self.preprocessed.into_iter().collect::<Vec<_>>().write());
        stream.extend(self.cached_mains.write());
        stream.extend(<usize as Hintable<InnerConfig>>::write(&self.common_main));
        stream.extend(self.after_challenge.write());

        stream
    }
}

impl Hintable<InnerConfig> for Proof<BabyBearPoseidon2Config> {
    type HintVariable = StarkProofVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let commitments = Commitments::<BabyBearPoseidon2Config>::read(builder);
        let opening = OpeningProof::<BabyBearPoseidon2Config>::read(builder);
        let per_air = Vec::<AirProofData<BabyBearPoseidon2Config>>::read(builder);
        let raw_air_perm_by_height = Vec::<usize>::read(builder);
        // A hacky way to transmute from Array of Var to Array of Usize.
        let air_perm_by_height = unsafe_array_transmute(raw_air_perm_by_height);

        StarkProofVariable {
            commitments,
            opening,
            per_air,
            air_perm_by_height,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();

        stream.extend(self.commitments.write());
        stream.extend(self.opening.write());
        stream.extend(<Vec<AirProofData<_>> as Hintable<_>>::write(&self.per_air));
        let air_perm_by_height: Vec<_> = (0..self.per_air.len())
            .sorted_by_key(|i| Reverse(self.per_air[*i].degree))
            .collect();
        stream.extend(air_perm_by_height.write());

        stream
    }
}

impl Hintable<InnerConfig> for AirProofData<BabyBearPoseidon2Config> {
    type HintVariable = AirProofDataVariable<InnerConfig>;
    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let air_id = Usize::Var(usize::read(builder));
        let log_degree = Usize::Var(usize::read(builder));
        let exposed_values_after_challenge = Vec::<Vec<InnerChallenge>>::read(builder);
        let public_values = Vec::<InnerVal>::read(builder);
        Self::HintVariable {
            air_id,
            log_degree,
            exposed_values_after_challenge,
            public_values,
        }
    }
    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();

        stream.extend(<usize as Hintable<InnerConfig>>::write(&self.air_id));
        stream.extend(<usize as Hintable<InnerConfig>>::write(&log2_strict_usize(
            self.degree,
        )));
        stream.extend(self.exposed_values_after_challenge.write());
        stream.extend(self.public_values.write());

        stream
    }
}

impl Hintable<InnerConfig> for OpeningProof<BabyBearPoseidon2Config> {
    type HintVariable = OpeningProofVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let proof = InnerFriProof::read(builder);
        let values = OpenedValues::read(builder);

        OpeningProofVariable { proof, values }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
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

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
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

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();
        stream.extend(self.local.write());
        stream.extend(self.next.write());
        stream
    }
}

impl Hintable<InnerConfig> for Commitments<BabyBearPoseidon2Config> {
    type HintVariable = CommitmentsVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let main_trace = Vec::<InnerDigest>::read(builder);
        let after_challenge = Vec::<InnerDigest>::read(builder);
        let quotient = InnerDigest::read(builder);

        CommitmentsVariable {
            main_trace,
            after_challenge,
            quotient,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
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
    use openvm_native_circuit::execute_program;
    use openvm_native_compiler::{
        asm::AsmBuilder,
        ir::{Ext, Felt, Var},
    };
    use openvm_stark_backend::p3_field::FieldAlgebra;

    use crate::hints::{Hintable, InnerChallenge, InnerVal};

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

        let program = builder.compile_isa();
        execute_program(program, stream);
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

        let program = builder.compile_isa();
        execute_program(program, stream);
    }
}
