use afs_compiler::ir::{
    Array, BigUintVar, Builder, Config, Ext, Felt, MemVariable, Var, DIGEST_SIZE,
};
use afs_ecc::types::{
    ECDSAInput, ECDSAInputVariable, ECDSASignature, ECDSASignatureVariable, ECPoint,
    ECPointVariable,
};
use afs_stark_backend::{
    keygen::types::TraceWidth,
    prover::{
        opener::{AdjacentOpenedValues, OpenedValues, OpeningProof},
        types::{Commitments, Proof},
    },
};
use ax_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use num_bigint_dig::BigUint;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_commit::ExtensionMmcs;
use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField, Field};
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof, TwoAdicFriPcsProof};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use stark_vm::modular_arithmetic::{big_uint_to_num_limbs, LIMB_SIZE, NUM_LIMBS};

use crate::types::{
    AdjacentOpenedValuesVariable, CommitmentsVariable, InnerConfig, OpenedValuesVariable,
    OpeningProofVariable, StarkProofVariable, TraceWidthVariable, VerifierInput,
    VerifierInputVariable,
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
        vec![vec![AbstractField::from_canonical_usize(*self)]]
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

impl<C: Config, I: VecAutoHintable + Hintable<C>> Hintable<C> for Vec<I> {
    type HintVariable = Array<C, I::HintVariable>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let len = builder.hint_var();
        let arr = builder.dyn_array(len);
        builder.range(0, len).for_each(|i, builder| {
            let hint = I::read(builder);
            builder.set(&arr, i, hint);
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
        let public_values = Vec::<Vec<InnerVal>>::read(builder);

        VerifierInputVariable {
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

impl<C: Config> Hintable<C> for Vec<u8> {
    type HintVariable = Array<C, Var<C::N>>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        builder.hint_vars()
    }

    fn write(&self) -> Vec<Vec<C::N>> {
        vec![self
            .iter()
            .map(|x| AbstractField::from_canonical_u8(*x))
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
        builder.range(0, len).for_each(|i, builder| {
            let hint = Vec::<InnerChallenge>::read(builder);
            builder.set(&arr, i, hint);
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
        let preprocessed = Vec::<usize>::read(builder);
        let partitioned_main = Vec::<usize>::read(builder);
        let after_challenge = Vec::<usize>::read(builder);

        TraceWidthVariable {
            preprocessed,
            partitioned_main,
            after_challenge,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();

        stream.extend(self.preprocessed.into_iter().collect::<Vec<_>>().write());
        stream.extend(self.partitioned_main.write());
        stream.extend(self.after_challenge.write());

        stream
    }
}

impl Hintable<InnerConfig> for Proof<BabyBearPoseidon2Config> {
    type HintVariable = StarkProofVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let commitments = Commitments::<BabyBearPoseidon2Config>::read(builder);
        let opening = OpeningProof::<BabyBearPoseidon2Config>::read(builder);
        let exposed_values_after_challenge = Vec::<Vec<Vec<InnerChallenge>>>::read(builder);

        StarkProofVariable {
            commitments,
            opening,
            exposed_values_after_challenge,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();

        stream.extend(self.commitments.write());
        stream.extend(self.opening.write());
        stream.extend(self.exposed_values_after_challenge.write());

        stream
    }
}

impl Hintable<InnerConfig> for OpeningProof<BabyBearPoseidon2Config> {
    type HintVariable = OpeningProofVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let proof = InnerPcsProof::read(builder);
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

impl Hintable<InnerConfig> for BigUint {
    type HintVariable = BigUintVar<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let ret = builder.uninit_biguint();
        for i in 0..NUM_LIMBS {
            // FIXME: range check for each element.
            let v = builder.hint_var();
            builder.set_value(&ret, i, v);
        }
        ret
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        vec![big_uint_to_num_limbs(self, LIMB_SIZE, NUM_LIMBS)
            .iter()
            .map(|x| <InnerConfig as Config>::N::from_canonical_usize(*x))
            .collect()]
    }
}

impl Hintable<InnerConfig> for ECPoint {
    type HintVariable = ECPointVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let x = BigUint::read(builder);
        let y = BigUint::read(builder);
        // ECPointVariable::`new` checks if the point is on the curve.
        ECPointVariable { x, y }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut ret: Vec<Vec<<InnerConfig as Config>::N>> = self.x.write();
        ret.extend(self.y.write());
        ret
    }
}

impl Hintable<InnerConfig> for ECDSASignature {
    type HintVariable = ECDSASignatureVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let r = BigUint::read(builder);
        let s = BigUint::read(builder);
        ECDSASignatureVariable { r, s }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut ret: Vec<Vec<<InnerConfig as Config>::N>> = self.r.write();
        ret.extend(self.s.write());
        ret
    }
}

impl Hintable<InnerConfig> for ECDSAInput {
    type HintVariable = ECDSAInputVariable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let pubkey = ECPoint::read(builder);
        let sig = ECDSASignature::read(builder);
        let msg_hash = BigUint::read(builder);
        Self::HintVariable {
            pubkey,
            sig,
            msg_hash,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut ret = self.pubkey.write();
        ret.extend(self.sig.write());
        ret.extend(self.msg_hash.write());
        ret
    }
}

#[cfg(test)]
mod test {
    use afs_compiler::{
        asm::AsmBuilder,
        ir::{Ext, Felt, Var},
        prelude::*,
        util::execute_program,
    };
    use afs_derive::{DslVariable, Hintable};
    use p3_field::AbstractField;

    use crate::{
        hints::{Hintable, InnerChallenge, InnerVal},
        types::InnerConfig,
    };

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

    #[derive(Hintable)]
    struct TestStruct {
        a: usize,
        b: usize,
        c: usize,
    }

    #[test]
    fn test_macro() {
        let x = TestStruct { a: 1, b: 2, c: 3 };
        let stream = Hintable::<InnerConfig>::write(&x);
        assert_eq!(
            stream,
            [1, 2, 3]
                .map(|x| vec![InnerVal::from_canonical_usize(x)])
                .to_vec()
        );
    }
}
