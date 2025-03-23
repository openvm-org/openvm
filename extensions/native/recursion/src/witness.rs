use core::borrow::Borrow;

use openvm_native_compiler::ir::{
    Array, Builder, Config, Ext, Felt, MemVariable, Usize, Var, Witness,
};
use openvm_stark_backend::{
    config::{Com, PcsProof},
    p3_util::log2_strict_usize,
    proof::{AdjacentOpenedValues, AirProofData, Commitments, OpenedValues, OpeningProof, Proof},
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2_root::BabyBearPoseidon2RootConfig, p3_baby_bear::BabyBear,
    p3_bn254_fr::Bn254Fr,
};
use p3_symmetric::Hash;

use crate::{
    config::outer::{OuterChallenge, OuterConfig, OuterVal},
    digest::{DigestVal, DigestVariable},
    hints::{InnerChallenge, InnerVal},
    vars::{
        AdjacentOpenedValuesVariable, AirProofDataVariable, CommitmentsVariable,
        OpenedValuesVariable, OpeningProofVariable, StarkProofVariable,
    },
};

pub trait Witnessable<C: Config> {
    type WitnessVariable: MemVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable;

    fn write(&self, witness: &mut Witness<C>);
}

type C = OuterConfig;
type OuterCom = Hash<BabyBear, Bn254Fr, 1>;

impl Witnessable<C> for Bn254Fr {
    type WitnessVariable = Var<Bn254Fr>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        builder.witness_var()
    }

    fn write(&self, witness: &mut Witness<C>) {
        witness.vars.push(*self);
    }
}

impl Witnessable<C> for OuterVal {
    type WitnessVariable = Felt<OuterVal>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        builder.witness_felt()
    }

    fn write(&self, witness: &mut Witness<C>) {
        witness.felts.push(*self);
    }
}

impl Witnessable<C> for OuterChallenge {
    type WitnessVariable = Ext<OuterVal, OuterChallenge>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        builder.witness_ext()
    }

    fn write(&self, witness: &mut Witness<C>) {
        witness.exts.push(*self);
    }
}

impl Witnessable<C> for OuterCom {
    type WitnessVariable = DigestVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let bv: &[Bn254Fr; 1] = self.borrow();
        let v = vec![bv[0].read(builder)];
        DigestVariable::Var(builder.vec(v))
    }

    fn write(&self, witness: &mut Witness<C>) {
        let bv: &[Bn254Fr; 1] = self.borrow();
        witness.vars.push(bv[0]);
    }
}

// In static mode, usize is hardcoded.
impl Witnessable<C> for usize {
    type WitnessVariable = Usize<<C as Config>::N>;

    fn read(&self, _builder: &mut Builder<C>) -> Self::WitnessVariable {
        Usize::from(*self)
    }

    fn write(&self, _witness: &mut Witness<C>) {
        // Do nothing
    }
}

pub trait VectorWitnessable<C: Config>: Witnessable<C> {}
impl VectorWitnessable<C> for Bn254Fr {}
impl VectorWitnessable<C> for OuterVal {}
impl VectorWitnessable<C> for OuterChallenge {}
impl VectorWitnessable<C> for OuterCom {}
impl VectorWitnessable<C> for usize {}
impl VectorWitnessable<C> for Vec<OuterChallenge> {}
impl VectorWitnessable<C> for Vec<Vec<OuterChallenge>> {}
impl VectorWitnessable<C> for Vec<OuterVal> {}
impl VectorWitnessable<C> for Vec<Vec<OuterVal>> {}

impl<I: VectorWitnessable<C>> Witnessable<C> for Vec<I> {
    type WitnessVariable = Array<C, I::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let raw_vec: Vec<I::WitnessVariable> = self.iter().map(|x| x.read(builder)).collect();
        builder.vec(raw_vec)
    }

    fn write(&self, witness: &mut Witness<C>) {
        self.iter().for_each(|x| x.write(witness));
    }
}

impl Witnessable<C> for DigestVal<C> {
    type WitnessVariable = DigestVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let result = vec![builder.witness_var()];
        DigestVariable::Var(builder.vec(result))
    }

    fn write(&self, witness: &mut Witness<C>) {
        if let DigestVal::N(v) = self {
            assert_eq!(v.len(), 1);
            witness.vars.push(v[0]);
        } else {
            panic!("DigestVal should use N in static mode")
        }
    }
}
impl VectorWitnessable<C> for DigestVal<C> {}

impl VectorWitnessable<C> for AirProofData<InnerVal, InnerChallenge> {}

impl Witnessable<C> for Proof<BabyBearPoseidon2RootConfig> {
    type WitnessVariable = StarkProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        if builder.flags.static_only {
            // Check that the trace heights are sorted
            assert!(
                self.per_air.windows(2).all(|w| w[0].degree >= w[1].degree),
                "Static verifier requires trace heights to be sorted descending"
            );
        }
        let commitments = self.commitments.read(builder);
        let opening = self.opening.read(builder);
        let per_air = self.per_air.read(builder);
        // This reads nothing because air_perm_by_height is a constant.
        let air_perm_by_height = builder.array(0);
        let log_up_pow_witness = self
            .rap_phase_seq_proof
            .as_ref()
            .map(|proof| proof.logup_pow_witness)
            .unwrap_or_default()
            .read(builder);

        StarkProofVariable {
            commitments,
            opening,
            per_air,
            air_perm_by_height,
            log_up_pow_witness,
        }
    }

    fn write(&self, witness: &mut Witness<C>) {
        self.commitments.write(witness);
        self.opening.write(witness);
        self.per_air.write(witness);
        // air_perm_by_height is a constant so we write nothing.
        let logup_pow_witness = self
            .rap_phase_seq_proof
            .as_ref()
            .map(|p| p.logup_pow_witness)
            .unwrap_or_default();
        logup_pow_witness.write(witness);
    }
}

impl Witnessable<C> for AirProofData<OuterVal, OuterChallenge> {
    type WitnessVariable = AirProofDataVariable<C>;
    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        // air_id is constant, skip
        let air_id = Usize::from(0);
        // log_degree is constant, skip
        let log_degree = Usize::from(log2_strict_usize(self.degree));
        let exposed_values_after_challenge = self.exposed_values_after_challenge.read(builder);
        let public_values = self.public_values.read(builder);
        Self::WitnessVariable {
            air_id,
            log_degree,
            exposed_values_after_challenge,
            public_values,
        }
    }
    fn write(&self, witness: &mut Witness<C>) {
        // air_id is constant, skip
        // log_degree is constant, skip
        <_ as Witnessable<_>>::write(&self.exposed_values_after_challenge, witness);
        <_ as Witnessable<_>>::write(&self.public_values, witness);
    }
}

impl Witnessable<OuterConfig> for Commitments<Com<BabyBearPoseidon2RootConfig>> {
    type WitnessVariable = CommitmentsVariable<OuterConfig>;

    fn read(&self, builder: &mut Builder<OuterConfig>) -> Self::WitnessVariable {
        let after_challenge = self.after_challenge.read(builder);
        let main_trace = self.main_trace.read(builder);
        let quotient = self.quotient.read(builder);
        Self::WitnessVariable {
            after_challenge,
            main_trace,
            quotient,
        }
    }

    fn write(&self, witness: &mut Witness<OuterConfig>) {
        self.after_challenge.write(witness);
        self.main_trace.write(witness);
        self.quotient.write(witness);
    }
}

impl Witnessable<C> for OpeningProof<PcsProof<BabyBearPoseidon2RootConfig>, OuterChallenge> {
    type WitnessVariable = OpeningProofVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let proof = self.proof.read(builder);
        let values = self.values.read(builder);
        OpeningProofVariable { proof, values }
    }

    fn write(&self, witness: &mut Witness<C>) {
        self.proof.write(witness);
        <_ as Witnessable<C>>::write(&self.values, witness);
    }
}

impl Witnessable<C> for OpenedValues<OuterChallenge> {
    type WitnessVariable = OpenedValuesVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let preprocessed = self.preprocessed.read(builder);
        let main = self.main.read(builder);
        let quotient = self.quotient.read(builder);
        let after_challenge = self.after_challenge.read(builder);

        OpenedValuesVariable {
            preprocessed,
            main,
            quotient,
            after_challenge,
        }
    }

    fn write(&self, witness: &mut Witness<C>) {
        <Vec<_> as Witnessable<C>>::write(&self.preprocessed, witness);
        <Vec<_> as Witnessable<C>>::write(&self.main, witness);
        <Vec<_> as Witnessable<C>>::write(&self.quotient, witness);
        <Vec<_> as Witnessable<C>>::write(&self.after_challenge, witness);
    }
}

impl Witnessable<C> for AdjacentOpenedValues<OuterChallenge> {
    type WitnessVariable = AdjacentOpenedValuesVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let local = self.local.read(builder);
        let next = self.next.read(builder);
        AdjacentOpenedValuesVariable { local, next }
    }

    fn write(&self, witness: &mut Witness<C>) {
        <Vec<_> as Witnessable<C>>::write(&self.local, witness);
        <Vec<_> as Witnessable<C>>::write(&self.next, witness);
    }
}
impl VectorWitnessable<C> for AdjacentOpenedValues<OuterChallenge> {}
impl VectorWitnessable<C> for Vec<AdjacentOpenedValues<OuterChallenge>> {}
