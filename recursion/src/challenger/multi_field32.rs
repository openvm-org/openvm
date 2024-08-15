use afs_compiler::ir::{Array, Builder, Config, Ext, Felt, RVar, Var};
use p3_field::{AbstractField, Field};

use crate::{
    challenger::{
        CanCheckWitness, CanObserveDigest, CanObserveVariable, CanSampleBitsVariable,
        CanSampleVariable, ChallengerVariable, FeltChallenger,
    },
    digest::DigestVariable,
    outer_poseidon2::{Poseidon2CircuitBuilder, SPONGE_SIZE},
    types::OuterDigestVariable,
    utils::{reduce_32, split_32},
};

#[derive(Clone)]
pub struct MultiField32ChallengerVariable<C: Config> {
    sponge_state: [Var<C::N>; SPONGE_SIZE],
    input_buffer: Vec<Felt<C::F>>,
    output_buffer: Vec<Felt<C::F>>,
    num_f_elms: usize,
}

impl<C: Config> MultiField32ChallengerVariable<C> {
    #[allow(dead_code)]
    pub fn new(builder: &mut Builder<C>) -> Self {
        assert!(builder.flags.static_only);
        MultiField32ChallengerVariable::<C> {
            sponge_state: [
                builder.eval(C::N::zero()),
                builder.eval(C::N::zero()),
                builder.eval(C::N::zero()),
            ],
            input_buffer: vec![],
            output_buffer: vec![],
            num_f_elms: C::N::bits() / 64,
        }
    }

    pub fn duplexing(&mut self, builder: &mut Builder<C>) {
        assert!(self.input_buffer.len() <= self.num_f_elms * SPONGE_SIZE);

        for (i, f_chunk) in self.input_buffer.chunks(self.num_f_elms).enumerate() {
            self.sponge_state[i] = reduce_32(builder, f_chunk);
        }
        self.input_buffer.clear();

        builder.p2_permute_mut(self.sponge_state);

        self.output_buffer.clear();
        for &pf_val in self.sponge_state.iter() {
            let f_vals = split_32(builder, pf_val, self.num_f_elms);
            for f_val in f_vals {
                self.output_buffer.push(f_val);
            }
        }
    }

    pub fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        self.output_buffer.clear();

        self.input_buffer.push(value);
        if self.input_buffer.len() == self.num_f_elms * SPONGE_SIZE {
            self.duplexing(builder);
        }
    }

    pub fn observe_commitment(&mut self, builder: &mut Builder<C>, value: OuterDigestVariable<C>) {
        value.into_iter().for_each(|v| {
            let f_vals: Vec<Felt<C::F>> = split_32(builder, v, self.num_f_elms);
            for f_val in f_vals {
                self.observe(builder, f_val);
            }
        });
    }

    pub fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing(builder);
        }

        self.output_buffer
            .pop()
            .expect("output buffer should be non-empty")
    }

    pub fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        let a = self.sample(builder);
        let b = self.sample(builder);
        let c = self.sample(builder);
        let d = self.sample(builder);
        builder.felts2ext(&[a, b, c, d])
    }

    pub fn sample_bits(&mut self, builder: &mut Builder<C>, bits: usize) -> Var<C::N> {
        let rand_f = self.sample(builder);
        let rand_f_bits = builder.num2bits_f_circuit(rand_f);
        builder.bits2num_v_circuit(&rand_f_bits[0..bits])
    }

    pub fn check_witness(&mut self, builder: &mut Builder<C>, bits: usize, witness: Felt<C::F>) {
        self.observe(builder, witness);
        let element = self.sample_bits(builder, bits);
        builder.assert_var_eq(element, C::N::from_canonical_usize(0));
    }
}

impl<C: Config> CanObserveVariable<C, Felt<C::F>> for MultiField32ChallengerVariable<C> {
    fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        MultiField32ChallengerVariable::observe(self, builder, value);
    }

    fn observe_slice(&mut self, builder: &mut Builder<C>, values: Array<C, Felt<C::F>>) {
        values.vec().into_iter().for_each(|value| {
            self.observe(builder, value);
        });
    }
}

impl<C: Config> CanSampleVariable<C, Felt<C::F>> for MultiField32ChallengerVariable<C> {
    fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        MultiField32ChallengerVariable::sample(self, builder)
    }
}

impl<C: Config> CanSampleBitsVariable<C> for MultiField32ChallengerVariable<C> {
    fn sample_bits(
        &mut self,
        builder: &mut Builder<C>,
        nb_bits: RVar<C::N>,
    ) -> Array<C, Var<C::N>> {
        let rand_f = self.sample(builder);
        let rand_f_bits = builder.num2bits_f_circuit(rand_f);
        builder.vec(rand_f_bits[..nb_bits.value()].to_vec())
    }
}

impl<C: Config> CanObserveDigest<C> for MultiField32ChallengerVariable<C> {
    fn observe_digest(&mut self, builder: &mut Builder<C>, commitment: DigestVariable<C>) {
        if let DigestVariable::Var(v_commit) = commitment {
            MultiField32ChallengerVariable::observe_commitment(
                self,
                builder,
                v_commit.vec().try_into().unwrap(),
            );
        } else {
            panic!("MultiField32ChallengerVariable expects Var commitment");
        }
    }
}

impl<C: Config> FeltChallenger<C> for MultiField32ChallengerVariable<C> {
    fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        MultiField32ChallengerVariable::sample_ext(self, builder)
    }
}

impl<C: Config> CanCheckWitness<C> for MultiField32ChallengerVariable<C> {
    fn check_witness(&mut self, builder: &mut Builder<C>, nb_bits: usize, witness: Felt<C::F>) {
        MultiField32ChallengerVariable::check_witness(self, builder, nb_bits, witness);
    }
}

impl<C: Config> ChallengerVariable<C> for MultiField32ChallengerVariable<C> {}

#[cfg(test)]
mod tests {
    use afs_compiler::ir::{Builder, SymbolicExt, Witness};
    use p3_baby_bear::BabyBear;
    use p3_bn254_fr::Bn254Fr;
    use p3_challenger::{CanObserve, CanSample, FieldChallenger};
    use p3_field::{extension::BinomialExtensionField, AbstractField};
    use p3_symmetric::Hash;

    use crate::{
        challenger::multi_field32::MultiField32ChallengerVariable,
        config::outer::{outer_perm, OuterChallenger, OuterConfig},
        halo2::Halo2Prover,
        OUTER_DIGEST_SIZE,
    };

    #[test]
    fn test_challenger() {
        let perm = outer_perm();
        let mut challenger = OuterChallenger::new(perm).unwrap();
        let a = BabyBear::from_canonical_usize(1);
        let b = BabyBear::from_canonical_usize(2);
        let c = BabyBear::from_canonical_usize(3);
        challenger.observe(a);
        challenger.observe(b);
        challenger.observe(c);
        let gt1: BabyBear = challenger.sample();
        challenger.observe(a);
        challenger.observe(b);
        challenger.observe(c);
        let gt2: BabyBear = challenger.sample();
        let gt3: BabyBear = challenger.sample();

        let mut builder = Builder::<OuterConfig>::default();
        builder.flags.static_only = true;
        let mut challenger = MultiField32ChallengerVariable::new(&mut builder);
        let a = builder.eval(a);
        let b = builder.eval(b);
        let c = builder.eval(c);
        challenger.observe(&mut builder, a);
        challenger.observe(&mut builder, b);
        challenger.observe(&mut builder, c);
        let result1 = challenger.sample(&mut builder);
        builder.assert_felt_eq(gt1, result1);
        challenger.observe(&mut builder, a);
        challenger.observe(&mut builder, b);
        challenger.observe(&mut builder, c);
        let result2 = challenger.sample(&mut builder);
        builder.assert_felt_eq(gt2, result2);
        let result3 = challenger.sample(&mut builder);
        builder.assert_felt_eq(gt3, result3);

        Halo2Prover::mock::<OuterConfig>(10, builder.operations, Witness::default());
    }

    #[test]
    fn test_challenger_sample_ext() {
        let perm = outer_perm();
        let mut challenger = OuterChallenger::new(perm).unwrap();
        let a = BabyBear::from_canonical_usize(1);
        let b = BabyBear::from_canonical_usize(2);
        let c = BabyBear::from_canonical_usize(3);
        let hash = Hash::from([Bn254Fr::two(); OUTER_DIGEST_SIZE]);
        challenger.observe(hash);
        challenger.observe(a);
        challenger.observe(b);
        challenger.observe(c);
        let gt1: BinomialExtensionField<BabyBear, 4> = challenger.sample_ext_element();
        challenger.observe(a);
        challenger.observe(b);
        challenger.observe(c);
        let gt2: BinomialExtensionField<BabyBear, 4> = challenger.sample_ext_element();

        let mut builder = Builder::<OuterConfig>::default();
        builder.flags.static_only = true;
        let mut challenger = MultiField32ChallengerVariable::new(&mut builder);
        let a = builder.eval(a);
        let b = builder.eval(b);
        let c = builder.eval(c);
        let hash = builder.eval(Bn254Fr::two());
        challenger.observe_commitment(&mut builder, [hash]);
        challenger.observe(&mut builder, a);
        challenger.observe(&mut builder, b);
        challenger.observe(&mut builder, c);
        let result1 = challenger.sample_ext(&mut builder);
        challenger.observe(&mut builder, a);
        challenger.observe(&mut builder, b);
        challenger.observe(&mut builder, c);
        let result2 = challenger.sample_ext(&mut builder);

        builder.assert_ext_eq(SymbolicExt::from_f(gt1), result1);
        builder.assert_ext_eq(SymbolicExt::from_f(gt2), result2);

        Halo2Prover::mock::<OuterConfig>(10, builder.operations, Witness::default());
    }
}
