use ax_stark_backend::p3_field::{AbstractField, Field};
use axvm_native_compiler::ir::{Array, Builder, Config, Ext, Felt, RVar, Var};

use crate::{
    challenger::{
        CanCheckWitness, CanObserveDigest, CanObserveVariable, CanSampleBitsVariable,
        CanSampleVariable, ChallengerVariable, FeltChallenger,
    },
    digest::DigestVariable,
    outer_poseidon2::{Poseidon2CircuitBuilder, RATE, SPONGE_SIZE},
    utils::{reduce_32, split_32},
    vars::OuterDigestVariable,
};

#[derive(Clone)]
pub struct MultiField32ChallengerVariable<C: Config> {
    sponge_state: [Var<C::N>; SPONGE_SIZE],
    input_buffer: Vec<Felt<C::F>>,
    output_buffer: Vec<Felt<C::F>>,
    num_f_elms: usize,
}

impl<C: Config> MultiField32ChallengerVariable<C> {
    pub fn new(builder: &mut Builder<C>) -> Self {
        assert!(builder.flags.static_only);
        MultiField32ChallengerVariable::<C> {
            sponge_state: [
                builder.eval(C::N::ZERO),
                builder.eval(C::N::ZERO),
                builder.eval(C::N::ZERO),
            ],
            input_buffer: vec![],
            output_buffer: vec![],
            num_f_elms: C::N::bits() / 64,
        }
    }

    pub fn duplexing(&mut self, builder: &mut Builder<C>) {
        assert!(self.input_buffer.len() <= self.num_f_elms * RATE);

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
        if self.input_buffer.len() == self.num_f_elms * RATE {
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

impl<C: Config> ChallengerVariable<C> for MultiField32ChallengerVariable<C> {
    fn new(builder: &mut Builder<C>) -> Self {
        MultiField32ChallengerVariable::new(builder)
    }
}
// Testing depends on halo2. Put it inside src/halo2/tests/multi_field32.rs
