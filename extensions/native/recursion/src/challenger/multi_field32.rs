use openvm_native_compiler::ir::{Array, Builder, Config, Ext, Felt, RVar, Var};
use openvm_stark_backend::p3_field::{
    max_absorb_injective_limbs, squeeze_field_order_num_limbs, PrimeCharacteristicRing,
};

use crate::{
    challenger::{
        CanCheckWitness, CanObserveDigest, CanObserveVariable, CanSampleBitsVariable,
        CanSampleVariable, ChallengerVariable, FeltChallenger,
    },
    digest::DigestVariable,
    outer_poseidon2::{Poseidon2CircuitBuilder, RATE, SPONGE_SIZE},
    utils::{absorb_radix_bits_for_config, reduce_packed, split_pf_to_field_order_limbs},
    vars::OuterDigestVariable,
};

/// In-circuit port of `p3_challenger::MultiField32Challenger`. p3's nested `DuplexChallenger` is
/// flattened: `sponge_state` and `inner_output_buffer` correspond to `inner.sponge_state` and
/// `inner.output_buffer`. There is no analogue of `inner.input_buffer` because the wrapper never
/// dispatches into a `DuplexChallenger::observe` path.
#[derive(Clone)]
pub struct MultiField32ChallengerVariable<C: Config> {
    sponge_state: [Var<C::N>; SPONGE_SIZE],
    inner_output_buffer: Vec<Var<C::N>>,
    f_buffer: Vec<Felt<C::F>>,
    f_squeeze_buffer: Vec<Felt<C::F>>,
    absorb_num_f_elms: usize,
    squeeze_num_f_elms: usize,
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
            inner_output_buffer: vec![],
            f_buffer: vec![],
            f_squeeze_buffer: vec![],
            absorb_num_f_elms: max_absorb_injective_limbs::<C::F, C::N>(),
            squeeze_num_f_elms: squeeze_field_order_num_limbs::<C::N, C::F>(),
        }
    }

    /// Mirrors `DuplexChallenger::absorb_rate_padded_with_tag`.
    fn absorb_rate_padded_with_tag(
        &mut self,
        builder: &mut Builder<C>,
        values: &[Var<C::N>],
        length_tag: u8,
    ) {
        assert!(values.len() <= RATE);
        self.inner_output_buffer.clear();

        // Each state slot must be a *fresh* Var. `p2_permute_mut`'s halo2 lowering rebinds
        // the value behind each state Var to the post-permutation value, so aliasing the
        // caller's Vars (e.g. the digest words) here would corrupt them for subsequent uses.
        for (i, &value) in values.iter().enumerate() {
            self.sponge_state[i] = builder.eval(value);
        }
        for i in values.len()..RATE {
            self.sponge_state[i] = builder.eval(C::N::ZERO);
        }
        self.sponge_state[RATE] = builder.eval(self.sponge_state[RATE] + C::N::from_u8(length_tag));

        builder.p2_permute_mut(self.sponge_state);
        self.inner_output_buffer
            .extend_from_slice(&self.sponge_state[..RATE]);
    }

    /// Mirrors `DuplexChallenger::duplexing` (with the always-empty `inner.input_buffer`).
    fn duplexing(&mut self, builder: &mut Builder<C>) {
        builder.p2_permute_mut(self.sponge_state);
        self.inner_output_buffer.clear();
        self.inner_output_buffer
            .extend_from_slice(&self.sponge_state[..RATE]);
    }

    fn flush_f_if_non_empty(&mut self, builder: &mut Builder<C>) {
        if self.f_buffer.is_empty() {
            return;
        }
        let n_in = self.f_buffer.len();
        assert!(n_in <= self.absorb_num_f_elms * RATE);
        let radix_bits = absorb_radix_bits_for_config::<C>();
        let packed = self
            .f_buffer
            .chunks(self.absorb_num_f_elms)
            .map(|f_chunk| reduce_packed(builder, f_chunk, radix_bits))
            .collect::<Vec<_>>();
        self.absorb_rate_padded_with_tag(builder, &packed, n_in as u8);
        self.f_buffer.clear();
        self.f_squeeze_buffer.clear();
    }

    fn refill_f_squeeze_from_inner(&mut self, builder: &mut Builder<C>) {
        self.f_squeeze_buffer.clear();
        for &pf_val in &self.inner_output_buffer {
            let f_vals = split_pf_to_field_order_limbs(builder, pf_val, self.squeeze_num_f_elms);
            self.f_squeeze_buffer.extend(f_vals);
        }
        // Match `DuplexChallenger` semantics: squeezing consumes the current rate row.
        self.inner_output_buffer.clear();
    }

    pub fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        self.inner_output_buffer.clear();
        self.f_squeeze_buffer.clear();
        self.f_buffer.push(value);
        if self.f_buffer.len() == self.absorb_num_f_elms * RATE {
            self.flush_f_if_non_empty(builder);
        }
    }

    pub fn observe_commitment(&mut self, builder: &mut Builder<C>, value: OuterDigestVariable<C>) {
        self.inner_output_buffer.clear();
        self.f_squeeze_buffer.clear();
        self.flush_f_if_non_empty(builder);

        for chunk in value.chunks(RATE) {
            self.absorb_rate_padded_with_tag(builder, chunk, chunk.len() as u8);
            self.f_squeeze_buffer.clear();
        }
    }

    pub fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        self.flush_f_if_non_empty(builder);
        if self.f_squeeze_buffer.is_empty() {
            // p3 also guards on `!inner.input_buffer.is_empty()`; that disjunct is vacuous here
            // because no path on this wrapper writes into the inner `DuplexChallenger::observe`.
            if self.inner_output_buffer.is_empty() {
                self.duplexing(builder);
            }
            self.refill_f_squeeze_from_inner(builder);
        }
        self.f_squeeze_buffer
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
        if bits == 0 {
            return;
        }
        self.observe(builder, witness);
        let element = self.sample_bits(builder, bits);
        builder.assert_var_eq(element, C::N::from_usize(0));
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
