use axvm_native_compiler::{
    ir::{RVar, DIGEST_SIZE, PERMUTATION_WIDTH},
    prelude::{Array, Builder, Config, Ext, Felt, Var},
};
use p3_field::{AbstractField, Field};

use crate::{
    challenger::{
        CanCheckWitness, CanObserveDigest, CanObserveVariable, CanSampleBitsVariable,
        CanSampleVariable, ChallengerVariable, FeltChallenger,
    },
    digest::DigestVariable,
};

/// Reference: [p3_challenger::DuplexChallenger]
#[derive(Clone)]
pub struct DuplexChallengerVariable<C: Config> {
    pub sponge_state: Array<C, Felt<C::F>>,
    pub nb_inputs: Var<C::N>,
    pub nb_outputs: Var<C::N>,
}

impl<C: Config> DuplexChallengerVariable<C> {
    /// Creates a new duplex challenger with the default state.
    pub fn new(builder: &mut Builder<C>) -> Self {
        let sponge_state = builder.dyn_array(PERMUTATION_WIDTH);

        builder.range(0, sponge_state.len()).for_each(|i, builder| {
            builder.set(&sponge_state, i, C::F::ZERO);
        });

        DuplexChallengerVariable::<C> {
            sponge_state,
            nb_inputs: builder.eval(C::N::ZERO),
            nb_outputs: builder.eval(C::N::ZERO),
        }
    }
    /// Creates a new challenger with the same state as an existing challenger.
    #[allow(dead_code)]
    pub fn copy(&self, builder: &mut Builder<C>) -> Self {
        let sponge_state = builder.dyn_array(PERMUTATION_WIDTH);
        builder.range(0, PERMUTATION_WIDTH).for_each(|i, builder| {
            let element = builder.get(&self.sponge_state, i);
            builder.set(&sponge_state, i, element);
        });
        let nb_inputs = builder.eval(self.nb_inputs);
        let nb_outputs = builder.eval(self.nb_outputs);
        DuplexChallengerVariable::<C> {
            sponge_state,
            nb_inputs,
            nb_outputs,
        }
    }

    /// Asserts that the state of this challenger is equal to the state of another challenger.
    #[allow(dead_code)]
    pub fn assert_eq(&self, builder: &mut Builder<C>, other: &Self) {
        builder.assert_var_eq(self.nb_inputs, other.nb_inputs);
        builder.assert_var_eq(self.nb_outputs, other.nb_outputs);
        builder.range(0, PERMUTATION_WIDTH).for_each(|i, builder| {
            let element = builder.get(&self.sponge_state, i);
            let other_element = builder.get(&other.sponge_state, i);
            builder.assert_felt_eq(element, other_element);
        });
    }

    #[allow(dead_code)]
    pub fn reset(&mut self, builder: &mut Builder<C>) {
        let zero: Var<_> = builder.eval(C::N::ZERO);
        let zero_felt: Felt<_> = builder.eval(C::F::ZERO);
        builder.range(0, PERMUTATION_WIDTH).for_each(|i, builder| {
            builder.set(&self.sponge_state, i, zero_felt);
        });
        builder.assign(&self.nb_inputs, zero);
        builder.assign(&self.nb_outputs, zero);
    }

    pub fn duplexing(&mut self, builder: &mut Builder<C>) {
        builder.assign(&self.nb_inputs, C::N::ZERO);

        builder.poseidon2_permute_mut(&self.sponge_state);

        builder.assign(
            &self.nb_outputs,
            C::N::from_canonical_usize(DIGEST_SIZE),
        );
    }

    fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        builder.assign(&self.nb_outputs, C::N::ZERO);

        builder.set(&self.sponge_state, self.nb_inputs, value);
        builder.assign(&self.nb_inputs, self.nb_inputs + C::N::ONE);

        builder
            .if_eq(
                self.nb_inputs,
                C::N::from_canonical_usize(DIGEST_SIZE),
            )
            .then(|builder| {
                self.duplexing(builder);
            })
    }

    fn observe_commitment(&mut self, builder: &mut Builder<C>, commitment: &Array<C, Felt<C::F>>) {
        for i in 0..DIGEST_SIZE {
            let element = builder.get(commitment, i);
            self.observe(builder, element);
        }
    }

    fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        let zero: Var<_> = builder.eval(C::N::ZERO);
        builder.if_ne(self.nb_inputs, zero).then_or_else(
            |builder| {
                self.clone().duplexing(builder);
            },
            |builder| {
                builder.if_eq(self.nb_outputs, zero).then(|builder| {
                    self.clone().duplexing(builder);
                });
            },
        );
        let idx: Var<_> = builder.eval(self.nb_outputs - C::N::ONE);
        let output = builder.get(&self.sponge_state, idx);
        builder.assign(&self.nb_outputs, idx);
        output
    }

    fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        let a = self.sample(builder);
        let b = self.sample(builder);
        let c = self.sample(builder);
        let d = self.sample(builder);
        builder.ext_from_base_slice(&[a, b, c, d])
    }

    fn sample_bits(&mut self, builder: &mut Builder<C>, nb_bits: RVar<C::N>) -> Array<C, Var<C::N>>
    where
        C::N: Field,
    {
        let rand_f = self.sample(builder);
        let bits = builder.num2bits_f(rand_f, C::N::bits() as u32);

        builder.range(nb_bits, bits.len()).for_each(|i, builder| {
            builder.set(&bits, i, C::N::ZERO);
        });

        bits
    }

    pub fn check_witness(&mut self, builder: &mut Builder<C>, nb_bits: usize, witness: Felt<C::F>) {
        self.observe(builder, witness);
        let element_bits = self.sample_bits(builder, RVar::from(nb_bits));
        builder.range(0, nb_bits).for_each(|i, builder| {
            let element = builder.get(&element_bits, i);
            builder.assert_var_eq(element, C::N::ZERO);
        });
    }
}

impl<C: Config> CanObserveVariable<C, Felt<C::F>> for DuplexChallengerVariable<C> {
    fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        DuplexChallengerVariable::observe(self, builder, value);
    }

    fn observe_slice(&mut self, builder: &mut Builder<C>, values: Array<C, Felt<C::F>>) {
        builder.range(0, values.len()).for_each(|i, builder| {
            let element = builder.get(&values, i);
            self.observe(builder, element);
        });
    }
}

impl<C: Config> CanSampleVariable<C, Felt<C::F>> for DuplexChallengerVariable<C> {
    fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        DuplexChallengerVariable::sample(self, builder)
    }
}

impl<C: Config> CanSampleBitsVariable<C> for DuplexChallengerVariable<C> {
    fn sample_bits(
        &mut self,
        builder: &mut Builder<C>,
        nb_bits: RVar<C::N>,
    ) -> Array<C, Var<C::N>> {
        DuplexChallengerVariable::sample_bits(self, builder, nb_bits)
    }
}

impl<C: Config> CanObserveDigest<C> for DuplexChallengerVariable<C> {
    fn observe_digest(&mut self, builder: &mut Builder<C>, commitment: DigestVariable<C>) {
        if let DigestVariable::Felt(commitment) = commitment {
            self.observe_commitment(builder, &commitment);
        } else {
            panic!("Expected a felt digest");
        }
    }
}

impl<C: Config> FeltChallenger<C> for DuplexChallengerVariable<C> {
    fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        DuplexChallengerVariable::sample_ext(self, builder)
    }
}

impl<C: Config> CanCheckWitness<C> for DuplexChallengerVariable<C> {
    fn check_witness(&mut self, builder: &mut Builder<C>, nb_bits: usize, witness: Felt<C::F>) {
        DuplexChallengerVariable::check_witness(self, builder, nb_bits, witness);
    }
}

impl<C: Config> ChallengerVariable<C> for DuplexChallengerVariable<C> {
    fn new(builder: &mut Builder<C>) -> Self {
        DuplexChallengerVariable::new(builder)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use ax_stark_sdk::{
        config::baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config},
        engine::StarkEngine,
    };
    use axvm_circuit::system::program::util::execute_program;
    use axvm_native_compiler::{
        asm::{AsmBuilder, AsmConfig},
        ir::Felt,
    };
    use rand::Rng;
    use p3_challenger::{CanObserve, CanSample};
    use p3_field::AbstractField;
    use p3_uni_stark::{StarkGenericConfig, Val};

    use super::DuplexChallengerVariable;

    fn test_compiler_challenger_with_num_challenges(num_challenges: usize) {
        let mut rng = rand::thread_rng();
        let observations = (0..num_challenges).map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..(1 << 30)))).collect::<Vec<_>>();

        type SC = BabyBearPoseidon2Config;
        type F = Val<SC>;
        type EF = <SC as StarkGenericConfig>::Challenge;

        let engine = default_engine();
        let mut challenger = engine.new_challenger();
        for observation in &observations {
            challenger.observe(*observation);
        }
        let result: F = challenger.sample();
        println!("expected result: {}", result);

        let mut builder = AsmBuilder::<F, EF>::default();

        let mut challenger = DuplexChallengerVariable::<AsmConfig<F, EF>>::new(&mut builder);
        for observation in &observations {
            let observation: Felt<_> = builder.eval(*observation);
            challenger.observe(&mut builder, observation);
        }
        let element = challenger.sample(&mut builder);

        let expected_result: Felt<_> = builder.eval(result);
        builder.assert_felt_eq(expected_result, element);

        builder.halt();

        let program = builder.compile_isa();
        execute_program(program, vec![]);
    }

    #[test]
    fn test_compiler_challenger() {
        test_compiler_challenger_with_num_challenges(1);
        test_compiler_challenger_with_num_challenges(4);
        test_compiler_challenger_with_num_challenges(8);
        test_compiler_challenger_with_num_challenges(10);
        test_compiler_challenger_with_num_challenges(16);
        test_compiler_challenger_with_num_challenges(20);
        test_compiler_challenger_with_num_challenges(50);
    }

}
