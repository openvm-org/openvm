use openvm_native_compiler::{
    ir::{RVar, DIGEST_SIZE, PERMUTATION_WIDTH},
    prelude::*,
};
use openvm_native_compiler_derive::iter_zip;
use openvm_stark_backend::p3_field::{Field, FieldAlgebra};

use crate::{
    challenger::{
        CanCheckWitness, CanObserveDigest, CanObserveVariable, CanSampleBitsVariable,
        CanSampleVariable, ChallengerVariable, FeltChallenger,
    },
    digest::DigestVariable,
};

/// Reference: [`openvm_stark_backend::p3_challenger::DuplexChallenger`]
#[derive(Clone)]
pub struct DuplexChallengerVariable<C: Config> {
    pub sponge_state: Array<C, Felt<C::F>>,

    pub input_ptr: Ptr<C::N>,
    pub output_ptr: Ptr<C::N>,
    pub io_empty_ptr: Ptr<C::N>,
    pub io_full_ptr: Ptr<C::N>,
}

impl<C: Config> DuplexChallengerVariable<C> {
    /// Creates a new duplex challenger with the default state.
    pub fn new(builder: &mut Builder<C>) -> Self {
        let sponge_state = builder.dyn_array(PERMUTATION_WIDTH);

        builder
            .range(0, sponge_state.len())
            .for_each(|i_vec, builder| {
                builder.set(&sponge_state, i_vec[0], C::F::ZERO);
            });
        let io_empty_ptr = sponge_state.ptr();
        let io_full_ptr: Ptr<_> =
            builder.eval(io_empty_ptr + C::N::from_canonical_usize(DIGEST_SIZE));
        let input_ptr = builder.eval(io_empty_ptr);
        let output_ptr = builder.eval(io_empty_ptr);

        DuplexChallengerVariable::<C> {
            sponge_state,
            input_ptr,
            output_ptr,
            io_empty_ptr,
            io_full_ptr,
        }
    }

    pub fn duplexing(&self, builder: &mut Builder<C>) {
        builder.assign(&self.input_ptr, self.io_empty_ptr);

        builder.poseidon2_permute_mut(&self.sponge_state);

        builder.assign(&self.output_ptr, self.io_full_ptr);
    }

    fn observe(&self, builder: &mut Builder<C>, value: Felt<C::F>) {
        builder.assign(&self.output_ptr, self.io_empty_ptr);

        builder.iter_ptr_set(&self.sponge_state, self.input_ptr.address.into(), value);
        builder.assign(&self.input_ptr, self.input_ptr + C::N::ONE);

        builder
            .if_eq(self.input_ptr.address, self.io_full_ptr.address)
            .then(|builder| {
                self.duplexing(builder);
            })
    }

    fn observe_commitment(&self, builder: &mut Builder<C>, commitment: &Array<C, Felt<C::F>>) {
        for i in 0..DIGEST_SIZE {
            let element = builder.get(commitment, i);
            self.observe(builder, element);
        }
    }

    fn sample(&self, builder: &mut Builder<C>) -> Felt<C::F> {
        builder
            .if_ne(self.input_ptr.address, self.io_empty_ptr.address)
            .then_or_else(
                |builder| {
                    self.duplexing(builder);
                },
                |builder| {
                    builder
                        .if_eq(self.output_ptr.address, self.io_empty_ptr.address)
                        .then(|builder| {
                            self.duplexing(builder);
                        });
                },
            );
        builder.assign(&self.output_ptr, self.output_ptr - C::N::ONE);
        builder.iter_ptr_get(&self.sponge_state, self.output_ptr.address.into())
    }

    fn sample_ext(&self, builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        let a = self.sample(builder);
        let b = self.sample(builder);
        let c = self.sample(builder);
        let d = self.sample(builder);
        builder.ext_from_base_slice(&[a, b, c, d])
    }

    fn sample_bits(&self, builder: &mut Builder<C>, nb_bits: RVar<C::N>) -> Array<C, Var<C::N>>
    where
        C::N: Field,
    {
        let rand_f = self.sample(builder);
        let bits = builder.num2bits_f(rand_f, C::N::bits() as u32);

        builder
            .range(nb_bits, bits.len())
            .for_each(|i_vec, builder| {
                builder.set(&bits, i_vec[0], C::N::ZERO);
            });
        bits
    }

    pub fn check_witness(&self, builder: &mut Builder<C>, nb_bits: usize, witness: Felt<C::F>) {
        self.observe(builder, witness);
        let element_bits = self.sample_bits(builder, RVar::from(nb_bits));
        let element_bits_truncated = element_bits.slice(builder, 0, nb_bits);
        iter_zip!(builder, element_bits_truncated).for_each(|ptr_vec, builder| {
            let element = builder.iter_ptr_get(&element_bits_truncated, ptr_vec[0]);
            builder.assert_var_eq(element, C::N::ZERO);
        });
    }
}

impl<C: Config> CanObserveVariable<C, Felt<C::F>> for DuplexChallengerVariable<C> {
    fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        DuplexChallengerVariable::observe(self, builder, value);
    }

    fn observe_slice(&mut self, builder: &mut Builder<C>, values: Array<C, Felt<C::F>>) {
        iter_zip!(builder, values).for_each(|ptr_vec, builder| {
            let element = builder.iter_ptr_get(&values, ptr_vec[0]);
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
    use openvm_native_circuit::execute_program;
    use openvm_native_compiler::{
        asm::{AsmBuilder, AsmConfig},
        ir::Felt,
    };
    use openvm_stark_backend::{
        config::{StarkGenericConfig, Val},
        p3_challenger::{CanObserve, CanSample},
        p3_field::FieldAlgebra,
    };
    use openvm_stark_sdk::{
        config::baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config},
        engine::StarkEngine,
        p3_baby_bear::BabyBear,
    };
    use rand::Rng;

    use super::DuplexChallengerVariable;

    fn test_compiler_challenger_with_num_challenges(num_challenges: usize) {
        let mut rng = rand::thread_rng();
        let observations = (0..num_challenges)
            .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..(1 << 30))))
            .collect::<Vec<_>>();

        type SC = BabyBearPoseidon2Config;
        type F = Val<SC>;
        type EF = <SC as StarkGenericConfig>::Challenge;

        let engine = default_engine();
        let mut challenger = engine.new_challenger();
        for observation in &observations {
            challenger.observe(*observation);
        }
        let result: F = challenger.sample();
        println!("expected result: {result}");

        let mut builder = AsmBuilder::<F, EF>::default();

        let challenger = DuplexChallengerVariable::<AsmConfig<F, EF>>::new(&mut builder);
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
